from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qrouter.data import QRouterBatchCollator, build_dataset
from qrouter.models import (
    ConversationalGroundingModule,
    DualVisionBackbone,
    MambaLLMBackbone,
    RegionRoutingVQAModel,
)
from qrouter.utils.checkpoint import maybe_load_checkpoint, save_checkpoint
from qrouter.utils.train_utils import format_metrics, move_batch_to_device


DEFAULT_CONFIG: dict[str, Any] = {
    "stage": "stage1",
    "qa_dataset_format": "auto",
    "cis_dataset_format": "jsonl",
    "qa_data_path": "your/path/to/qa_train.jsonl",
    "qa_image_root": "your/path/to/images",
    "cis_data_path": None,
    "cis_image_root": None,
    "output_dir": "outputs/qrouter_stage1",
    "resume": None,
    "init_checkpoint": None,
    "llm_id": "mamba-2.8b-zephyr",
    "vision_backbone_id": "dinosiglip-vit-so-384px",
    "grounding_qwen_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "sam2_config": "your/path/to/sam2_hiera_l.yaml",
    "sam2_checkpoint": "your/path/to/sam2_hiera_large.pt",
    "batch_size": 16,
    "num_workers": 4,
    "grad_accum_steps": 1,
    "learning_rate": 2e-5,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.03,
    "max_steps": 20000,
    "save_every": 1000,
    "log_every": 50,
    "val_every": 1000,
    "qa_val_ratio": 0.01,
    "seed": 42,
    "max_length": 2048,
    "num_region_tokens": 32,
    "num_context_tokens": 128,
    "confidence_threshold": 0.5,
    "alignment_loss_weight": 0.1,
    "compactness_loss_weight": 0.01,
    "diversity_loss_weight": 0.01,
    "lambda_dice": 0.25,
    "grounding_history_turns": 1,
    "precision": "bf16",
    "gradient_checkpointing": True,
    "qa_to_cis_ratio": [2, 1],
}

STAGE_DEFAULTS = {
    "stage1": {
        "max_steps": 20000,
        "output_dir": "outputs/qrouter_stage1",
    },
    "stage2": {
        "max_steps": 10000,
        "output_dir": "outputs/qrouter_stage2",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train QRouter with Stage I / Stage II scheduling.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument("--stage", choices=["stage1", "stage2"], default=None)
    parser.add_argument("--qa-data-path", type=str, default=None)
    parser.add_argument("--qa-image-root", type=str, default=None)
    parser.add_argument("--cis-data-path", type=str, default=None)
    parser.add_argument("--cis-image-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--sam2-config", type=str, default=None)
    parser.add_argument("--sam2-checkpoint", type=str, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    merged = deepcopy(DEFAULT_CONFIG)
    merged.update(config)
    for key, value in {
        "stage": args.stage,
        "qa_data_path": args.qa_data_path,
        "qa_image_root": args.qa_image_root,
        "cis_data_path": args.cis_data_path,
        "cis_image_root": args.cis_image_root,
        "output_dir": args.output_dir,
        "resume": args.resume,
        "init_checkpoint": args.init_checkpoint,
        "sam2_config": args.sam2_config,
        "sam2_checkpoint": args.sam2_checkpoint,
    }.items():
        if value is not None:
            merged[key] = value

    stage = merged["stage"]
    if stage not in STAGE_DEFAULTS:
        raise ValueError(f"Unsupported stage `{stage}`.")
    for key, value in STAGE_DEFAULTS[stage].items():
        if key not in config:
            merged[key] = value

    if stage == "stage2" and not merged.get("cis_data_path"):
        raise ValueError("Stage II requires `cis_data_path` in the config or CLI.")
    return merged


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def log_rank0(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def setup_distributed() -> torch.device:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def split_indices(dataset_size: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"`qa_val_ratio` must be in [0, 1), got {val_ratio}.")
    if dataset_size <= 1 or val_ratio == 0.0:
        return list(range(dataset_size)), []
    val_size = max(1, int(round(dataset_size * val_ratio)))
    if val_size >= dataset_size:
        val_size = dataset_size - 1
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()
    return permutation[val_size:], permutation[:val_size]


def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool):
    sampler = None
    if is_distributed():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=QRouterBatchCollator(),
        persistent_workers=num_workers > 0,
    )
    return sampler, loader


class CyclingLoader:
    def __init__(self, dataloader: DataLoader, sampler: Optional[DistributedSampler]) -> None:
        self.dataloader = dataloader
        self.sampler = sampler
        self.epoch = 0
        self.iterator = None
        self.reset()

    def reset(self) -> None:
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)
        self.epoch += 1
        self.iterator = iter(self.dataloader)

    def next(self):
        if self.iterator is None:
            self.reset()
        try:
            return next(self.iterator)
        except StopIteration:
            self.reset()
            return next(self.iterator)


def build_scheduler(optimizer: torch.optim.Optimizer, max_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup_steps = int(max_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_optimizer(model, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def precision_to_amp_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else None
    raise ValueError(f"Unsupported precision `{precision}`.")


def autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


def save_named_checkpoint(
    output_dir: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    step: int,
    stage: str,
    config: dict[str, Any],
    filename: str,
) -> Path:
    checkpoint_path = save_checkpoint(
        output_dir=output_dir,
        model=unwrap_model(model),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=step,
        stage=stage,
        args_dict=config,
    )
    target_path = output_dir / filename
    shutil.copy2(checkpoint_path, target_path)
    return target_path


@torch.no_grad()
def run_validation(model, dataloader: DataLoader, sampler: Optional[DistributedSampler], device: torch.device, amp_dtype: Optional[torch.dtype]) -> dict[str, float]:
    if dataloader is None:
        return {"validation_loss": float("nan"), "validation_answer_loss": float("nan"), "num_samples": 0}
    if sampler is not None:
        sampler.set_epoch(0)
    was_training = model.training
    model.eval()

    total_loss = torch.zeros(1, device=device, dtype=torch.float64)
    total_answer_loss = torch.zeros(1, device=device, dtype=torch.float64)
    total_samples = torch.zeros(1, device=device, dtype=torch.float64)

    for batch in dataloader:
        batch = move_batch_to_device(batch, device=device)
        with autocast_context(device, amp_dtype):
            output = model(batch)
        batch_size = batch["input_ids"].shape[0]
        total_loss += output.loss.detach().float().to(torch.float64) * batch_size
        total_answer_loss += output.metrics["answer_loss"].detach().float().to(torch.float64) * batch_size
        total_samples += batch_size

    if is_distributed():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_answer_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    denom = total_samples.clamp_min(1.0)
    metrics = {
        "validation_loss": (total_loss / denom).item(),
        "validation_answer_loss": (total_answer_loss / denom).item(),
        "num_samples": int(total_samples.item()),
    }
    if was_training:
        model.train()
    return metrics


def build_model(config: dict[str, Any], device: torch.device) -> RegionRoutingVQAModel:
    vision_backbone = DualVisionBackbone(vision_backbone_id=config["vision_backbone_id"])
    llm_backbone = MambaLLMBackbone(
        llm_id=config["llm_id"],
        max_length=int(config["max_length"]),
    )
    if bool(config.get("gradient_checkpointing", False)):
        llm_backbone.enable_gradient_checkpointing()

    grounding_module = ConversationalGroundingModule(
        sam2_cfg=config["sam2_config"],
        sam2_ckpt=config["sam2_checkpoint"],
        qwen_id=config["grounding_qwen_id"],
        device=str(device),
        use_lora=True,
    )
    model = RegionRoutingVQAModel(
        grounding_module=grounding_module,
        vision_backbone=vision_backbone,
        llm_backbone=llm_backbone,
        num_region_tokens=int(config["num_region_tokens"]),
        num_context_tokens=int(config["num_context_tokens"]),
        alignment_loss_weight=float(config["alignment_loss_weight"]),
        compactness_loss_weight=float(config["compactness_loss_weight"]),
        diversity_loss_weight=float(config["diversity_loss_weight"]),
        segmentation_loss_weight=float(config["lambda_dice"]),
    )
    model.region_tokenizer.confidence_threshold = float(config["confidence_threshold"])
    model.set_stage(config["stage"])
    model.to(device)
    return model


def validate_path_argument(name: str, value: Optional[str], should_exist: bool = True) -> None:
    if value is None:
        raise ValueError(f"`{name}` is required but was not provided.")
    if "your/path/to" in value:
        raise ValueError(f"`{name}` still points to a placeholder path: {value}")
    if should_exist and not Path(value).exists():
        raise FileNotFoundError(f"`{name}` does not exist: {value}")


def validate_config(config: dict[str, Any]) -> None:
    validate_path_argument("qa_data_path", config["qa_data_path"])
    validate_path_argument("qa_image_root", config["qa_image_root"])
    validate_path_argument("sam2_config", config["sam2_config"])
    validate_path_argument("sam2_checkpoint", config["sam2_checkpoint"])
    if config["stage"] == "stage2":
        validate_path_argument("cis_data_path", config["cis_data_path"])
        validate_path_argument("cis_image_root", config["cis_image_root"] or config["qa_image_root"])
    if config.get("init_checkpoint"):
        validate_path_argument("init_checkpoint", config["init_checkpoint"])
    if config.get("resume"):
        validate_path_argument("resume", config["resume"])


def build_datasets(config: dict[str, Any], model: RegionRoutingVQAModel):
    qa_dataset = build_dataset(
        dataset_format=config.get("qa_dataset_format", "auto"),
        data_path=config["qa_data_path"],
        image_root=config["qa_image_root"],
        tokenizer=model.llm_backbone.tokenizer,
        vision_transform=model.vision_backbone.image_transform,
        max_length=int(config["max_length"]),
        grounding_history_turns=int(config["grounding_history_turns"]),
        task_type="qa",
    )
    train_indices, val_indices = split_indices(len(qa_dataset), float(config["qa_val_ratio"]), int(config["seed"]))
    qa_train = Subset(qa_dataset, train_indices)
    qa_val = Subset(qa_dataset, val_indices) if val_indices else None

    cis_dataset = None
    if config["stage"] == "stage2":
        cis_dataset = build_dataset(
            dataset_format=config.get("cis_dataset_format", "jsonl"),
            data_path=config["cis_data_path"],
            image_root=config["cis_image_root"] or config["qa_image_root"],
            tokenizer=model.llm_backbone.tokenizer,
            vision_transform=model.vision_backbone.image_transform,
            max_length=int(config["max_length"]),
            grounding_history_turns=0,
            task_type="grounding",
        )
    return qa_train, qa_val, cis_dataset


def next_stage2_task(step_index: int, qa_ratio: int, cis_ratio: int) -> str:
    pattern = ["qa"] * qa_ratio + ["grounding"] * cis_ratio
    return pattern[step_index % len(pattern)]


def main() -> None:
    args = parse_args()
    config = load_config(args)
    validate_config(config)
    device = setup_distributed()
    set_seed(int(config["seed"]) + get_rank())

    output_dir = Path(config["output_dir"])
    if is_main_process():
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
        with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

    log_rank0(f"[init] stage={config['stage']} output_dir={output_dir}")
    model = build_model(config, device=device)
    qa_train, qa_val, cis_dataset = build_datasets(config, model)

    qa_train_sampler, qa_train_loader = build_loader(
        qa_train,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        shuffle=True,
    )
    qa_val_sampler, qa_val_loader = build_loader(
        qa_val,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        shuffle=False,
    ) if qa_val is not None else (None, None)

    cis_sampler, cis_loader = (None, None)
    if cis_dataset is not None:
        cis_sampler, cis_loader = build_loader(
            cis_dataset,
            batch_size=int(config["batch_size"]),
            num_workers=int(config["num_workers"]),
            shuffle=True,
        )

    training_model = model
    if is_distributed():
        training_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)

    optimizer = build_optimizer(
        training_model,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = build_scheduler(
        optimizer,
        max_steps=int(config["max_steps"]),
        warmup_ratio=float(config["warmup_ratio"]),
    )
    amp_dtype = precision_to_amp_dtype(str(config["precision"]))
    use_scaler = amp_dtype == torch.float16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    start_step = 0
    if config.get("init_checkpoint"):
        maybe_load_checkpoint(
            checkpoint_path=config["init_checkpoint"],
            model=unwrap_model(training_model),
        )
    if config.get("resume"):
        start_step = maybe_load_checkpoint(
            checkpoint_path=config["resume"],
            model=unwrap_model(training_model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_scaler else None,
        )

    qa_train_cycle = CyclingLoader(qa_train_loader, qa_train_sampler)
    cis_cycle = CyclingLoader(cis_loader, cis_sampler) if cis_loader is not None else None

    qa_ratio, cis_ratio = config["qa_to_cis_ratio"]
    best_validation_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)

    log_rank0(
        "[data] "
        f"qa_train={len(qa_train)} qa_val={0 if qa_val is None else len(qa_val)} "
        f"cis_train={0 if cis_dataset is None else len(cis_dataset)} "
        f"batch_size={config['batch_size']} grad_accum={config['grad_accum_steps']} max_steps={config['max_steps']}"
    )

    try:
        for global_step in range(start_step, int(config["max_steps"])):
            micro_losses = []
            last_output = None
            for accum_index in range(int(config["grad_accum_steps"])):
                if config["stage"] == "stage2" and cis_cycle is not None:
                    task_name = next_stage2_task(global_step * int(config["grad_accum_steps"]) + accum_index, int(qa_ratio), int(cis_ratio))
                    batch = qa_train_cycle.next() if task_name == "qa" else cis_cycle.next()
                else:
                    task_name = "qa"
                    batch = qa_train_cycle.next()

                batch = move_batch_to_device(batch, device=device)
                sync_context = (
                    training_model.no_sync()
                    if is_distributed() and accum_index < int(config["grad_accum_steps"]) - 1
                    else contextlib.nullcontext()
                )
                with sync_context:
                    with autocast_context(device, amp_dtype):
                        output = training_model(batch)
                        loss = output.loss / int(config["grad_accum_steps"])
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                micro_losses.append(float(output.loss.detach().item()))
                last_output = output

            if use_scaler:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(training_model.parameters(), max_norm=float(config["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(training_model.parameters(), max_norm=float(config["max_grad_norm"]))
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            step_number = global_step + 1
            metrics = {key: value.detach().float().cpu() for key, value in last_output.metrics.items()}
            learning_rate = scheduler.get_last_lr()[0]

            if is_main_process() and step_number % int(config["log_every"]) == 0:
                print(
                        f"[step {step_number}] "
                        f"lr={learning_rate:.6e} "
                        f"loss={np.mean(micro_losses):.4f} "
                        f"grad_norm={float(grad_norm):.4f} "
                        f"{format_metrics(metrics)}",
                        flush=True,
                    )

            if is_main_process() and step_number % int(config["save_every"]) == 0:
                save_named_checkpoint(
                    output_dir=output_dir / "checkpoints",
                    model=training_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if use_scaler else None,
                    step=step_number,
                    stage=config["stage"],
                    config=config,
                    filename="latest.pt",
                )

            if qa_val_loader is not None and step_number % int(config["val_every"]) == 0:
                validation_metrics = run_validation(
                    model=training_model,
                    dataloader=qa_val_loader,
                    sampler=qa_val_sampler,
                    device=device,
                    amp_dtype=amp_dtype,
                )
                if is_main_process():
                    print(
                        f"[val {step_number}] "
                        f"validation_loss={validation_metrics['validation_loss']:.4f} "
                        f"validation_answer_loss={validation_metrics['validation_answer_loss']:.4f}",
                        flush=True,
                    )
                    save_named_checkpoint(
                        output_dir=output_dir / "checkpoints",
                        model=training_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler if use_scaler else None,
                        step=step_number,
                        stage=config["stage"],
                        config=config,
                        filename="latest.pt",
                    )
                    if validation_metrics["validation_loss"] < best_validation_loss:
                        best_validation_loss = validation_metrics["validation_loss"]
                        save_named_checkpoint(
                            output_dir=output_dir / "checkpoints",
                            model=training_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler if use_scaler else None,
                            step=step_number,
                            stage=config["stage"],
                            config=config,
                            filename="best.pt",
                        )

        if is_main_process():
            save_named_checkpoint(
                output_dir=output_dir / "checkpoints",
                model=training_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if use_scaler else None,
                step=int(config["max_steps"]),
                stage=config["stage"],
                config=config,
                filename="final.pt",
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
