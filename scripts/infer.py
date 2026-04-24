from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qrouter.models import ConversationalGroundingModule, DualVisionBackbone, MambaLLMBackbone, RegionRoutingVQAModel
from qrouter.models.prompting import IGNORE_INDEX, build_zephyr_prompt
from qrouter.utils.checkpoint import maybe_load_checkpoint
from qrouter.utils.image_ops import preprocess_grounding_image


SHORT_ANSWER_SUFFIX = " Answer the question using a single word or phrase."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run QRouter inference on GQA-style or generic VQA files.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--questions", type=str, required=True, help="Path to a JSON/JSONL evaluation file.")
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--out-jsonl", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--sam2-checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append-short-answer-instruction", action="store_true")
    return parser.parse_args()


def dtype_from_precision(precision: str, device: str) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16 if device == "cuda" else None
    if precision == "bf16":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if device == "cuda" else None
    raise ValueError(f"Unsupported precision `{precision}`.")


def normalize_answer(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        examples = []
        for question_id, item in payload.items():
            examples.append(
                {
                    "question_id": str(question_id),
                    "image": item.get("image") or item.get("imageId"),
                    "question": item["question"],
                    "answer": item.get("answer", ""),
                }
            )
        return examples
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported evaluation file format.")


def resolve_image_path(image_root: str | Path, image_ref: str) -> Path:
    root = Path(image_root)
    candidate = Path(str(image_ref))
    if candidate.is_absolute() and candidate.exists():
        return candidate
    suffixes = ["", ".jpg", ".jpeg", ".png"]
    for suffix in suffixes:
        current = root / f"{image_ref}{suffix}"
        if current.exists():
            return current.resolve()
    raise FileNotFoundError(f"Could not resolve image `{image_ref}` under `{root}`.")


def build_model_from_checkpoint(checkpoint_path: str | Path, sam2_config: str, sam2_checkpoint: str, device: torch.device) -> RegionRoutingVQAModel:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get("args", {})
    model = RegionRoutingVQAModel(
        grounding_module=ConversationalGroundingModule(
            sam2_cfg=sam2_config,
            sam2_ckpt=sam2_checkpoint,
            qwen_id=config.get("grounding_qwen_id", "Qwen/Qwen2.5-VL-3B-Instruct"),
            device=str(device),
            use_lora=True,
        ),
        vision_backbone=DualVisionBackbone(vision_backbone_id=config.get("vision_backbone_id", "dinosiglip-vit-so-384px")),
        llm_backbone=MambaLLMBackbone(
            llm_id=config.get("llm_id", "mamba-2.8b-zephyr"),
            max_length=int(config.get("max_length", 2048)),
        ),
        num_region_tokens=int(config.get("num_region_tokens", 32)),
        num_context_tokens=int(config.get("num_context_tokens", 128)),
        alignment_loss_weight=float(config.get("alignment_loss_weight", 0.1)),
        compactness_loss_weight=float(config.get("compactness_loss_weight", 0.01)),
        diversity_loss_weight=float(config.get("diversity_loss_weight", 0.01)),
        segmentation_loss_weight=float(config.get("lambda_dice", 0.25)),
    )
    model.region_tokenizer.confidence_threshold = float(config.get("confidence_threshold", 0.5))
    model.set_stage(config.get("stage", "stage2"))
    maybe_load_checkpoint(checkpoint_path=checkpoint_path, model=model)
    model.to(device)
    model.eval()
    return model


def validate_path_argument(name: str, value: str) -> None:
    if "your/path/to" in value:
        raise ValueError(f"`{name}` still points to a placeholder path: {value}")
    if not Path(value).exists():
        raise FileNotFoundError(f"`{name}` does not exist: {value}")


def build_eval_prompt(question: str, append_short_answer_instruction: bool) -> str:
    normalized = question.strip()
    if append_short_answer_instruction:
        normalized = f"{normalized}{SHORT_ANSWER_SUFFIX}"
    prompt_text, _ = build_zephyr_prompt(question=normalized, answer=None)
    return prompt_text


def tokenize_question(model: RegionRoutingVQAModel, prompt_text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = model.llm_backbone.tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    return input_ids, attention_mask, labels


@torch.no_grad()
def generate_answer(
    model: RegionRoutingVQAModel,
    image_path: Path,
    question: str,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_new_tokens: int,
    append_short_answer_instruction: bool,
) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        pixel_values = model.vision_backbone.image_transform(image)
        grounding_image = preprocess_grounding_image(image)

    prompt_text = build_eval_prompt(question, append_short_answer_instruction=append_short_answer_instruction)
    input_ids, attention_mask, labels = tokenize_question(model, prompt_text, device=device)

    pixel_values = {key: value.unsqueeze(0).to(device) for key, value in pixel_values.items()}
    grounding_images = grounding_image.unsqueeze(0).to(device)

    amp_context = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device.type == "cuda" and amp_dtype is not None
        else contextlib.nullcontext()
    )

    with amp_context:
        vision_outputs = model.vision_backbone(pixel_values)
        question_embed = model.llm_backbone.encode_questions(
            input_ids=input_ids,
            attention_mask=((labels == IGNORE_INDEX) & attention_mask.bool()).long(),
        )
        grounding_outputs = model.grounding_module(
            images=grounding_images,
            questions=[question],
            image_paths=[str(image_path)],
        )
        region_outputs = model.region_tokenizer(
            patch_tokens=vision_outputs["patch_tokens"],
            patch_hw=vision_outputs["patch_hw"],
            masks=grounding_outputs.masks,
            scores=grounding_outputs.scores,
            question_embed=question_embed,
        )
        visual_embeds = model.projector(region_outputs.visual_tokens)

    generated_ids = input_ids.clone()
    eos_token_id = model.llm_backbone.tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        with amp_context:
            text_embeddings = model.llm_backbone.embed_input_ids(generated_ids)
            fused_embeddings = torch.cat([visual_embeds, text_embeddings], dim=1)
            outputs = model.llm_backbone(
                inputs_embeds=fused_embeddings,
                attention_mask=torch.ones(fused_embeddings.shape[:2], dtype=torch.long, device=device),
                labels=None,
            )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
            break

    prompt_ids = model.llm_backbone.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    decoded = model.llm_backbone.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    decoded_prompt = model.llm_backbone.tokenizer.decode(prompt_ids, skip_special_tokens=False)
    answer = decoded[len(decoded_prompt):] if decoded.startswith(decoded_prompt) else decoded
    answer = answer.replace("<|assistant|>", " ").replace("<|user|>", " ").replace("<|endoftext|>", " ")
    return " ".join(answer.replace("\n", " ").split()).strip()


def main() -> None:
    args = parse_args()
    validate_path_argument("checkpoint", args.checkpoint)
    validate_path_argument("questions", args.questions)
    validate_path_argument("image_root", args.image_root)
    validate_path_argument("sam2_config", args.sam2_config)
    validate_path_argument("sam2_checkpoint", args.sam2_checkpoint)
    device = torch.device(args.device)
    amp_dtype = dtype_from_precision(args.precision, args.device)

    examples = load_examples(args.questions)
    if args.limit is not None:
        examples = examples[: args.limit]

    checkpoint_path = Path(args.checkpoint).resolve()
    out_path = Path(args.out_jsonl).resolve()
    if out_path.exists():
        out_path.unlink()
    summary_path = Path(args.summary_json).resolve() if args.summary_json else out_path.with_suffix(".summary.json")

    model = build_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        device=device,
    )

    scored = 0
    correct = 0
    for index, example in enumerate(examples, start=1):
        image_ref = str(example.get("image", "")).strip()
        if not image_ref:
            raise ValueError("Each evaluation example must contain `image` or `imageId`.")
        image_path = resolve_image_path(args.image_root, image_ref)
        question = str(example["question"]).strip()
        ground_truth = str(example.get("answer", "")).strip()
        prediction = generate_answer(
            model=model,
            image_path=image_path,
            question=question,
            device=device,
            amp_dtype=amp_dtype,
            max_new_tokens=args.max_new_tokens,
            append_short_answer_instruction=args.append_short_answer_instruction,
        )
        is_correct = False
        if ground_truth:
            is_correct = normalize_answer(prediction) == normalize_answer(ground_truth)
            scored += 1
            correct += int(is_correct)
        record = {
            "question_id": str(example.get("question_id", index)),
            "image_id": image_ref,
            "image_path": str(image_path),
            "question": question,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
        }
        append_jsonl(out_path, record)
        print(f"[{index}/{len(examples)}] prediction={prediction!r}", flush=True)

    summary = {
        "checkpoint": str(checkpoint_path),
        "num_results": len(examples),
        "num_scored": scored,
        "num_correct": correct,
        "accuracy": (correct / scored) if scored > 0 else None,
        "metric": "normalized_exact_match",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"checkpoint: {checkpoint_path}")
    print(f"out_jsonl: {out_path}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
