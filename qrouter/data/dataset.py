import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from qrouter.models.prompting import build_grounding_prompt, encode_prompt_only, encode_vqa_example
from qrouter.utils.image_ops import (
    GROUNDING_IMAGE_SIZE,
    auto_detect_dataset_format,
    load_mask_image,
    load_pil_rgb,
    preprocess_grounding_image,
    preprocess_grounding_mask,
    resolve_path,
)


@dataclass
class CanonicalSample:
    image_path: str
    question: str
    grounding_prompt: str
    answer: str = ""
    grounding_mask_path: Optional[str] = None
    task_type: str = "qa"
    metadata: Optional[dict[str, Any]] = None


def _load_json_payload(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        if str(path).lower().endswith(".jsonl"):
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def _canonicalize_jsonl(
    payload: list[dict[str, Any]],
    image_root: Optional[str | Path],
    task_type: str,
) -> list[CanonicalSample]:
    samples = []
    for item in payload:
        if "image" not in item:
            raise ValueError("JSONL entries must contain an `image` field.")
        if task_type == "qa":
            if "question" not in item or "answer" not in item:
                raise ValueError("QA jsonl entries must contain `image`, `question`, and `answer`.")
            question = str(item["question"]).strip()
            answer = str(item["answer"]).strip()
        elif task_type == "grounding":
            raw_prompt = item.get("prompt", item.get("question"))
            if raw_prompt is None:
                raise ValueError("Grounding jsonl entries must contain `prompt` or `question`.")
            question = str(raw_prompt).strip()
            answer = str(item.get("answer", "")).strip()
        else:
            raise ValueError(f"Unsupported task_type `{task_type}`.")
        grounding_mask_path = item.get("grounding_mask") or item.get("mask")
        samples.append(
            CanonicalSample(
                image_path=resolve_path(image_root, item["image"]),
                question=question,
                answer=answer,
                grounding_prompt=item.get("grounding_prompt", question).strip(),
                grounding_mask_path=resolve_path(image_root, grounding_mask_path) if grounding_mask_path else None,
                task_type=task_type,
                metadata=item.get("metadata", {"concept": item.get("concept")}),
            )
        )
    return samples


def _extract_last_turn_example(item: dict[str, Any], image_root: Optional[str | Path], grounding_history_turns: int) -> CanonicalSample:
    if "image" not in item:
        raise ValueError("This project only supports image-grounded samples. Missing `image` field.")
    conversations = item.get("conversations", [])
    normalized = []
    for turn in conversations:
        role = (turn.get("from") or turn.get("role") or "").lower()
        value = str(turn.get("value") or turn.get("content") or "").replace("<image>", "").strip()
        if role in {"human", "user"} and value:
            normalized.append(("human", value))
        elif role in {"gpt", "assistant"} and value:
            normalized.append(("assistant", value))
    if len(normalized) < 2:
        raise ValueError("LLaVA sample does not contain a valid question/answer pair.")
    answer_idx = None
    for idx in range(len(normalized) - 1, -1, -1):
        if normalized[idx][0] == "assistant":
            answer_idx = idx
            break
    if answer_idx is None or answer_idx == 0 or normalized[answer_idx - 1][0] != "human":
        raise ValueError("Unable to find the final human->assistant turn pair in conversation.")
    question = normalized[answer_idx - 1][1]
    answer = normalized[answer_idx][1]
    historical_humans = [text for role, text in normalized[: answer_idx - 1] if role == "human"]
    grounding_prompt = build_grounding_prompt(question, history=historical_humans, max_history=grounding_history_turns)
    grounding_mask_path = item.get("grounding_mask") or item.get("mask")
    return CanonicalSample(
        image_path=resolve_path(image_root, item["image"]),
        question=question,
        answer=answer,
        grounding_prompt=grounding_prompt,
        grounding_mask_path=resolve_path(image_root, grounding_mask_path) if grounding_mask_path else None,
        task_type="qa",
        metadata={"raw_conversations": conversations, "id": item.get("id")},
    )


def _canonicalize_llava(payload: list[dict[str, Any]], image_root: Optional[str | Path], grounding_history_turns: int) -> list[CanonicalSample]:
    return [_extract_last_turn_example(item, image_root=image_root, grounding_history_turns=grounding_history_turns) for item in payload]


class CanonicalVQADataset(Dataset):
    def __init__(
        self,
        samples: list[CanonicalSample],
        tokenizer: PreTrainedTokenizerBase,
        vision_transform,
        max_length: int,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.vision_transform = vision_transform
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = load_pil_rgb(sample.image_path)
        pixel_values = self.vision_transform(image)
        grounding_image = preprocess_grounding_image(image, image_size=GROUNDING_IMAGE_SIZE)
        mask_image = load_mask_image(sample.grounding_mask_path)
        grounding_mask = preprocess_grounding_mask(mask_image, image_size=GROUNDING_IMAGE_SIZE) if mask_image else None

        prompt_encoding = encode_vqa_example(
            tokenizer=self.tokenizer,
            question=sample.question,
            answer=sample.answer,
            max_length=self.max_length,
        ) if sample.task_type == "qa" else encode_prompt_only(
            tokenizer=self.tokenizer,
            question=sample.question,
            max_length=self.max_length,
        )
        return {
            "pixel_values": pixel_values,
            "grounding_images": grounding_image,
            "grounding_masks": grounding_mask,
            "input_ids": prompt_encoding.input_ids,
            "attention_mask": prompt_encoding.attention_mask,
            "labels": prompt_encoding.labels,
            "question_text": sample.question,
            "grounding_prompt": sample.grounding_prompt,
            "answer_text": sample.answer,
            "image_path": sample.image_path,
            "task_type": sample.task_type,
            "metadata": sample.metadata or {},
        }


def build_dataset(
    dataset_format: str,
    data_path: str | Path,
    image_root: Optional[str | Path],
    tokenizer: PreTrainedTokenizerBase,
    vision_transform,
    max_length: int,
    grounding_history_turns: int = 1,
    task_type: str = "qa",
) -> CanonicalVQADataset:
    if dataset_format == "auto":
        dataset_format = auto_detect_dataset_format(data_path)
    payload = _load_json_payload(data_path)
    if dataset_format == "jsonl":
        samples = _canonicalize_jsonl(payload, image_root=image_root, task_type=task_type)
    elif dataset_format == "llava":
        if task_type != "qa":
            raise ValueError("The llava dataset format is only supported for QA training.")
        samples = _canonicalize_llava(payload, image_root=image_root, grounding_history_turns=grounding_history_turns)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
    return CanonicalVQADataset(
        samples=samples,
        tokenizer=tokenizer,
        vision_transform=vision_transform,
        max_length=max_length,
    )


QRouterDataset = CanonicalVQADataset
