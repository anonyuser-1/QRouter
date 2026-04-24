import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize


ImageFile.LOAD_TRUNCATED_IMAGES = True

GROUNDING_IMAGE_SIZE = 1024
VISION_IMAGE_SIZE = 384
IGNORE_INDEX = -100


@lru_cache(maxsize=8192)
def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def load_pil_rgb(path: str | Path) -> Image.Image:
    image = Image.open(io.BytesIO(_read_file_bytes(str(path))))
    image.info.pop("icc_profile", None)
    return image.convert("RGB")


def load_mask_image(path: Optional[str | Path]) -> Optional[Image.Image]:
    if path is None:
        return None
    mask = Image.open(io.BytesIO(_read_file_bytes(str(path))))
    mask.info.pop("icc_profile", None)
    return mask.convert("L")


def resize_longest_side(image: Image.Image, target: int, interpolation: InterpolationMode) -> Image.Image:
    width, height = image.size
    scale = float(target) / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return resize(image, [new_height, new_width], interpolation=interpolation, antialias=True)


def pad_to_square(image: Image.Image, target: int, fill: int | tuple[int, int, int] = 0) -> Image.Image:
    width, height = image.size
    canvas = Image.new(image.mode, (target, target), color=fill)
    left = (target - width) // 2
    top = (target - height) // 2
    canvas.paste(image, (left, top))
    return canvas


def preprocess_grounding_image(image: Image.Image, image_size: int = GROUNDING_IMAGE_SIZE) -> torch.Tensor:
    resized = resize_longest_side(image, image_size, InterpolationMode.BILINEAR)
    squared = pad_to_square(resized, image_size, fill=(0, 0, 0))
    tensor = pil_to_tensor(squared).float() / 255.0
    return tensor


def preprocess_grounding_mask(mask: Image.Image, image_size: int = GROUNDING_IMAGE_SIZE) -> torch.Tensor:
    resized = resize_longest_side(mask, image_size, InterpolationMode.NEAREST)
    squared = pad_to_square(resized, image_size, fill=0)
    tensor = pil_to_tensor(squared).float() / 255.0
    return (tensor > 0.5).float()


def resolve_path(image_root: Optional[str | Path], path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    if image_root is None:
        return str(candidate)
    return str((Path(image_root) / candidate).resolve())


def auto_detect_dataset_format(data_path: str | Path) -> str:
    path = Path(data_path)
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and payload and "conversations" in payload[0]:
            return "llava"
        if isinstance(payload, list) and payload and "question" in payload[0]:
            return "jsonl"
    raise ValueError(f"Unable to infer dataset format from `{path}`.")


def pad_last_dim(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_length = max(tensor.shape[0] for tensor in tensors)
    output = torch.full((len(tensors), max_length), pad_value, dtype=tensors[0].dtype)
    for idx, tensor in enumerate(tensors):
        output[idx, : tensor.shape[0]] = tensor
    return output


def to_bool_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return (mask > 0.5).float()


def batched_index_select(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(features.size(0), device=features.device).unsqueeze(-1)
    return features[batch_indices, indices]


def safe_normalize(weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    return weights / denom


def maybe_stack(items: list[Any]) -> Any:
    if items and isinstance(items[0], torch.Tensor):
        return torch.stack(items, dim=0)
    if items and isinstance(items[0], dict):
        return {key: torch.stack([sample[key] for sample in items], dim=0) for key in items[0]}
    return items


def interpolate_mask(masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(masks, size=size, mode="bilinear", align_corners=False)
