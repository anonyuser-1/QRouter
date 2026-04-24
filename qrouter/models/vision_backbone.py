from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from torchvision.transforms import Compose, Resize


VISION_BACKBONES = {
    "dinosiglip-vit-so-384px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_384",
        "image_size": 384,
        "patch_size": 14,
    }
}


def unpack_tuple(fn):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, (tuple, list)):
            return result[0]
        return result

    return wrapper


def normalize_patch_tokens(tokens: torch.Tensor | list | tuple, num_patches: int) -> torch.Tensor:
    if isinstance(tokens, (list, tuple)):
        if len(tokens) == 0:
            raise ValueError("Vision backbone returned an empty token container.")
        tokens = tokens[0]
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"Expected Tensor patch tokens, got {type(tokens).__name__}.")
    if tokens.ndim != 3:
        raise ValueError(f"Expected patch tokens with shape [B, N, D], got {tuple(tokens.shape)}.")
    if tokens.shape[1] == num_patches + 1:
        tokens = tokens[:, 1:, :]
    return tokens


@dataclass
class DinoSigLIPTransform:
    dino_transform: Compose
    siglip_transform: Compose

    def __call__(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        return {
            "dino": self.dino_transform(image),
            "siglip": self.siglip_transform(image),
        }


class DualVisionBackbone(nn.Module):
    def __init__(self, vision_backbone_id: str = "dinosiglip-vit-so-384px", image_resize_strategy: str = "resize-naive") -> None:
        super().__init__()
        if vision_backbone_id not in VISION_BACKBONES:
            raise ValueError(f"Unsupported vision backbone `{vision_backbone_id}`.")
        spec = VISION_BACKBONES[vision_backbone_id]
        self.identifier = vision_backbone_id
        self.image_resize_strategy = image_resize_strategy
        self.image_size = spec["image_size"]
        self.patch_size = spec["patch_size"]

        # DINOv2 and SigLIP are loaded from their official public releases.
        # We thank the original authors for releasing these pretrained vision encoders.
        self.dino_featurizer: VisionTransformer = timm.create_model(
            spec["dino"],
            pretrained=True,
            num_classes=0,
            img_size=self.image_size,
        )
        self.siglip_featurizer: VisionTransformer = timm.create_model(
            spec["siglip"],
            pretrained=True,
            num_classes=0,
            img_size=self.image_size,
        )
        self.dino_featurizer.eval()
        self.siglip_featurizer.eval()
        self.dino_featurizer.forward = unpack_tuple(
            partial(self.dino_featurizer.get_intermediate_layers, n={len(self.dino_featurizer.blocks) - 2})
        )
        self.siglip_featurizer.forward = unpack_tuple(
            partial(self.siglip_featurizer.get_intermediate_layers, n={len(self.siglip_featurizer.blocks) - 2})
        )

        dino_cfg = timm.data.resolve_model_data_config(self.dino_featurizer)
        dino_cfg["input_size"] = (3, self.image_size, self.image_size)
        siglip_cfg = timm.data.resolve_model_data_config(self.siglip_featurizer)
        siglip_cfg["input_size"] = (3, self.image_size, self.image_size)

        dino_transform = timm.data.create_transform(**dino_cfg, is_training=False)
        siglip_transform = timm.data.create_transform(**siglip_cfg, is_training=False)
        if image_resize_strategy != "resize-naive":
            raise ValueError("Only `resize-naive` is enabled in the first training code release.")
        dino_transform = Compose(
            [Resize((self.image_size, self.image_size), interpolation=dino_transform.transforms[0].interpolation), *dino_transform.transforms[1:]]
        )
        siglip_transform = Compose(
            [Resize((self.image_size, self.image_size), interpolation=siglip_transform.transforms[0].interpolation), *siglip_transform.transforms[1:]]
        )
        self.image_transform = DinoSigLIPTransform(dino_transform=dino_transform, siglip_transform=siglip_transform)

        for parameter in self.parameters():
            parameter.requires_grad = False

    @property
    def embed_dim(self) -> int:
        return self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        return self.dino_featurizer.patch_embed.num_patches

    @property
    def patch_hw(self) -> Tuple[int, int]:
        side = int(self.num_patches**0.5)
        return side, side

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> dict[str, torch.Tensor | Tuple[int, int]]:
        dino_patches = normalize_patch_tokens(self.dino_featurizer(pixel_values["dino"]), self.num_patches)
        siglip_patches = normalize_patch_tokens(self.siglip_featurizer(pixel_values["siglip"]), self.num_patches)
        patch_tokens = torch.cat([dino_patches, siglip_patches], dim=-1)
        return {
            "patch_tokens": patch_tokens,
            "patch_hw": self.patch_hw,
            "dino_tokens": dino_patches,
            "siglip_tokens": siglip_patches,
        }
