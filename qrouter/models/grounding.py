from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from qrouter.models.language_adapter import LanguageAdapter

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:  # pragma: no cover - resolved on user server
    build_sam2 = None
    SAM2ImagePredictor = None


@dataclass
class GroundingOutput:
    masks: torch.Tensor
    scores: torch.Tensor
    low_res_logits: torch.Tensor
    aux: dict


class ConversationalGroundingModule(nn.Module):
    def __init__(
        self,
        sam2_cfg: str,
        sam2_ckpt: str,
        qwen_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        num_masks: int = 3,
        precision: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        use_lora: bool = True,
    ) -> None:
        super().__init__()
        if build_sam2 is None or SAM2ImagePredictor is None:
            raise ImportError("SAM2 is not installed. Install it in the runtime environment before training.")
        self.num_masks = num_masks
        self.device = torch.device(device)
        # The SAM2 checkpoint is loaded from the official open-source release.
        # Please refer to the SAM2 project page for download instructions and licensing.
        self.predictor = SAM2ImagePredictor(build_sam2(sam2_cfg, sam2_ckpt, device=device))
        self.predictor.model.eval()

        transformer_dim = self.predictor.model.sam_mask_decoder.transformer_dim
        self.prompt_encoder = LanguageAdapter(
            model_name=qwen_id,
            transformer_dim=transformer_dim,
            n_sparse_tokens=0,
            use_dense_bias=True,
            dtype=precision,
            device=device,
            use_lora=use_lora,
        )
        self.set_stage("stage1")

    def set_stage(self, stage: str) -> None:
        # The SAM2 image tower remains frozen in both stages.
        for parameter in self.predictor.model.parameters():
            parameter.requires_grad = False
        for parameter in self.prompt_encoder.parameters():
            parameter.requires_grad = False

        if stage == "stage1":
            return
        if stage != "stage2":
            raise ValueError(f"Unsupported stage `{stage}`.")

        # Stage II only tunes lightweight language-side adapters.
        for parameter in self.prompt_encoder.parameters():
            parameter.requires_grad = True

    def _tensor_to_uint8_rgb(self, image_tensor: torch.Tensor) -> np.ndarray:
        image = image_tensor.detach().cpu().clamp(0, 1)
        image = (image * 255.0).round().byte().permute(1, 2, 0).numpy()
        return image

    def forward(self, images: torch.Tensor, questions: list[str], image_paths: list[str]) -> GroundingOutput:
        mask_list = []
        score_list = []
        low_res_list = []
        image_embed_norms = []

        amp_ctx = contextlib.nullcontext()
        if self.device.type == "cuda":
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        with amp_ctx:
            for image_tensor, question, image_path in zip(images, questions, image_paths):
                image_np = self._tensor_to_uint8_rgb(image_tensor)
                self.predictor.set_image(image_np)
                image_embed = self.predictor._features["image_embed"][-1].unsqueeze(0)
                high_res = [level[-1].unsqueeze(0) for level in self.predictor._features["high_res_feats"]]

                sparse_prompt, dense_prompt = self.prompt_encoder(
                    [question],
                    H=image_embed.shape[-2],
                    W=image_embed.shape[-1],
                    image_paths=[image_path],
                )

                decoder = self.predictor.model.sam_mask_decoder
                decoder_device = next(decoder.parameters()).device
                decoder_dtype = next(decoder.parameters()).dtype
                image_pe = self.predictor.model.sam_prompt_encoder.get_dense_pe().to(decoder_device, decoder_dtype)
                low_res_masks, scores, _, _ = decoder(
                    image_embeddings=image_embed.to(decoder_device, decoder_dtype),
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_prompt.to(decoder_device, decoder_dtype),
                    dense_prompt_embeddings=dense_prompt.to(decoder_device, decoder_dtype),
                    multimask_output=True,
                    repeat_image=False,
                    high_res_features=[feat.to(decoder_device, decoder_dtype) for feat in high_res],
                )
                post_masks = self.predictor._transforms.postprocess_masks(
                    low_res_masks,
                    self.predictor._orig_hw[-1],
                )
                post_masks = torch.sigmoid(post_masks[:, : self.num_masks])
                scores = scores[:, : self.num_masks]
                low_res_masks = low_res_masks[:, : self.num_masks]

                mask_list.append(post_masks.squeeze(0))
                score_list.append(scores.squeeze(0))
                low_res_list.append(low_res_masks.squeeze(0))
                image_embed_norms.append(image_embed.float().norm(dim=1).mean().detach())

        masks = torch.stack(mask_list, dim=0)
        scores = torch.stack(score_list, dim=0)
        low_res_logits = torch.stack(low_res_list, dim=0)
        aux = {
            "image_embed_norm": torch.stack(image_embed_norms, dim=0),
        }
        return GroundingOutput(masks=masks, scores=scores, low_res_logits=low_res_logits, aux=aux)
