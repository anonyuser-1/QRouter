from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qrouter.models.grounding import ConversationalGroundingModule
from qrouter.models.llm_backbone import MambaLLMBackbone
from qrouter.models.projector import MLPProjector
from qrouter.models.region_tokenizer import RegionTokenizer
from qrouter.models.vision_backbone import DualVisionBackbone


IGNORE_INDEX = -100


@dataclass
class ModelOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    metrics: dict[str, torch.Tensor]
    grounding_valid: torch.Tensor


def dice_loss_from_probs(pred_probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred_probs * target).sum(dim=(-2, -1))
    denom = pred_probs.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - ((2.0 * inter + eps) / (denom + eps))


class RegionRoutingVQAModel(nn.Module):
    def __init__(
        self,
        grounding_module: ConversationalGroundingModule,
        vision_backbone: DualVisionBackbone,
        llm_backbone: MambaLLMBackbone,
        num_region_tokens: int = 32,
        num_context_tokens: int = 128,
        alignment_loss_weight: float = 0.1,
        compactness_loss_weight: float = 0.01,
        diversity_loss_weight: float = 0.01,
        segmentation_loss_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.grounding_module = grounding_module
        self.vision_backbone = vision_backbone
        self.llm_backbone = llm_backbone
        self.region_tokenizer = RegionTokenizer(
            vision_dim=vision_backbone.embed_dim,
            question_dim=llm_backbone.embed_dim,
            num_region_tokens=num_region_tokens,
            num_context_tokens=num_context_tokens,
        )
        self.projector = MLPProjector(vision_dim=vision_backbone.embed_dim, llm_dim=llm_backbone.embed_dim)
        self.alignment_loss_weight = alignment_loss_weight
        self.compactness_loss_weight = compactness_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.segmentation_loss_weight = segmentation_loss_weight
        self.stage = "stage1"

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.grounding_module.set_stage(stage)
        for parameter in self.vision_backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.region_tokenizer.parameters():
            parameter.requires_grad = True
        for parameter in self.projector.parameters():
            parameter.requires_grad = True

    def _question_mask(self, batch: dict[str, Any]) -> torch.Tensor:
        return ((batch["labels"] == IGNORE_INDEX) & batch["attention_mask"].bool()).long()

    def _build_multimodal_sequence(
        self,
        visual_embeds: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_embeds = self.llm_backbone.embed_input_ids(text_input_ids)
        attention_prefix = torch.ones(
            visual_embeds.shape[:2],
            dtype=text_attention_mask.dtype,
            device=text_attention_mask.device,
        )
        label_prefix = torch.full(
            visual_embeds.shape[:2],
            IGNORE_INDEX,
            dtype=labels.dtype,
            device=labels.device,
        )
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        attention_mask = torch.cat([attention_prefix, text_attention_mask], dim=1)
        full_labels = torch.cat([label_prefix, labels], dim=1)
        return inputs_embeds, attention_mask, full_labels

    def _prepare_visual_support(self, batch: dict[str, Any]) -> tuple[torch.Tensor, Any, Any]:
        vision_outputs = self.vision_backbone(batch["pixel_values"])
        question_embed = self.llm_backbone.encode_questions(
            input_ids=batch["input_ids"],
            attention_mask=self._question_mask(batch),
        )
        grounding_outputs = self.grounding_module(
            images=batch["grounding_images"],
            questions=batch["grounding_prompts"],
            image_paths=batch["image_paths"],
        )
        region_outputs = self.region_tokenizer(
            patch_tokens=vision_outputs["patch_tokens"],
            patch_hw=vision_outputs["patch_hw"],
            masks=grounding_outputs.masks,
            scores=grounding_outputs.scores,
            question_embed=question_embed,
        )
        visual_embeds = self.projector(region_outputs.visual_tokens)
        return visual_embeds, grounding_outputs, region_outputs

    def _alignment_loss(self, routing_weights: torch.Tensor, patch_prior: torch.Tensor, grounding_valid: torch.Tensor) -> torch.Tensor:
        kl = F.kl_div(
            routing_weights.clamp_min(1e-6).log(),
            patch_prior.clamp_min(1e-6),
            reduction="none",
        ).sum(dim=-1)
        return (kl * grounding_valid.float()).sum() / grounding_valid.float().sum().clamp_min(1.0)

    def _compactness_loss(self, routing_weights: torch.Tensor, grounding_valid: torch.Tensor) -> torch.Tensor:
        entropy = -(routing_weights.clamp_min(1e-6) * routing_weights.clamp_min(1e-6).log()).sum(dim=-1)
        return (entropy * grounding_valid.float()).sum() / grounding_valid.float().sum().clamp_min(1.0)

    def _diversity_loss(self, region_tokens: torch.Tensor) -> torch.Tensor:
        if region_tokens.shape[1] <= 1:
            return region_tokens.new_zeros(())
        normalized = F.normalize(region_tokens, dim=-1)
        sim = torch.matmul(normalized, normalized.transpose(1, 2))
        eye = torch.eye(sim.shape[-1], device=sim.device, dtype=sim.dtype).unsqueeze(0)
        penalty = (sim - eye).pow(2) * (1.0 - eye)
        return penalty.mean()

    def _segmentation_loss(
        self,
        pred_masks: torch.Tensor,
        gt_masks: Optional[torch.Tensor],
        has_grounding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if gt_masks is None or has_grounding_mask is None or not has_grounding_mask.any():
            return pred_masks.new_zeros(())
        pred_best = pred_masks[:, 0]
        gt = gt_masks.squeeze(1).to(pred_best.device)
        valid = has_grounding_mask.float().to(pred_best.device)
        bce = F.binary_cross_entropy(pred_best, gt, reduction="none").mean(dim=(-2, -1))
        dice = dice_loss_from_probs(pred_best, gt)
        loss = (bce + dice) * valid
        return loss.sum() / valid.sum().clamp_min(1.0)

    def _compute_routing_losses(self, batch: dict[str, Any], grounding_outputs, region_outputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alignment_loss = self._alignment_loss(
            routing_weights=region_outputs.routing_weights,
            patch_prior=region_outputs.patch_prior,
            grounding_valid=region_outputs.grounding_valid,
        )
        compactness_loss = self._compactness_loss(
            routing_weights=region_outputs.routing_weights,
            grounding_valid=region_outputs.grounding_valid,
        )
        diversity_loss = self._diversity_loss(region_outputs.region_tokens)
        segmentation_loss = self._segmentation_loss(
            pred_masks=grounding_outputs.masks,
            gt_masks=batch.get("grounding_masks"),
            has_grounding_mask=batch.get("has_grounding_mask"),
        )
        return alignment_loss, compactness_loss, diversity_loss, segmentation_loss

    def forward_grounding_only(self, batch: dict[str, Any]) -> ModelOutput:
        visual_embeds, grounding_outputs, region_outputs = self._prepare_visual_support(batch)
        alignment_loss, compactness_loss, diversity_loss, segmentation_loss = self._compute_routing_losses(
            batch=batch,
            grounding_outputs=grounding_outputs,
            region_outputs=region_outputs,
        )

        answer_loss = visual_embeds.new_zeros(())
        total_loss = (
            self.alignment_loss_weight * alignment_loss
            + self.compactness_loss_weight * compactness_loss
            + self.diversity_loss_weight * diversity_loss
            + self.segmentation_loss_weight * segmentation_loss
        )
        metrics = {
            "answer_loss": answer_loss.detach(),
            "alignment_loss": alignment_loss.detach(),
            "compactness_loss": compactness_loss.detach(),
            "diversity_loss": diversity_loss.detach(),
            "segmentation_loss": segmentation_loss.detach(),
            "grounding_valid_ratio": region_outputs.grounding_valid.float().mean().detach(),
        }
        return ModelOutput(
            loss=total_loss,
            logits=visual_embeds.new_zeros((visual_embeds.shape[0], 0, 0)),
            metrics=metrics,
            grounding_valid=region_outputs.grounding_valid,
        )

    def forward_qa(self, batch: dict[str, Any]) -> ModelOutput:
        visual_embeds, grounding_outputs, region_outputs = self._prepare_visual_support(batch)
        inputs_embeds, attention_mask, labels = self._build_multimodal_sequence(
            visual_embeds=visual_embeds,
            text_input_ids=batch["input_ids"],
            text_attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        llm_output = self.llm_backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        alignment_loss, compactness_loss, diversity_loss, segmentation_loss = self._compute_routing_losses(
            batch=batch,
            grounding_outputs=grounding_outputs,
            region_outputs=region_outputs,
        )
        total_loss = llm_output.loss
        total_loss = total_loss + self.alignment_loss_weight * alignment_loss
        total_loss = total_loss + self.compactness_loss_weight * compactness_loss
        total_loss = total_loss + self.diversity_loss_weight * diversity_loss
        total_loss = total_loss + self.segmentation_loss_weight * segmentation_loss
        metrics = {
            "answer_loss": llm_output.loss.detach(),
            "alignment_loss": alignment_loss.detach(),
            "compactness_loss": compactness_loss.detach(),
            "diversity_loss": diversity_loss.detach(),
            "segmentation_loss": segmentation_loss.detach(),
            "grounding_valid_ratio": region_outputs.grounding_valid.float().mean().detach(),
        }
        return ModelOutput(
            loss=total_loss,
            logits=llm_output.logits,
            metrics=metrics,
            grounding_valid=region_outputs.grounding_valid,
        )

    def forward(self, batch: dict[str, Any]) -> ModelOutput:
        task_types = batch.get("task_types")
        if task_types is None:
            task_type = batch.get("task_type", "qa")
        else:
            unique_task_types = sorted(set(task_types))
            if len(unique_task_types) != 1:
                raise ValueError(f"Expected a homogeneous batch, but received task types: {unique_task_types}")
            task_type = unique_task_types[0]
        if task_type == "grounding":
            return self.forward_grounding_only(batch)
        if task_type != "qa":
            raise ValueError(f"Unsupported task type `{task_type}`.")
        return self.forward_qa(batch)
