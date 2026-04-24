from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from qrouter.utils.image_ops import batched_index_select, safe_normalize


@dataclass
class RegionTokenizerOutput:
    visual_tokens: torch.Tensor
    region_tokens: torch.Tensor
    context_tokens: torch.Tensor
    background_token: torch.Tensor
    patch_prior: torch.Tensor
    routing_weights: torch.Tensor
    selected_indices: torch.Tensor
    grounding_valid: torch.Tensor


class MaskDownsampler(nn.Module):
    def forward(self, masks: torch.Tensor, patch_hw: tuple[int, int]) -> torch.Tensor:
        batch_size, num_masks = masks.shape[:2]
        flattened = masks.reshape(batch_size * num_masks, 1, masks.shape[-2], masks.shape[-1])
        pooled = F.adaptive_avg_pool2d(flattened, patch_hw)
        return pooled.reshape(batch_size, num_masks, -1)


class RegionPooler(nn.Module):
    def __init__(self, vision_dim: int) -> None:
        super().__init__()
        self.geometry_mlp = nn.Sequential(
            nn.Linear(5, vision_dim),
            nn.SiLU(),
            nn.Linear(vision_dim, vision_dim),
        )

    def _geometry(self, mask_grid: torch.Tensor, patch_hw: tuple[int, int]) -> torch.Tensor:
        batch_size, num_masks, num_patches = mask_grid.shape
        height, width = patch_hw
        mask_2d = mask_grid.view(batch_size, num_masks, height, width)
        geom = torch.zeros(batch_size, num_masks, 5, device=mask_grid.device, dtype=mask_grid.dtype)
        for batch_idx in range(batch_size):
            for mask_idx in range(num_masks):
                coords = torch.nonzero(mask_2d[batch_idx, mask_idx] > 0.05, as_tuple=False)
                if coords.numel() == 0:
                    continue
                y_min = coords[:, 0].min().float() / max(height - 1, 1)
                x_min = coords[:, 1].min().float() / max(width - 1, 1)
                y_max = coords[:, 0].max().float() / max(height - 1, 1)
                x_max = coords[:, 1].max().float() / max(width - 1, 1)
                area = (mask_2d[batch_idx, mask_idx] > 0.05).float().mean()
                geom[batch_idx, mask_idx] = torch.tensor([x_min, y_min, x_max, y_max, area], device=mask_grid.device)
        return geom

    def forward(self, patch_tokens: torch.Tensor, mask_grid: torch.Tensor, scores: torch.Tensor, patch_hw: tuple[int, int]) -> torch.Tensor:
        normalized_masks = safe_normalize(mask_grid.clamp_min(0.0))
        pooled = torch.einsum("bkn,bnd->bkd", normalized_masks, patch_tokens)
        weighted = pooled * scores.unsqueeze(-1)
        geometry = self.geometry_mlp(self._geometry(mask_grid, patch_hw))
        return weighted + geometry


class RoutingScorer(nn.Module):
    def __init__(self, vision_dim: int, question_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(vision_dim, hidden_dim)
        self.prior_proj = nn.Linear(1, hidden_dim)
        self.question_proj = nn.Linear(question_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, patch_tokens: torch.Tensor, patch_prior: torch.Tensor, question_embed: torch.Tensor) -> torch.Tensor:
        visual = self.visual_proj(patch_tokens)
        prior = self.prior_proj(patch_prior.unsqueeze(-1))
        question = self.question_proj(question_embed).unsqueeze(1)
        hidden = torch.tanh(visual + prior + question)
        return self.output(hidden).squeeze(-1)


class BackgroundPooler(nn.Module):
    def forward(self, patch_tokens: torch.Tensor, selected_weights: torch.Tensor) -> torch.Tensor:
        residual = (1.0 - selected_weights).clamp_min(0.0)
        norm = residual.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = residual / norm
        return torch.einsum("bn,bnd->bd", normalized, patch_tokens)


class RegionTokenizer(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        question_dim: int,
        num_region_tokens: int = 32,
        num_context_tokens: int = 128,
        confidence_threshold: float = 0.5,
        routing_hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.num_region_tokens = num_region_tokens
        self.num_context_tokens = num_context_tokens
        self.confidence_threshold = confidence_threshold
        self.downsampler = MaskDownsampler()
        self.pooler = RegionPooler(vision_dim=vision_dim)
        self.router = RoutingScorer(vision_dim=vision_dim, question_dim=question_dim, hidden_dim=routing_hidden_dim)
        self.background_pooler = BackgroundPooler()
        self.token_type_embedding = nn.Embedding(3, vision_dim)

    def _build_uniform_masks(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, _ = patch_tokens.shape
        return torch.full(
            (batch_size, self.num_region_tokens, num_patches),
            1.0 / max(num_patches, 1),
            dtype=patch_tokens.dtype,
            device=patch_tokens.device,
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_hw: tuple[int, int],
        masks: torch.Tensor,
        scores: torch.Tensor,
        question_embed: torch.Tensor,
    ) -> RegionTokenizerOutput:
        batch_size, num_patches, _ = patch_tokens.shape
        grounding_valid = (masks.flatten(2).sum(dim=-1) > 0).any(dim=-1)
        sort_scores, sort_indices = scores.sort(dim=1, descending=True)
        masks = masks.gather(
            1,
            sort_indices[:, :, None, None].expand(-1, -1, masks.shape[-2], masks.shape[-1]),
        )
        scores = sort_scores
        masks = masks[:, : self.num_region_tokens]
        scores = scores[:, : self.num_region_tokens]
        valid_scores = scores >= self.confidence_threshold
        masks = masks * valid_scores[:, :, None, None].to(masks.dtype)
        scores = torch.where(valid_scores, scores, torch.zeros_like(scores))

        if masks.shape[1] < self.num_region_tokens:
            pad_count = self.num_region_tokens - masks.shape[1]
            masks = torch.cat(
                [
                    masks,
                    torch.zeros(batch_size, pad_count, *masks.shape[-2:], device=masks.device, dtype=masks.dtype),
                ],
                dim=1,
            )
            scores = torch.cat(
                [
                    scores,
                    torch.zeros(batch_size, pad_count, device=scores.device, dtype=scores.dtype),
                ],
                dim=1,
            )

        mask_grid = self.downsampler(masks, patch_hw=patch_hw)
        uniform_masks = self._build_uniform_masks(patch_tokens)
        mask_grid = torch.where(grounding_valid[:, None, None], mask_grid, uniform_masks)

        scores = torch.where(grounding_valid[:, None], scores, torch.ones_like(scores))
        region_tokens = self.pooler(patch_tokens=patch_tokens, mask_grid=mask_grid, scores=scores, patch_hw=patch_hw)

        patch_prior = (mask_grid * scores.unsqueeze(-1)).sum(dim=1)
        patch_prior = safe_normalize(patch_prior.clamp_min(0.0))
        uniform_prior = torch.full_like(patch_prior, 1.0 / max(num_patches, 1))
        patch_prior = torch.where(grounding_valid[:, None], patch_prior, uniform_prior)

        routing_scores = self.router(patch_tokens=patch_tokens, patch_prior=patch_prior, question_embed=question_embed)
        routing_weights = torch.softmax(routing_scores, dim=-1)

        top_k = min(self.num_context_tokens, num_patches)
        selected_scores, selected_indices = torch.topk(routing_scores, k=top_k, dim=-1)
        context_tokens = batched_index_select(patch_tokens, selected_indices)
        if top_k < self.num_context_tokens:
            pad = torch.zeros(
                batch_size,
                self.num_context_tokens - top_k,
                patch_tokens.shape[-1],
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
            context_tokens = torch.cat([context_tokens, pad], dim=1)
            selected_indices = torch.cat(
                [
                    selected_indices,
                    selected_indices[:, -1:].expand(-1, self.num_context_tokens - top_k),
                ],
                dim=1,
            )

        selected_mask = torch.zeros_like(routing_weights)
        selected_mask.scatter_(1, selected_indices[:, :top_k], 1.0)
        background_token = self.background_pooler(patch_tokens=patch_tokens, selected_weights=selected_mask)

        region_type = self.token_type_embedding(torch.zeros(self.num_region_tokens, dtype=torch.long, device=patch_tokens.device))
        context_type = self.token_type_embedding(torch.ones(self.num_context_tokens, dtype=torch.long, device=patch_tokens.device))
        background_type = self.token_type_embedding(torch.full((1,), 2, dtype=torch.long, device=patch_tokens.device))

        region_tokens = region_tokens + region_type.unsqueeze(0)
        context_tokens = context_tokens + context_type.unsqueeze(0)
        background_token = background_token.unsqueeze(1) + background_type.unsqueeze(0)
        visual_tokens = torch.cat([region_tokens, context_tokens, background_token], dim=1)

        return RegionTokenizerOutput(
            visual_tokens=visual_tokens,
            region_tokens=region_tokens,
            context_tokens=context_tokens,
            background_token=background_token,
            patch_prior=patch_prior,
            routing_weights=routing_weights,
            selected_indices=selected_indices,
            grounding_valid=grounding_valid,
        )
