from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer

from qrouter.models.mamba_modeling import MambaForCausalLM


SUPPORTED_LLM_MODELS = {
    "mamba-2.8b-zephyr": "xiuyul/mamba-2.8b-zephyr",
}


@dataclass
class LLMBackboneOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor


class MambaLLMBackbone(nn.Module):
    def __init__(
        self,
        llm_id: str = "mamba-2.8b-zephyr",
        max_length: int = 2048,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if llm_id not in SUPPORTED_LLM_MODELS:
            raise ValueError(f"Unsupported llm_id `{llm_id}`. Supported: {sorted(SUPPORTED_LLM_MODELS)}")
        hf_id = SUPPORTED_LLM_MODELS[llm_id]
        self.identifier = llm_id
        self.hf_id = hf_id
        self.max_length = max_length

        # The language backbone is initialized from a public Mamba-based VLM release.
        # We thank the original authors for making these weights publicly available.
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, model_max_length=max_length)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = MambaForCausalLM.from_pretrained(hf_id)
        self.llm.config.use_cache = False
        self.llm.enable_input_require_grads()

        if use_lora:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["in_proj", "out_proj", "x_proj", "dt_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.llm = get_peft_model(self.llm, lora_cfg)

    @property
    def embed_dim(self) -> int:
        return self.llm.get_input_embeddings().embedding_dim

    def enable_gradient_checkpointing(self) -> None:
        self.llm.gradient_checkpointing_enable()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    def encode_questions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed_input_ids(input_ids)
        weights = attention_mask.unsqueeze(-1).float()
        summed = (embeddings * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> LLMBackboneOutput:
        output = self.llm(
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True,
        )
        return LLMBackboneOutput(loss=output.loss, logits=output.logits)
