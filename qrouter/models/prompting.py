from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedTokenizerBase


IGNORE_INDEX = -100


@dataclass
class PromptEncoding:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_text: str


def build_grounding_prompt(question: str, history: Optional[list[str]] = None, max_history: int = 1) -> str:
    history = history or []
    trimmed = [item.strip() for item in history[-max_history:] if item and item.strip()]
    current = question.strip()
    if not trimmed:
        return current
    return "\n".join(trimmed + [current])


def build_zephyr_prompt(question: str, answer: Optional[str] = None) -> tuple[str, str]:
    prompt = f"<|user|>\n{question.strip()}<|endoftext|>\n<|assistant|>\n"
    full_text = prompt if answer is None else f"{prompt}{answer.strip()}<|endoftext|>"
    return prompt, full_text


def encode_prompt_only(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_length: int,
) -> PromptEncoding:
    prompt_text, full_text = build_zephyr_prompt(question=question, answer=None)
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    return PromptEncoding(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_text=prompt_text,
    )


def encode_vqa_example(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    answer: str,
    max_length: int,
) -> PromptEncoding:
    prompt_text, full_text = build_zephyr_prompt(question=question, answer=answer)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = full["input_ids"][0]
    attention_mask = full["attention_mask"][0]
    labels = input_ids.clone()
    prompt_length = min(len(prompt_ids), labels.shape[0])
    labels[:prompt_length] = IGNORE_INDEX
    return PromptEncoding(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_text=prompt_text,
    )
