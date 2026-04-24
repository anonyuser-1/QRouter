from typing import Any

import torch

from qrouter.utils.image_ops import IGNORE_INDEX, maybe_stack, pad_last_dim


class QRouterBatchCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = maybe_stack([sample["pixel_values"] for sample in batch])
        grounding_images = torch.stack([sample["grounding_images"] for sample in batch], dim=0)

        grounding_masks = [sample["grounding_masks"] for sample in batch]
        has_mask = [mask is not None for mask in grounding_masks]
        if any(has_mask):
            template = next(mask for mask in grounding_masks if mask is not None)
            stacked_masks = []
            for mask in grounding_masks:
                if mask is None:
                    stacked_masks.append(torch.zeros_like(template))
                else:
                    stacked_masks.append(mask)
            grounding_masks = torch.stack(stacked_masks, dim=0)
        else:
            grounding_masks = None

        input_ids = pad_last_dim([sample["input_ids"] for sample in batch], pad_value=0).long()
        attention_mask = pad_last_dim([sample["attention_mask"] for sample in batch], pad_value=0).long()
        labels = pad_last_dim([sample["labels"] for sample in batch], pad_value=IGNORE_INDEX).long()

        return {
            "pixel_values": pixel_values,
            "grounding_images": grounding_images,
            "grounding_masks": grounding_masks,
            "has_grounding_mask": torch.tensor(has_mask, dtype=torch.bool),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question_texts": [sample["question_text"] for sample in batch],
            "grounding_prompts": [sample["grounding_prompt"] for sample in batch],
            "answer_texts": [sample["answer_text"] for sample in batch],
            "image_paths": [sample["image_path"] for sample in batch],
            "task_types": [sample["task_type"] for sample in batch],
            "metadata": [sample["metadata"] for sample in batch],
        }


CanonicalVQACollator = QRouterBatchCollator
