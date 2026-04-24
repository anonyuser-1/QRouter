# QRouter:QRouter: Question-Conditioned Region Routing for Efficient Mamba-based Visual Question Answering

## Introduction
### Abstract
Efficient vision-language models have recently made strong progress on visual question answering (VQA), yet most of them still encode images as dense and largely question-agnostic token sequences. This design limits their ability to focus on the small but semantically critical regions required for fine-grained reasoning. We present QRouter, a framework that introduces question-conditioned grounding as an intermediate structural prior for fine-grained visual question answering. Instead of treating grounding as a standalone output task, QRouter uses grounded regions to reorganize dense visual features into a compact structured visual sequence consisting of region tokens, routed context tokens, and a background token. The resulting sequence is then processed by an efficient Mamba-based multimodal backbone. Experiments on seven benchmarks show that QRouter consistently improves over strong open-source baselines, including recent efficient state-space vision-language models, on both open-ended and grounding-sensitive question answering tasks. The improvements are particularly pronounced on benchmarks that require compositional reasoning, spatial verification, and hallucination suppression. Crucially, QRouter preserves the favorable efficiency characteristics of compact Mamba-based backbones, keeping end-to-end inference cost close to strong efficiency-oriented baselines. These results suggest that explicitly organizing question-relevant visual evidence is a practical and effective direction for efficient fine-grained visual question answering. 

### Overview
The overall framework of QRouter is as follows：
![overview](https://github.com/anonyuser-1/QRouter/blob/main/framework.png)




This repository focuses on the **training mainline** used in the paper. It includes:

- the online QRouter model with question-conditioned grounding and structured token routing;
- Stage I / Stage II training scripts aligned with the paper setup;
- optional inference utilities for question-answer prediction;
- auxiliary GQA analysis scripts used for the fine-grained appendix results.

The repository intentionally does **not** include private data files, cached outputs, or unpublished checkpoints.

## Repository Layout

```text
qrouter/
  data/               dataset and collator code
  models/             grounding, vision, language, routing, and multimodal model code
  utils/              checkpoint, image, and training helpers
scripts/
  train.py            main training entrypoint
  infer.py            optional inference helper
  analysis/           GQA structural / semantic / steps analysis utilities
configs/
  stage1.yaml         paper-aligned Stage I template
  stage2.yaml         paper-aligned Stage II template
```

## Environment Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

This project also depends on the official SAM2 package. Please install SAM2 from its public repository before training.

## External Dependencies and Acknowledgements

QRouter builds on several publicly released pretrained models and libraries. We thank the original authors and maintainers for making these resources available:

- **DINOv2** for dense visual features.
- **SigLIP** for complementary visual features.
- **SAM2** for high-resolution segmentation decoding and image-grounded mask prediction.
- **Qwen2.5-VL** for language-conditioned grounding prompts.
- A **public Mamba-based VLM initialization checkpoint** for the language-side answer-generation backbone.

Please download these assets from their official release pages and follow their respective licenses and terms of use.

## Data Preparation

### 1. QA training data

The QA training file is expected to be a JSONL file with entries like:

```json
{"image": "images/example.jpg", "question": "What color is the sign?", "answer": "red"}
```

Optional fields:

- `grounding_prompt`: custom prompt sent to the grounding branch;
- `grounding_mask` or `mask`: optional supervision mask path;
- `metadata`: arbitrary JSON metadata copied into the batch output.

If your QA mixture is stored in a conversation-style LLaVA JSON format, set `qa_dataset_format: llava` in the training config instead of `auto` or `jsonl`.

### 2. CIS grounding data

The Stage II grounding-only data is expected to be a JSONL file with entries like:

```json
{"image": "images/example.jpg", "prompt": "Segment the primary dining surface.", "mask": "masks/example.png", "concept": "relation"}
```

Supported fields:

- `prompt` or `question`: text query used for grounding-only supervision;
- `mask` or `grounding_mask`: binary supervision mask;
- `concept`: optional concept tag preserved in metadata.

## Paper-Aligned Defaults

The default configs match the paper setup:

- **Stage I**: `20k` optimization steps
- **Stage II**: `10k` optimization steps
- **Stage II QA:CIS ratio**: `2:1`
- **Region / context token budgets**: `K=32`, `L=128`
- **Confidence threshold**: `tau_s = 0.5`
- **Dice weight**: `lambda_dice = 0.25`
- **Gradient clipping**: `max_grad_norm = 1.0`
- **Checkpoint selection**: the best checkpoint is selected by the lowest validation loss on a held-out QA validation split
- **Grounding training**: the SAM2 image tower remains frozen; Stage II only enables lightweight language-side grounding adaptation

The main results in the paper are reported from a single run per model configuration due to compute constraints.

## Training

### Stage I

```bash
torchrun --nproc_per_node=8 scripts/train.py --config configs/stage1.yaml
```

### Stage II

```bash
torchrun --nproc_per_node=8 scripts/train.py --config configs/stage2.yaml
```

Both config files use placeholder paths such as `your/path/to/...`. Please update them before training.

## Optional Inference

After training, you can run the optional inference helper on a GQA-style JSON file or a generic JSONL/JSON question file:

```bash
python scripts/infer.py \
  --checkpoint outputs/qrouter_stage2/checkpoints/best.pt \
  --questions your/path/to/testdev_balanced_questions.json \
  --image-root your/path/to/images \
  --sam2-config your/path/to/sam2_hiera_l.yaml \
  --sam2-checkpoint your/path/to/sam2_hiera_large.pt \
  --out-jsonl outputs/qrouter_eval/results.jsonl \
  --append-short-answer-instruction
```

## GQA Analysis Utilities

These scripts are auxiliary tools for reproducing the fine-grained GQA tables in the paper appendix.

```bash
python scripts/analysis/analyze_gqa_structural_accuracy.py \
  --questions-json your/path/to/testdev_balanced_questions.json \
  --results-jsonl outputs/qrouter_eval/results.jsonl \
  --out-jsonl outputs/gqa_structural_analysis.jsonl \
  --summary-json outputs/gqa_structural_summary.json
```

```bash
python scripts/analysis/analyze_gqa_semantic_accuracy.py \
  --questions-json your/path/to/testdev_balanced_questions.json \
  --results-jsonl outputs/qrouter_eval/results.jsonl \
  --out-jsonl outputs/gqa_semantic_analysis.jsonl \
  --summary-json outputs/gqa_semantic_summary.json
```

```bash
python scripts/analysis/analyze_gqa_steps_accuracy.py \
  --questions-json your/path/to/testdev_balanced_questions.json \
  --results-jsonl outputs/qrouter_eval/results.jsonl \
  --out-jsonl outputs/gqa_steps_analysis.jsonl \
  --summary-json outputs/gqa_steps_summary.json
```

## Notes

- This release is intentionally anonymous and replaces machine-specific paths with configurable placeholders.
- If a required checkpoint or dependency is missing, the scripts will raise an explicit error instead of silently falling back to private locations.
- The public training path in this repository is the one maintained to match the paper setup. Additional internal demo scripts and temporary experiments are intentionally omitted from this release.
