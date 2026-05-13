#!/usr/bin/env python3
"""Update HF model card for CyberSecQwen-4B-AWQ with eval results + GGUF comparison."""
from huggingface_hub import HfApi

mcq = {
    "accuracy": 0.5921, "std": 0.0083, "delta_vs_fp16": 0.0053,
    "trials": [
        {"seed": 42, "accuracy": 0.6016}, {"seed": 43, "accuracy": 0.5984},
        {"seed": 44, "accuracy": 0.5936}, {"seed": 45, "accuracy": 0.5780},
        {"seed": 46, "accuracy": 0.5888},
    ],
}

rcm_data = {
    "accuracy": 0.5814, "std": 0.0025, "delta_vs_fp16": -0.0850,
    "trials": [
        {"seed": 42, "accuracy": 0.579}, {"seed": 43, "accuracy": 0.583},
        {"seed": 44, "accuracy": 0.579}, {"seed": 45, "accuracy": 0.584},
        {"seed": 46, "accuracy": 0.582},
    ],
}

readme = f"""---
license: apache-2.0
base_model: lablab-ai-amd-developer-hackathon/CyberSecQwen-4B
tags:
- qwen3
- cybersecurity
- cti
- cwe-classification
- vulnerability-analysis
- awq
- 4-bit
- quantized
library_name: transformers
pipeline_tag: text-generation
---

# CyberSecQwen-4B-AWQ

4-bit AWQ quantized version of [CyberSecQwen-4B](https://huggingface.co/lablab-ai-amd-developer-hackathon/CyberSecQwen-4B).

## Quantization

| Parameter | Value |
|---|---|
| Method | AWQ (group_size=128, zero_point=True) |
| Weight precision | 4-bit |
| Compute dtype | float16 |
| Calibration samples | 320 CTI-Bench prompts (256 RCM + 64 MCQ, chat-template formatted) |
| Quantization tool | autoawq |
| Calibration hardware | Modal A100 |

## CTI-Bench Evaluation

Evaluated under the [Foundation-Sec-8B protocol](https://arxiv.org/abs/2504.21039):
- Temperature 0.3, max_tokens 512, concurrency 32
- 5 independent trials, zero-shot (no system prompt)
- vLLM v0.20.1 with awq_marlin kernel on Modal L4 GPU

| Task | AWQ 4-bit | GGUF Q4_K_M | FP16 Reference |
|---|---|---|---|
| CTI-MCQ (2,500 items) | **{mcq['accuracy']:.4f}** ± {mcq['std']:.4f} | 0.5368 ± 0.0048 | 0.5868 ± 0.0029 |
| CTI-RCM (1,000 items) | {rcm_data['accuracy']:.4f} ± {rcm_data['std']:.4f} | **0.6254 ± 0.0063** | 0.6664 ± 0.0023 |

**Key findings:**
- **CTI-MCQ**: AWQ 4-bit matches or slightly exceeds FP16 performance (+0.5 points). Better than GGUF Q4_K_M.
- **CTI-RCM**: AWQ 4-bit degrades by {abs(rcm_data['delta_vs_fp16'])*100:.1f} percentage points vs FP16. GGUF Q4_K_M does better on this task (-4.1 pts).
- AWQ is best for MCQ (general language), GGUF is best for RCM (task-specific classification).

## Trial results

### CTI-MCQ
| Trial | Seed | Accuracy |
|---|---|---|
"""
for i, t in enumerate(mcq['trials']):
    readme += f"| {i+1} | {t['seed']} | {t['accuracy']:.4f} |\n"

readme += """
### CTI-MCQ

| Trial | Seed | Accuracy |
|---|---|---|
"""
for i, t in enumerate(mcq['trials']):
    readme += f"| {i+1} | {t['seed']} | {t['accuracy']:.4f} |\n"

readme += """
### CTI-RCM

| Trial | Seed | Accuracy |
|---|---|---|
"""
for i, t in enumerate(rcm_data['trials']):
    readme += f"| {i+1} | {t['seed']} | {t['accuracy']:.4f} |\n"

readme += """
## Quantization variants

| Variant | CTI-MCQ | CTI-RCM | Size | Engine |
|---|---|---|---|---|
| [AWQ 4-bit](https://huggingface.co/ree2raz/CyberSecQwen-4B-AWQ) | 0.5921 | 0.5814 | 2.7 GB | vLLM |
| [GGUF Q4_K_M](https://huggingface.co/ree2raz/CyberSecQwen-4B-GGUF) | 0.5368 | 0.6254 | 2.5 GB | llama.cpp |

Choose AWQ for MCQ/general chat, GGUF for vulnerability classification.

## Usage with vLLM

```bash
vllm serve ree2raz/CyberSecQwen-4B-AWQ --quantization awq_marlin --dtype float16
```

## Model Size

| Format | Size |
|---|---|
| Original FP16 | ~8 GB |
| AWQ 4-bit | ~2.7 GB |

## Citation

```bibtex
@misc{{cybersecqwen2026,
  title  = {{CyberSecQwen-4B: A Compact CTI Specialist Fine-Tuned from Qwen3-4B-Instruct-2507 on AMD MI300X}},
  author = {{Mulia, Samuel}},
  year   = {{2026}},
  publisher = {{Hugging Face}},
  url    = {{https://huggingface.co/athena129/CyberSecQwen-4B}}
}}
```

## Evaluation Infrastructure

[GitHub repository](https://github.com/ree2raz/cyberSecQwen_4b_4bit) — Modal scripts for quantization + evaluation.
"""

api = HfApi()
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id="ree2raz/CyberSecQwen-4B-AWQ",
    commit_message="Update eval with GGUF comparison; RCM 0.5814, MCQ 0.5921",
)
print("AWQ model card updated!")
