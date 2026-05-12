#!/usr/bin/env python3
"""Update HF model card for CyberSecQwen-4B-AWQ with eval results."""
import json

from huggingface_hub import HfApi

# CTI-Bench evaluation results (5 trials each)
# CTI-MCQ: from full eval run (vLLM AWQ on L4, temp=0.3, concurrency=32)
mcq = {
    "accuracy": 0.5921, "std": 0.0083, "delta_vs_fp16": 0.0053,
    "trials": [
        {"seed": 42, "accuracy": 0.6016, "correct": 1504},
        {"seed": 43, "accuracy": 0.5984, "correct": 1496},
        {"seed": 44, "accuracy": 0.5936, "correct": 1484},
        {"seed": 45, "accuracy": 0.5780, "correct": 1445},
        {"seed": 46, "accuracy": 0.5888, "correct": 1472},
    ],
    "correct": 7401, "total": 12500,
}

# CTI-RCM: from corrected eval (first-CWE extraction, not last-CWE)
rcm_data = {
    "accuracy": 0.5558, "std": 0.0040, "delta_vs_fp16": -0.1106,
    "trials": [
        {"seed": 42, "accuracy": 0.552, "correct": 552},
        {"seed": 43, "accuracy": 0.550, "correct": 550},
        {"seed": 44, "accuracy": 0.560, "correct": 560},
        {"seed": 45, "accuracy": 0.558, "correct": 558},
        {"seed": 46, "accuracy": 0.559, "correct": 559},
    ],
    "correct": 2779, "total": 5000,
}

readme = f"""---
license: apache-2.0
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
| Calibration samples | 128 CTI-Bench prompts |
| Quantization tool | autoawq |
| Calibration hardware | Modal A100 |

## CTI-Bench Evaluation

Evaluated under the [Foundation-Sec-8B protocol](https://arxiv.org/abs/2504.21039) (arXiv:2504.21039):
- Temperature 0.3, max_tokens 512, concurrency 32
- 5 independent trials, zero-shot (no system prompt)
- vLLM v0.20.1 with awq_marlin kernel on Modal L4 GPU

| Task | AWQ 4-bit | FP16 Reference | Delta |
|---|---|---|---|
| CTI-MCQ (2,500 items) | {mcq['accuracy']:.4f} +/- {mcq['std']:.4f} | 0.5868 +/- 0.0029 | {mcq['delta_vs_fp16']:+.4f} |
| CTI-RCM (1,000 items) | {rcm_data['accuracy']:.4f} +/- {rcm_data['std']:.4f} | 0.6664 +/- 0.0023 | {rcm_data['delta_vs_fp16']:+.4f} |

**Key findings:**
- **CTI-MCQ**: AWQ 4-bit matches or slightly exceeds FP16 performance (+0.5 points). No measurable accuracy loss.
- **CTI-RCM**: AWQ 4-bit degrades by {abs(rcm_data['delta_vs_fp16']):.1f} points vs FP16. Parseable rate > 99.8% so answer extraction is working correctly. The model retains correct CWE identification in reasoning but sometimes diverges on final answers. This gap can likely be reduced with more calibration data.

## Trial results

### CTI-MCQ
| Trial | Seed | Accuracy |
|---|---|---|
"""
for t in mcq.get("trials", []):
    readme += f"| {mcq['trials'].index(t) + 1} | {t['seed']} | {t['accuracy']:.4f} |\n"

readme += """
### CTI-RCM
| Trial | Seed | Accuracy |
|---|---|
"""
for t in rcm_data.get("trials", []):
    readme += f"| {rcm_data['trials'].index(t) + 1} | {t['seed']} | {t['accuracy']:.4f} |\n"

readme += """
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
@misc{cybersecqwen2026,
  title  = {CyberSecQwen-4B: A Compact CTI Specialist Fine-Tuned from Qwen3-4B-Instruct-2507 on AMD MI300X},
  author = {Mulia, Samuel},
  year   = {2026},
  publisher = {Hugging Face},
  url    = {https://huggingface.co/athena129/CyberSecQwen-4B}
}
```

## Evaluation Infrastructure

[GitHub repository](https://github.com/ree2raz/cyberSecQwen_4b_4bit) — Modal scripts for AWQ quantization + vLLM CTI-Bench evaluation.
"""

print("Uploading README to ree2raz/CyberSecQwen-4B-AWQ ...")
api = HfApi()
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id="ree2raz/CyberSecQwen-4B-AWQ",
    commit_message="Add CTI-Bench evaluation results (AWQ 4-bit vs FP16)",
)
print("Done! Model card updated.")
