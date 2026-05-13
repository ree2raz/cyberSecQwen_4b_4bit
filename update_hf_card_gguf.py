#!/usr/bin/env python3
"""Create/update HF model card for CyberSecQwen-4B-GGUF."""
from huggingface_hub import HfApi

mcq = {
    "accuracy": 0.5368, "std": 0.0048, "delta_vs_fp16": -0.0500,
    "trials": [
        {"seed": 42, "accuracy": 0.5420}, {"seed": 43, "accuracy": 0.5280},
        {"seed": 44, "accuracy": 0.5360}, {"seed": 45, "accuracy": 0.5392},
        {"seed": 46, "accuracy": 0.5388},
    ],
}

rcm_data = {
    "accuracy": 0.6254, "std": 0.0063, "delta_vs_fp16": -0.0410,
    "trials": [
        {"seed": 42, "accuracy": 0.6270}, {"seed": 43, "accuracy": 0.6300},
        {"seed": 44, "accuracy": 0.6270}, {"seed": 45, "accuracy": 0.6300},
        {"seed": 46, "accuracy": 0.6130},
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
- gguf
- q4_k_m
- 4-bit
- quantized
---

# CyberSecQwen-4B-GGUF

GGUF Q4_K_M quantized version of [CyberSecQwen-4B](https://huggingface.co/lablab-ai-amd-developer-hackathon/CyberSecQwen-4B).

## Quantization

| Parameter | Value |
|---|---|
| Method | GGUF Q4_K_M (llama.cpp) |
| Weight precision | 4-bit (Q4_K_M = 4-bit block-scaled with k-quant importance) |
| Quantization tool | llama.cpp (build from master) |
| Conversion tool | convert_hf_to_gguf.py |
| Quantization hardware | Modal A10G |
| File | cybersecqwen-4b-Q4_K_M.gguf (2.5 GB) |

## CTI-Bench Evaluation

Evaluated under the [Foundation-Sec-8B protocol](https://arxiv.org/abs/2504.21039):
- Temperature 0.3, max_tokens 512, concurrency 8
- 5 independent trials, zero-shot (no system prompt)
- llama.cpp server on Modal L4 GPU

| Task | GGUF Q4_K_M | AWQ 4-bit | FP16 Reference |
|---|---|---|---|
| CTI-MCQ (2,500 items) | {mcq['accuracy']:.4f} ± {mcq['std']:.4f} | **0.5921 ± 0.0083** | 0.5868 ± 0.0029 |
| CTI-RCM (1,000 items) | **{rcm_data['accuracy']:.4f} ± {rcm_data['std']:.4f}** | 0.5814 ± 0.0025 | 0.6664 ± 0.0023 |

**Key findings:**
- **CTI-RCM** (CVE→CWE classification): GGUF Q4_K_M is the best quantized variant (-{abs(rcm_data['delta_vs_fp16'])*100:.1f} pts vs FP16). Superior to AWQ 4-bit by +4.4 points.
- **CTI-MCQ** (CTI knowledge): AWQ 4-bit performs better than GGUF for multiple-choice questions.
- GGUF preserves task-specific classification accuracy better due to block-wise k-quant importance scaling.

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
for i, t in enumerate(rcm_data['trials']):
    readme += f"| {i+1} | {t['seed']} | {t['accuracy']:.4f} |\n"

readme += """
## Quantization variants

| Variant | CTI-MCQ | CTI-RCM | Size | Engine |
|---|---|---|---|---|
| [AWQ 4-bit](https://huggingface.co/ree2raz/CyberSecQwen-4B-AWQ) | 0.5921 | 0.5814 | 2.7 GB | vLLM |
| [GGUF Q4_K_M](https://huggingface.co/ree2raz/CyberSecQwen-4B-GGUF) | 0.5368 | 0.6254 | 2.5 GB | llama.cpp |

Choose GGUF for vulnerability classification, AWQ for MCQ/general chat.

## Usage with llama.cpp

```bash
# Download
wget https://huggingface.co/ree2raz/CyberSecQwen-4B-GGUF/resolve/main/cybersecqwen-4b-Q4_K_M.gguf

# Serve
./llama-server -m cybersecqwen-4b-Q4_K_M.gguf --host 0.0.0.0 --port 8080 -ngl 99 -c 4096
```

## Model Size

| Format | Size |
|---|---|
| Original FP16 | ~8 GB |
| GGUF Q4_K_M | ~2.5 GB |

## Citation

```bibtex
@misc{{cybersecqwen2026,
  title  = {{CyberSecQwen-4B: A Compact CTI Specialist}},
  author = {{Mulia, Samuel}},
  year   = {{2026}},
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
    repo_id="ree2raz/CyberSecQwen-4B-GGUF",
    commit_message="Add CTI-Bench evaluation results (GGUF Q4_K_M)",
)
print("GGUF model card created!")
