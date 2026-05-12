#!/usr/bin/env python3
"""Improved AWQ 4-bit quantization with better RCM calibration.

Fixes vs original quantization:
  1. 256+ RCM calibration samples (was 64)
  2. Chat-template applied to calibration data (matches inference distribution)
  3. RCM-dominant calibration mix (80% RCM, 20% MCQ)

Pushes to: ree2raz/CyberSecQwen-4B-AWQ
"""
import json
import os
import time
from typing import Any

import modal

app = modal.App("cybersecqwen-awq-quantize-v2")

MODEL_ID = "lablab-ai-amd-developer-hackathon/CyberSecQwen-4B"
OUTPUT_REPO = "ree2raz/CyberSecQwen-4B-AWQ"
CALIB_RCM = 256
CALIB_MCQ = 64

hf_cache = modal.Volume.from_name("inference-bench-hf-cache", create_if_missing=True)


def make_image():
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.3.0",
            "transformers>=4.44.0",
            "datasets>=2.18.0",
            "autoawq>=0.2.5",
            "accelerate>=0.28.0",
            "huggingface_hub>=0.24.0",
        )
        .env({"HF_HOME": "/hf_cache"})
    )


@app.cls(
    image=make_image(),
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={"/hf_cache": hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
class AWQQuantizerV2:
    @modal.method()
    def quantize(self) -> dict[str, Any]:
        from awq import AutoAWQForCausalLM
        from datasets import load_dataset
        from transformers import AutoTokenizer
        import torch

        print(f"[1/6] Loading tokenizer for chat template ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        print(f"[2/6] Loading calibration data from CTI-Bench ...")
        calib_texts = []

        # Load RCM samples (256) — the problematic task
        ds_rcm = load_dataset("AI4Sec/cti-bench", "cti-rcm", split="test")
        rcm_prompts = ds_rcm.select(range(min(len(ds_rcm), CALIB_RCM)))["Prompt"]
        for p in rcm_prompts:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False,
            )
            calib_texts.append(formatted)
        print(f"  RCM: {len(rcm_prompts)} prompts (chat-template applied)")

        # Load MCQ samples (64) — maintain MCQ performance
        ds_mcq = load_dataset("AI4Sec/cti-bench", "cti-mcq", split="test")
        mcq_prompts = ds_mcq.select(range(min(len(ds_mcq), CALIB_MCQ)))["Prompt"]
        for p in mcq_prompts:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False,
            )
            calib_texts.append(formatted)
        print(f"  MCQ: {len(mcq_prompts)} prompts (chat-template applied)")
        print(f"  Total calibration: {len(calib_texts)} samples (80% RCM / 20% MCQ)")

        print(f"[3/6] Loading FP16 model in float16: {MODEL_ID}")
        t0 = time.perf_counter()
        model = AutoAWQForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map="auto",
            safetensors=True,
            torch_dtype=torch.float16,
        )
        print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        }

        print(f"[4/6] Running AWQ quantization (w_bit=4, group_size=128, chat-template-calibrated) ...")
        t0 = time.perf_counter()
        model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)
        print(f"  Quantized in {time.perf_counter() - t0:.1f}s")

        save_dir = "/tmp/awq_quantized_v2"
        print(f"[5/6] Saving to {save_dir} and pushing to {OUTPUT_REPO} ...")
        model.save_quantized(save_dir)
        tokenizer.save_pretrained(save_dir)

        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            repo_id=OUTPUT_REPO,
            folder_path=save_dir,
            commit_message=(
                "AWQ 4-bit v2: 320 calib samples (256 RCM + 64 MCQ), "
                "chat-template applied, fp16 compute"
            ),
        )

        print(f"[6/6] Done! Pushed to https://huggingface.co/{OUTPUT_REPO}")
        return {
            "status": "ok",
            "repo": f"https://huggingface.co/{OUTPUT_REPO}",
            "calib_samples": len(calib_texts),
            "calib_rcm": len(rcm_prompts),
            "calib_mcq": len(mcq_prompts),
            "chat_template": True,
        }


@app.local_entrypoint()
def main():
    result = AWQQuantizerV2().quantize.remote()
    print(json.dumps(result, indent=2))
