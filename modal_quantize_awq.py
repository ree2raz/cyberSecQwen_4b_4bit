#!/usr/bin/env python3
"""Quantize CyberSecQwen-4B (FP16) to AWQ 4-bit on Modal A100.

Pushes quantized model to HF: ree2raz/CyberSecQwen-4B-AWQ
"""
import json
import os
import time
from typing import Any

import modal

app = modal.App("cybersecqwen-awq-quantize")

MODEL_ID = "lablab-ai-amd-developer-hackathon/CyberSecQwen-4B"
OUTPUT_REPO = "ree2raz/CyberSecQwen-4B-AWQ"
CALIBRATION_SAMPLES = 128

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
class AWQQuantizer:
    @modal.method()
    def quantize(self) -> dict[str, Any]:
        from awq import AutoAWQForCausalLM
        from datasets import load_dataset
        from transformers import AutoTokenizer

        print(f"[1/5] Loading calibration data from CTI-Bench ...")
        calib = []
        for task in ["cti-mcq", "cti-rcm"]:
            ds = load_dataset("AI4Sec/cti-bench", task, split="test")
            n = min(len(ds), CALIBRATION_SAMPLES // 2)
            calib.extend(ds.select(range(n))["Prompt"])
        calib = calib[:CALIBRATION_SAMPLES]
        print(f"  {len(calib)} calibration samples")

        print(f"[2/5] Loading tokenizer: {MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        print(f"[3/5] Loading model in float16: {MODEL_ID}")
        t0 = time.perf_counter()
        import torch
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

        print(f"[4/5] Running AWQ quantization (group_size=128, w_bit=4) ...")
        t0 = time.perf_counter()
        model.quantize(tokenizer, quant_config=quant_config, calib_data=calib)
        print(f"  Quantized in {time.perf_counter() - t0:.1f}s")

        save_dir = "/tmp/awq_quantized"
        print(f"[5/5] Saving to {save_dir} and pushing to {OUTPUT_REPO} ...")
        model.save_quantized(save_dir)
        tokenizer.save_pretrained(save_dir)

        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(OUTPUT_REPO, exist_ok=True)
        api.upload_folder(
            repo_id=OUTPUT_REPO,
            folder_path=save_dir,
            commit_message="AWQ 4-bit quantization (group_size=128, w_bit=4, zero_point=True)",
        )

        print(f"\nDone! Pushed to https://huggingface.co/{OUTPUT_REPO}")
        return {"status": "ok", "repo": f"https://huggingface.co/{OUTPUT_REPO}"}


@app.local_entrypoint()
def main():
    result = AWQQuantizer().quantize.remote()
    print(json.dumps(result, indent=2))
