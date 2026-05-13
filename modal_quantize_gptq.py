#!/usr/bin/env python3
"""GPTQ 4-bit quantization with improved RCM calibration on Modal A100.

Why GPTQ over AWQ:
  - Hessian-based per-layer sensitivity analysis (not activation-based)
  - Better preserves discriminative features for fine-grained classification
  - desc_act=True retains activation order for better downstream accuracy

Uses same calibration data as AWQ v2:
  - 320 samples (256 RCM + 64 MCQ, chat-template formatted)

Pushes to: ree2raz/CyberSecQwen-4B-GPTQ
"""
import json
import os
import time
from typing import Any

import modal

app = modal.App("cybersecqwen-gptq-quantize")

MODEL_ID = "lablab-ai-amd-developer-hackathon/CyberSecQwen-4B"
TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_REPO = "ree2raz/CyberSecQwen-4B-GPTQ"
CALIB_RCM = 256
CALIB_MCQ = 64

hf_cache = modal.Volume.from_name("inference-bench-hf-cache", create_if_missing=True)


def make_image():
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch>=2.3.0,<2.7.0")
        .pip_install("transformers==4.51.3")
        .pip_install("auto-gptq", "peft==0.13.2", extra_index_url="https://huggingface.github.io/autogptq-index/whl/cu124/")
        .pip_install("datasets>=2.18.0", "accelerate>=0.28.0", "huggingface_hub>=0.24.0")
        .env({"HF_HOME": "/hf_cache"})
    )


@app.cls(
    image=make_image(),
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={"/hf_cache": hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
class GPTQQuantizer:
    @modal.method()
    def quantize(self) -> dict[str, Any]:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from datasets import load_dataset
        from transformers import AutoTokenizer
        import torch

        print("[1/7] Loading tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=False)

        print("[2/7] Loading calibration data from CTI-Bench ...")
        calib = []

        ds_rcm = load_dataset("AI4Sec/cti-bench", "cti-rcm", split="test")
        for p in ds_rcm.select(range(min(len(ds_rcm), CALIB_RCM)))["Prompt"]:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False,
            )
            calib.append(formatted)
        print(f"  RCM: {len(calib)} prompts (chat-template applied)")

        ds_mcq = load_dataset("AI4Sec/cti-bench", "cti-mcq", split="test")
        for p in ds_mcq.select(range(min(len(ds_mcq), CALIB_MCQ)))["Prompt"]:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False,
            )
            calib.append(formatted)
        print(f"  Total calibration: {len(calib)} samples (80% RCM / 20% MCQ)")

        print("[3/7] Tokenizing calibration data for GPTQ ...")
        tokenized = tokenizer(
            calib,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3072,
        )
        examples = [
            {
                "input_ids": tokenized["input_ids"][i].to("cuda"),
                "attention_mask": tokenized["attention_mask"][i].to("cuda"),
            }
            for i in range(len(calib))
        ]

        print("[4/7] Building GPTQ config (bits=4, group_size=128, desc_act=True) ...")
        quant_config = BaseQuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=True,
            damp_percent=0.01,
            sym=True,
        )

        print(f"[5/7] Loading FP16 model: {MODEL_ID}")
        t0 = time.perf_counter()
        model = AutoGPTQForCausalLM.from_pretrained(
            MODEL_ID,
            quantize_config=quant_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

        print("[6/7] Running GPTQ quantization ...")
        t0 = time.perf_counter()
        model.quantize(
            examples,
            batch_size=4,
            use_triton=False,
        )
        print(f"  Quantized in {time.perf_counter() - t0:.1f}s")

        save_dir = "/tmp/gptq_quantized"
        print(f"[7/7] Saving to {save_dir} and pushing to {OUTPUT_REPO} ...")
        model.save_quantized(save_dir, use_safetensors=True)
        tokenizer.save_pretrained(save_dir)

        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(OUTPUT_REPO, exist_ok=True)
        api.upload_folder(
            repo_id=OUTPUT_REPO,
            folder_path=save_dir,
            commit_message=(
                "GPTQ 4-bit: 320 calib samples (256 RCM + 64 MCQ), "
                "chat-template applied, desc_act=True, group_size=128"
            ),
        )

        print(f"\nDone! Pushed to https://huggingface.co/{OUTPUT_REPO}")
        return {
            "status": "ok",
            "repo": f"https://huggingface.co/{OUTPUT_REPO}",
            "calib_samples": len(calib),
            "bits": 4,
            "desc_act": True,
        }


@app.local_entrypoint()
def main():
    result = GPTQQuantizer().quantize.remote()
    print(json.dumps(result, indent=2))
