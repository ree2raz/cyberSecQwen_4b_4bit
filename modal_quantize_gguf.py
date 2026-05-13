#!/usr/bin/env python3
"""Convert + quantize CyberSecQwen-4B to GGUF Q4_K_M on Modal A100.

GGUF requires no calibration — llama.cpp's convert + quantize tools handle it.
Pushes single .gguf file to: ree2raz/CyberSecQwen-4B-GGUF
"""
import json
import os
import subprocess
import sys
import time
from typing import Any

import modal

app = modal.App("cybersecqwen-gguf-quantize")

MODEL_ID = "lablab-ai-amd-developer-hackathon/CyberSecQwen-4B"
OUTPUT_REPO = "ree2raz/CyberSecQwen-4B-GGUF"
GGUF_QUANT = "Q4_K_M"
LLAMACPP_VERSION = "master"

hf_cache = modal.Volume.from_name("inference-bench-hf-cache", create_if_missing=True)


def make_image():
    apt_pkgs = "cmake build-essential curl git".split()
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(*apt_pkgs)
        .pip_install("huggingface_hub>=0.24.0", "transformers>=4.44.0", "torch>=2.0.0", "sentencepiece")
        .env({"HF_HOME": "/hf_cache"})
    )


@app.cls(
    image=make_image(),
    gpu="A10G",
    timeout=60 * 60 * 2,
    volumes={"/hf_cache": hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
class GGUFQuantizer:
    @modal.method()
    def quantize(self) -> dict[str, Any]:
        print(f"[1/5] Cloning llama.cpp (release {LLAMACPP_VERSION}) ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", LLAMACPP_VERSION,
             "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama.cpp"],
            check=True,
        )

        print("[2/5] Building llama-quantize via CMake ...")
        ncpu = os.cpu_count() or 4
        subprocess.run(["cmake", "-B", "build", "-S", "."], cwd="/tmp/llama.cpp", check=True)
        subprocess.run(["cmake", "--build", "build", "--target", "llama-quantize", "-j", str(ncpu)], cwd="/tmp/llama.cpp", check=True)

        print("[3/5] Downloading FP16 model from HF ...")
        from huggingface_hub import snapshot_download
        t0 = time.perf_counter()
        model_dir = snapshot_download(MODEL_ID, cache_dir="/hf_cache")
        print(f"  Downloaded in {time.perf_counter() - t0:.1f}s")

        f16_gguf = "/tmp/model-F16.gguf"
        print(f"[4/5] Converting to GGUF FP16 -> {f16_gguf} ...")
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, "/tmp/llama.cpp/convert_hf_to_gguf.py",
             model_dir, "--outfile", f16_gguf, "--outtype", "f16"],
            check=True,
        )
        print(f"  Converted in {time.perf_counter() - t0:.1f}s")

        out_gguf = f"/tmp/model-{GGUF_QUANT}.gguf"
        print(f"[5/5] Quantizing to {GGUF_QUANT} ...")
        t0 = time.perf_counter()
        subprocess.run(
            ["/tmp/llama.cpp/build/bin/llama-quantize", f16_gguf, out_gguf, GGUF_QUANT],
            check=True,
        )
        print(f"  Quantized in {time.perf_counter() - t0:.1f}s")

        file_size = os.path.getsize(out_gguf) / (1024**3)
        print(f"  GGUF file size: {file_size:.2f} GB")

        print(f"\nPushing to https://huggingface.co/{OUTPUT_REPO} ...")
        from huggingface_hub import HfApi, create_repo
        create_repo(OUTPUT_REPO, exist_ok=True)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=out_gguf,
            path_in_repo=f"cybersecqwen-4b-{GGUF_QUANT}.gguf",
            repo_id=OUTPUT_REPO,
            commit_message=f"GGUF {GGUF_QUANT} quantization via llama.cpp {LLAMACPP_VERSION}",
        )

        print(f"\nDone! https://huggingface.co/{OUTPUT_REPO}")
        return {
            "status": "ok",
            "repo": f"https://huggingface.co/{OUTPUT_REPO}",
            "quantization": GGUF_QUANT,
            "file_size_gb": round(file_size, 2),
        }


@app.local_entrypoint()
def main():
    result = GGUFQuantizer().quantize.remote()
    print(json.dumps(result, indent=2))
