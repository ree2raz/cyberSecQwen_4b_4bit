#!/usr/bin/env python3
"""Evaluate GGUF-quantized CyberSecQwen-4B on CTI-Bench using llama.cpp on Modal L4.

Protocol: matches Foundation-Sec-8B (arXiv:2504.21039 §B.3-B.4)
  - Zero-shot, no system prompt
  - Dataset's Prompt column as user message
  - Temperature 0.3, max_tokens 512
  - 5 independent trials per task
  - Metric: strict accuracy

Reference FP16 scores (from CyberSecQwen-4B model card):
  - CTI-MCQ: 0.5868 ± 0.0029
  - CTI-RCM: 0.6664 ± 0.0023
"""
import asyncio
import json
import os
import random
import re
import subprocess
import time
from typing import Any

import modal

app = modal.App("cybersecqwen-eval-llamacpp")

MODEL_REPO = "ree2raz/CyberSecQwen-4B-GGUF"
MODEL_FILE = "cybersecqwen-4b-Q4_K_M.gguf"
DATASET_ID = "AI4Sec/cti-bench"
GPU_TYPE = "L4"
NUM_TRIALS = 5
MAX_TOKENS = 512
TEMPERATURE = 0.3
CONCURRENCY = 8
SERVER_URL = "http://localhost:8080"

REF_SCORES: dict[str, dict] = {
    "cti-mcq": {"accuracy": 0.5868, "std": 0.0029},
    "cti-rcm": {"accuracy": 0.6664, "std": 0.0023},
}

hf_cache = modal.Volume.from_name("inference-bench-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("cybersecqwen-eval-results", create_if_missing=True)


def make_image():
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("curl", "libcurl4", "libgomp1")
        .pip_install("httpx>=0.27", "numpy>=1.26", "datasets>=2.18.0", "pyyaml>=6.0")
        .env({
            "HF_HOME": "/hf_cache",
            "LD_LIBRARY_PATH": "/hf_cache/llamacpp",
        })
    )


def extract_mcq_answer(response: str) -> str:
    text = response.strip().upper()
    for pat in [
        r"^(A|B|C|D)$",
        r"(?:^|\n)(A|B|C|D)(?:\s|$|\n|\.)",
        r"(?:answer|choice|option)[:\s]+([A-D])\b",
    ]:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            return m.group(1)
    return ""


def extract_cwe_answer(response: str) -> str:
    m = re.search(r"CWE-?\d+", response, re.IGNORECASE)
    if m:
        raw = m.group(0).upper()
        if "-" not in raw:
            raw = raw.replace("CWE", "CWE-")
        return raw
    return ""


def wait_for_server(timeout: int = 300, interval: int = 3) -> bool:
    import httpx
    waited = 0
    while waited < timeout:
        try:
            resp = httpx.get(f"{SERVER_URL}/health", timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
        waited += interval
    return False


def download_llama_server():
    """Download pre-built llama.cpp server binary."""
    import stat
    import tarfile

    bin_dir = "/hf_cache/llamacpp"
    os.makedirs(bin_dir, exist_ok=True)

    # Search for existing binary first
    for root, dirs, files in os.walk(bin_dir):
        if "llama-server" in files:
            server_bin = os.path.join(root, "llama-server")
            os.chmod(server_bin, os.stat(server_bin).st_mode | stat.S_IEXEC)
            return server_bin

    # Download pre-built binary from llama.cpp releases
    tag = "b4933"
    url = f"https://github.com/ggerganov/llama.cpp/releases/download/{tag}/llama-{tag}-bin-ubuntu-x64.tar.gz"
    tarball = "/tmp/llama.tar.gz"
    subprocess.run(["wget", "-q", url, "-O", tarball], check=True)

    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(path=bin_dir, filter="data")

    for root, dirs, files in os.walk(bin_dir):
        if "llama-server" in files:
            server_bin = os.path.join(root, "llama-server")
            os.chmod(server_bin, os.stat(server_bin).st_mode | stat.S_IEXEC)
            # Set LD_LIBRARY_PATH for shared libs
            lib_dir = os.path.dirname(server_bin)
            os.environ["LD_LIBRARY_PATH"] = lib_dir
            return server_bin

    raise FileNotFoundError("llama-server binary not found in release tarball")


@app.cls(
    image=make_image(),
    gpu=GPU_TYPE,
    timeout=60 * 60 * 4,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class LlamaEval:
    @modal.enter()
    def start_server(self):
        import stat
        from huggingface_hub import hf_hub_download

        server_bin = "/hf_cache/llamacpp/llama-server"
        st = os.stat(server_bin)
        if not (st.st_mode & stat.S_IEXEC):
            os.chmod(server_bin, st.st_mode | stat.S_IEXEC)

        print(f"Downloading GGUF model: {MODEL_REPO}/{MODEL_FILE} ...")
        gguf_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            cache_dir="/hf_cache",
        )

        self.proc = subprocess.Popen([
            server_bin,
            "-m", gguf_path,
            "--host", "0.0.0.0",
            "--port", "8080",
            "-ngl", "99",
            "-c", "4096",
            "--parallel", str(CONCURRENCY),
            "-np", str(CONCURRENCY),
        ])

        t0 = time.perf_counter()
        assert wait_for_server(timeout=300), "llama-server failed to start"
        print(f"[start] llama-server ready in {time.perf_counter() - t0:.1f}s")

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)

    def _save_snapshot(self, data: dict, tag: str = ""):
        path = "/results/eval_results.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        results_vol.commit()
        print(f"  [snapshot{tag}] Saved to {path}")

    @modal.method()
    def eval_all(self, tasks: list[str]) -> dict[str, Any]:
        from datasets import load_dataset
        import httpx
        import numpy as np

        header = {
            "model": f"{MODEL_REPO}/{MODEL_FILE}",
            "quantization": "GGUF Q4_K_M",
            "gpu": GPU_TYPE,
            "protocol": {
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "num_trials": NUM_TRIALS,
                "concurrency": CONCURRENCY,
                "prompt_source": "dataset_Prompt_column",
                "system_prompt": "none",
            },
            "tasks": {},
        }

        for task in tasks:
            print(f"\n[task] {task}")
            ds = load_dataset(DATASET_ID, task, split="test")
            items = [dict(item) for item in ds]
            print(f"[task] Loaded {len(items)} items")

            trial_results = []
            for trial_idx in range(NUM_TRIALS):
                seed = 42 + trial_idx
                random.seed(seed)
                np.random.seed(seed)

                print(f"[trial] {trial_idx + 1}/{NUM_TRIALS} seed={seed} ...", end=" ", flush=True)
                t0 = time.perf_counter()

                prompts = [item["Prompt"] for item in items]
                gt_list = [item["GT"].strip().upper() for item in items]

                sem = asyncio.Semaphore(CONCURRENCY)

                async def _send(prompt: str) -> str:
                    async with sem:
                        async with httpx.AsyncClient(timeout=300) as client:
                            payload = {
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": MAX_TOKENS,
                                "temperature": TEMPERATURE,
                                "stream": False,
                            }
                            for attempt in range(3):
                                try:
                                    resp = await client.post(
                                        f"{SERVER_URL}/v1/chat/completions",
                                        json=payload,
                                    )
                                    if resp.status_code == 200:
                                        body = resp.json()
                                        return body["choices"][0]["message"]["content"]
                                except Exception:
                                    await asyncio.sleep(2)
                            return ""

                async def _run_all():
                    return await asyncio.gather(*[_send(p) for p in prompts])

                responses = asyncio.run(_run_all())

                correct = 0
                parseable = 0
                errors = 0
                for resp, gt in zip(responses, gt_list):
                    if not resp:
                        errors += 1
                        continue
                    if task == "cti-mcq":
                        pred = extract_mcq_answer(resp)
                    else:
                        pred = extract_cwe_answer(resp)
                    if pred:
                        parseable += 1
                    if pred == gt:
                        correct += 1

                elapsed = time.perf_counter() - t0
                acc = correct / len(items)
                print(f"acc={acc:.4f} ({correct}/{len(items)}) parseable={parseable} errors={errors} took={elapsed:.1f}s")

                trial_results.append({
                    "seed": seed,
                    "accuracy": acc,
                    "correct": correct,
                    "total": len(items),
                })

                partial = {
                    **header,
                    "tasks": {
                        task: {
                            "accuracy": round(float(np.mean([t["accuracy"] for t in trial_results])), 4),
                            "completed_trials": trial_idx + 1,
                            "trials": trial_results,
                        }
                    },
                }
                self._save_snapshot(partial, f" {task} trial {trial_idx+1}/{NUM_TRIALS}")

            accs = [t["accuracy"] for t in trial_results]
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))

            ref = REF_SCORES.get(task, {})
            ref_acc = ref.get("accuracy")
            delta = round(mean_acc - ref_acc, 4) if ref_acc is not None else None

            print(f"[result] {task}: {mean_acc:.4f} ± {std_acc:.4f}")
            if delta is not None:
                sign = "+" if delta >= 0 else ""
                print(f"[result]   vs FP16 ref ({ref_acc}): Δ={sign}{delta}")

            header["tasks"][task] = {
                "accuracy": round(mean_acc, 4),
                "std": round(std_acc, 4),
                "delta_vs_fp16": delta,
                "trials": trial_results,
            }
            self._save_snapshot(header, f" task {task} done")

        print(f"\n[final] All tasks complete.")
        return header


@app.local_entrypoint()
def main(task: str = "all"):
    tasks = ["cti-mcq", "cti-rcm"] if task == "all" else [task]

    print("=" * 60)
    print("  CyberSecQwen-4B-GGUF CTI-Bench Evaluation (llama.cpp)")
    print("=" * 60)
    print(f"  Model:       {MODEL_REPO}/{MODEL_FILE}")
    print(f"  GPU:         {GPU_TYPE}")
    print(f"  Tasks:       {tasks}")
    print(f"  Trials:      {NUM_TRIALS}")
    print(f"  Temp:        {TEMPERATURE}")
    print(f"  Max tokens:  {MAX_TOKENS}")
    print(f"  Concurrency: {CONCURRENCY}")
    print("=" * 60)

    bench = LlamaEval()
    results = bench.eval_all.remote(tasks)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Task':<12} {'GGUF Acc':>10} {'Std':>10} {'FP16 Ref':>10} {'Δ':>10}")
    print("-" * 60)
    for t in tasks:
        tr = results["tasks"][t]
        ref = REF_SCORES.get(t, {}).get("accuracy")
        ref_str = f"{ref:.4f}" if ref else "N/A"
        delta = tr.get("delta_vs_fp16")
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        print(f"{t:<12} {tr['accuracy']:>10.4f} {tr['std']:>10.4f} {ref_str:>10} {delta_str:>10}")
    print("=" * 60)
    print("\nResults persisted to Modal volume: cybersecqwen-eval-results")
