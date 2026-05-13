#!/usr/bin/env python3
"""Evaluate GPTQ-quantized CyberSecQwen-4B on CTI-Bench using vLLM on Modal L4.

Protocol: matches Foundation-Sec-8B (arXiv:2504.21039 §B.3-B.4)
  - Zero-shot, no system prompt
  - Dataset's Prompt column as user message (vLLM applies chat template)
  - Temperature 0.3, max_tokens 512
  - Concurrency 32, 5 independent trials
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

app = modal.App("cybersecqwen-eval-gptq")

MODEL_GPTQ = "ree2raz/CyberSecQwen-4B-GPTQ"
DATASET_ID = "AI4Sec/cti-bench"
GPU_TYPE = "L4"
NUM_TRIALS = 5
MAX_TOKENS = 512
TEMPERATURE = 0.3
CONCURRENCY = 32
SERVER_URL = "http://localhost:8000"

REF_SCORES: dict[str, dict] = {
    "cti-mcq": {"accuracy": 0.5868, "std": 0.0029},
    "cti-rcm": {"accuracy": 0.6664, "std": 0.0023},
}

hf_cache = modal.Volume.from_name("inference-bench-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name(
    "cybersecqwen-eval-results", create_if_missing=True
)

_SYMLINK_PYTHON = [
    "RUN for p in /usr/local/bin/python3 /usr/bin/python3 /opt/conda/bin/python3; do "
    "if [ -f \"$p\" ]; then ln -sf \"$p\" /usr/local/bin/python && break; fi; done "
    "&& python --version"
]


def make_vllm_image():
    return (
        modal.Image.from_registry(
            "vllm/vllm-openai:v0.20.1",
            setup_dockerfile_commands=_SYMLINK_PYTHON,
        )
        .entrypoint([])
        .pip_install("httpx>=0.27", "numpy>=1.26", "datasets>=2.18.0")
        .env({
            "HF_HOME": "/hf_cache",
            "VLLM_LOGGING_LEVEL": "WARNING",
        })
    )


VLLM_SERVER_ARGS = [
    "vllm", "serve", MODEL_GPTQ,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-num-seqs", "64",
    "--max-model-len", "4096",
    "--quantization", "gptq_marlin",
    "--dtype", "float16",
    "--gpu-memory-utilization", "0.90",
    "--no-enable-log-requests",
]


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
    """Extract first CWE-XXX from model response (matches original protocol)."""
    m = re.search(r"CWE-?\d+", response, re.IGNORECASE)
    if m:
        raw = m.group(0).upper()
        if "-" not in raw:
            raw = raw.replace("CWE", "CWE-")
        return raw
    return ""


def wait_for_server(timeout: int = 600, interval: int = 5) -> bool:
    import httpx
    waited = 0
    while waited < timeout:
        for endpoint in ["/health", "/v1/models"]:
            try:
                resp = httpx.get(f"{SERVER_URL}{endpoint}", timeout=5)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
        time.sleep(interval)
        waited += interval
    return False


def _load_dataset(task: str) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, task, split="test")
    return [dict(item) for item in ds]


def _run_trial(task: str, items: list[dict], seed: int) -> dict:
    import httpx
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    prompts = [item["Prompt"] for item in items]
    gt_list = [item["GT"].strip().upper() for item in items]

    sem = asyncio.Semaphore(CONCURRENCY)

    async def _send(prompt: str, gt: str) -> dict:
        async with sem:
            async with httpx.AsyncClient(timeout=300) as client:
                payload = {
                    "model": MODEL_GPTQ,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                }
                try:
                    resp = await client.post(
                        f"{SERVER_URL}/v1/chat/completions",
                        json=payload,
                    )
                    if resp.status_code != 200:
                        return {
                            "correct": False,
                            "parseable": False,
                            "error": f"HTTP {resp.status_code}",
                        }
                    body = resp.json()
                    response = body["choices"][0]["message"]["content"]
                except Exception as e:
                    return {"correct": False, "parseable": False, "error": str(e)}

        if task == "cti-mcq":
            pred = extract_mcq_answer(response)
        else:
            pred = extract_cwe_answer(response)

        return {
            "gt": gt,
            "pred": pred,
            "correct": pred == gt,
            "parseable": bool(pred),
        }

    async_tasks = [_send(p, g) for p, g in zip(prompts, gt_list)]

    async def _run_all():
        return await asyncio.gather(*async_tasks)

    results = asyncio.run(_run_all())

    correct = sum(1 for r in results if r.get("correct"))
    parseable = sum(1 for r in results if r.get("parseable"))
    errors = sum(1 for r in results if "error" in r)

    return {
        "seed": seed,
        "correct": correct,
        "total": len(results),
        "accuracy": correct / len(results),
        "parseable": parseable,
        "errors": errors,
    }


@app.cls(
    image=make_vllm_image(),
    gpu=GPU_TYPE,
    timeout=60 * 60 * 4,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class VllmEval:
    @modal.enter()
    def start_server(self):
        print(f"Loading AWQ model: {MODEL_GPTQ}")
        self.proc = subprocess.Popen(VLLM_SERVER_ARGS)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=600), "vLLM server failed to start"
        print(f"[start] vLLM ready in {time.perf_counter() - t0:.1f}s")

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
        import numpy as np

        header = {
            "model": MODEL_GPTQ,
        "quantization": "GPTQ 4-bit (group_size=128, desc_act=True)",
        "gpu": GPU_TYPE,
        "protocol": {
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "num_trials": NUM_TRIALS,
                "concurrency": CONCURRENCY,
                "prompt_source": "dataset_Prompt_column",
                "system_prompt": "none",
                "chat_template": "vllm_auto_applied",
            },
            "tasks": {},
        }

        for task in tasks:
            print(f"\n[task] {task}")
            items = _load_dataset(task)
            print(f"[task] Loaded {len(items)} items")

            trial_results = []
            for trial_idx in range(NUM_TRIALS):
                seed = 42 + trial_idx
                print(
                    f"[trial] {trial_idx + 1}/{NUM_TRIALS} seed={seed} ...",
                    end=" ",
                    flush=True,
                )
                t0 = time.perf_counter()
                result = _run_trial(task, items, seed)
                elapsed = time.perf_counter() - t0
                print(
                    f"acc={result['accuracy']:.4f} "
                    f"({result['correct']}/{result['total']}) "
                    f"parseable={result['parseable']} "
                    f"errors={result['errors']} "
                    f"took={elapsed:.1f}s"
                )
                trial_results.append(result)

                # incremental snapshot after each trial
                partial = {
                    **header,
                    "tasks": {
                        task: {
                            "accuracy": round(
                                float(np.mean([t["accuracy"] for t in trial_results])), 4
                            ),
                            "completed_trials": trial_idx + 1,
                            "trials": [
                                {"seed": t["seed"], "accuracy": t["accuracy"],
                                 "correct": t["correct"], "total": t["total"]}
                                for t in trial_results
                            ],
                        }
                    },
                }
                self._save_snapshot(partial, f" {task} trial {trial_idx+1}/{NUM_TRIALS}")

            accs = [t["accuracy"] for t in trial_results]
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            total_correct = sum(t["correct"] for t in trial_results)
            total_items = sum(t["total"] for t in trial_results)
            total_parseable = sum(t["parseable"] for t in trial_results)

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
                "correct": total_correct,
                "total": total_items,
                "parseable": total_parseable,
                "parseable_rate": round(total_parseable / total_items, 4),
                "delta_vs_fp16": delta,
                "trials": [
                    {"seed": t["seed"], "accuracy": t["accuracy"],
                     "correct": t["correct"], "total": t["total"]}
                    for t in trial_results
                ],
            }
            self._save_snapshot(header, f" task {task} done")

        print(f"\n[final] All tasks complete.")
        return header


@app.local_entrypoint()
def main(task: str = "all"):
    tasks = ["cti-mcq", "cti-rcm"] if task == "all" else [task]

    print("=" * 60)
    print("  CyberSecQwen-4B-GPTQ CTI-Bench Evaluation (vLLM)")
    print("=" * 60)
    print(f"  Model:       {MODEL_GPTQ}")
    print(f"  GPU:         {GPU_TYPE}")
    print(f"  Tasks:       {tasks}")
    print(f"  Trials:      {NUM_TRIALS}")
    print(f"  Temp:        {TEMPERATURE}")
    print(f"  Max tokens:  {MAX_TOKENS}")
    print(f"  Concurrency: {CONCURRENCY}")
    print("=" * 60)

    bench = VllmEval()
    results = bench.eval_all.remote(tasks)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(
        f"{'Task':<12} {'AWQ Acc':>10} {'Std':>10} "
        f"{'FP16 Ref':>10} {'\u0394':>10}"
    )
    print("-" * 60)
    for t in tasks:
        tr = results["tasks"][t]
        ref = REF_SCORES.get(t, {}).get("accuracy")
        ref_str = f"{ref:.4f}" if ref else "N/A"
        delta = tr.get("delta_vs_fp16")
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        print(
            f"{t:<12} {tr['accuracy']:>10.4f} {tr['std']:>10.4f} "
            f"{ref_str:>10} {delta_str:>10}"
        )
    print("=" * 60)
    print("\nResults persisted to Modal volume: cybersecqwen-eval-results")
