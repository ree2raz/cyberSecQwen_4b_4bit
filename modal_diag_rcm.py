#!/usr/bin/env python3
"""Diagnostic: compare first-CWE vs last-CWE extraction on CTI-RCM.
Runs 100 samples (1 trial), saves raw responses + both extraction scores."""
import asyncio
import json
import os
import re
import subprocess
import time
from typing import Any

import modal

app = modal.App("cybersecqwen-rcm-diag")

MODEL_AWQ = "ree2raz/CyberSecQwen-4B-AWQ"
DATASET_ID = "AI4Sec/cti-bench"
SERVER_URL = "http://localhost:8000"
MAX_TOKENS = 512
TEMPERATURE = 0.3
CONCURRENCY = 32
N_SAMPLES = 200

hf_cache = modal.Volume.from_name("inference-bench-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("cybersecqwen-eval-results", create_if_missing=True)

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
        .env({"HF_HOME": "/hf_cache"})
    )


VLLM_SERVER_ARGS = [
    "vllm", "serve", MODEL_AWQ,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-num-seqs", "64",
    "--max-model-len", "4096",
    "--quantization", "awq_marlin",
    "--dtype", "float16",
    "--gpu-memory-utilization", "0.90",
    "--no-enable-log-requests",
]


def extract_first_cwe(response: str) -> str:
    m = re.search(r"CWE-?\d+", response, re.IGNORECASE)
    if m:
        raw = m.group(0).upper()
        if "-" not in raw:
            raw = raw.replace("CWE", "CWE-")
        return raw
    return ""


def extract_last_cwe(response: str) -> str:
    matches = re.findall(r"CWE-?\d+", response, re.IGNORECASE)
    if matches:
        raw = matches[-1].upper()
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


@app.cls(
    image=make_vllm_image(),
    gpu="L4",
    timeout=60 * 30,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class RCMDiag:
    @modal.enter()
    def start_server(self):
        print(f"Loading {MODEL_AWQ} ...")
        self.proc = subprocess.Popen(VLLM_SERVER_ARGS)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=600), "vLLM server failed to start"
        print(f"[start] vLLM ready in {time.perf_counter() - t0:.1f}s")

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)

    @modal.method()
    def diagnose(self) -> dict[str, Any]:
        from datasets import load_dataset
        import httpx

        ds = load_dataset(DATASET_ID, "cti-rcm", split="test")
        items = [dict(item) for item in ds.select(range(min(len(ds), N_SAMPLES)))]
        print(f"Loaded {len(items)} RCM items")

        prompts = [item["Prompt"] for item in items]
        gt_list = [item["GT"].strip().upper() for item in items]
        sem = asyncio.Semaphore(CONCURRENCY)

        async def _send(prompt: str) -> str:
            async with sem:
                async with httpx.AsyncClient(timeout=300) as client:
                    payload = {
                        "model": MODEL_AWQ,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": MAX_TOKENS,
                        "temperature": TEMPERATURE,
                    }
                    resp = await client.post(
                        f"{SERVER_URL}/v1/chat/completions",
                        json=payload,
                    )
                    body = resp.json()
                    return body["choices"][0]["message"]["content"]

        async def _run_all():
            return await asyncio.gather(*[_send(p) for p in prompts])

        print("Sending requests ...")
        t0 = time.perf_counter()
        responses = asyncio.run(_run_all())
        print(f"Done in {time.perf_counter() - t0:.1f}s")

        results = []
        first_correct = 0
        last_correct = 0
        first_total = 0
        last_total = 0
        match_count = 0

        for i, (resp, gt) in enumerate(zip(responses, gt_list)):
            f_cwe = extract_first_cwe(resp)
            l_cwe = extract_last_cwe(resp)

            f_ok = f_cwe == gt if f_cwe else False
            l_ok = l_cwe == gt if l_cwe else False
            if f_cwe:
                first_total += 1
            if l_cwe:
                last_total += 1
            if f_cwe and l_cwe and f_cwe == l_cwe:
                match_count += 1

            if f_ok:
                first_correct += 1
            if l_ok:
                last_correct += 1

            results.append({
                "idx": i,
                "gt": gt,
                "response": resp[:500],
                "first_cwe": f_cwe,
                "last_cwe": l_cwe,
                "first_correct": f_ok,
                "last_correct": l_ok,
                "diverge": f_cwe != l_cwe if f_cwe and l_cwe else False,
            })

        output = {
            "n_samples": len(results),
            "first_cwe_accuracy": round(first_correct / len(results), 4),
            "last_cwe_accuracy": round(last_correct / len(results), 4),
            "first_parseable": first_total,
            "last_parseable": last_total,
            "first_last_match_rate": round(match_count / min(first_total, last_total), 4) if min(first_total, last_total) > 0 else 0,
            "total_matched_samples": match_count,
            "per_item": results,
        }

        print(f"\n{'='*60}")
        print("CTI-RCM EXTRACTION COMPARISON")
        print(f"{'='*60}")
        print(f"  First-CWE accuracy:  {output['first_cwe_accuracy']:.4f} ({first_correct}/{len(results)})")
        print(f"  Last-CWE accuracy:   {output['last_cwe_accuracy']:.4f} ({last_correct}/{len(results)})")
        print(f"  First parseable:     {first_total}/{len(results)}")
        print(f"  Last parseable:      {last_total}/{len(results)}")
        print(f"  First==Last matches: {match_count}/{min(first_total, last_total)}")
        print(f"{'='*60}")

        path = "/results/rcm_diagnostic.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        results_vol.commit()
        print(f"\nSaved to {path}")

        # Show some divergent examples
        divergent = [r for r in results if r["diverge"]]
        print(f"\nDivergent extractions ({len(divergent)}):")
        for d in divergent[:5]:
            print(f"  [{d['idx']}] GT={d['gt']}  first={d['first_cwe']}  last={d['last_cwe']}")
            print(f"      response: {d['response'][:150]}...")

        return output


@app.local_entrypoint()
def main():
    bench = RCMDiag()
    result = bench.diagnose.remote()
    print(json.dumps({k: v for k, v in result.items() if k != "per_item"}, indent=2))
