#!/usr/bin/env python3
"""
CTI-Bench evaluation for CyberSecQwen-4B-4bit on Modal L4.

Evaluates ree2raz/CyberSecQwen-4B-4bit on:
  - CTI-MCQ (2,500 items): CTI knowledge multiple-choice
  - CTI-RCM (1,000 items): CVE → CWE root-cause mapping

Protocol: matches Foundation-Sec-8B (arXiv:2504.21039 §B.3-B.4)
  - Zero-shot, no system prompt
  - Dataset's Prompt column as user message
  - Chat template via tokenizer
  - Temperature 0.3, max_tokens 512
  - 5 independent trials per task
  - Metric: strict accuracy

Reference FP16 scores (from CyberSecQwen-4B model card):
  - CTI-MCQ: 0.5868 ± 0.0029
  - CTI-RCM: 0.6664 ± 0.0023
"""
import gc
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import modal
import torch

app = modal.App("cybersecqwen-eval")


DATASET_ID = "AI4Sec/cti-bench"
MODEL_ID = "ree2raz/CyberSecQwen-4B-4bit"

GPU_TYPE = "L4"
NUM_TRIALS = 5
MAX_TOKENS = 512
TEMPERATURE = 0.3
BATCH_SIZE = 16

EVAL_VOLUME_NAME = "cybersecqwen-eval-results"

REF_SCORES = {
    "cti-mcq": {"accuracy": 0.5868, "std": 0.0029},
    "cti-rcm": {"accuracy": 0.6664, "std": 0.0023},
}

hf_cache = modal.Volume.from_name("inference-bench-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name(EVAL_VOLUME_NAME, create_if_missing=True)


@dataclass
class EvalResult:
    task: str
    accuracy: float
    correct: int
    total: int
    std: float
    parseable_rate: float
    parseable: int
    trial_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "accuracy": round(self.accuracy, 4),
            "correct": self.correct,
            "total": self.total,
            "std": round(self.std, 4),
            "parseable_rate": round(self.parseable_rate, 4),
            "parseable": self.parseable,
            "trial_results": self.trial_results,
        }


def make_image():
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "bitsandbytes>=0.41.0",
            "accelerate>=0.28.0",
            "datasets>=2.18.0",
            "scipy>=1.11.0",
        )
        .pip_install("huggingface_hub>=0.20.0")
        .env({
            "HF_HOME": "/hf_cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        })
    )


def extract_mcq_answer(response: str) -> str:
    """Extract A/B/C/D from model response for CTI-MCQ."""
    text = response.strip().upper()
    patterns = [
        r"^(A|B|C|D)$",
        r"(?:^|\n)(A|B|C|D)(?:\s|$|\n|\.)",
        r"(?:answer|choice|option)[:\s]+([A-D])\b",
        r"\b([A-D])\b(?=.*(?:correct|right|chosen|selected))",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            return m.group(1)
    return ""


def extract_cwe_answer(response: str) -> str:
    """Extract CWE-XXX from model response for CTI-RCM."""
    matches = re.findall(r"CWE-\d+", response, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return ""


def build_messages(prompt_text: str) -> list[dict]:
    """Build a chat message list from the dataset's Prompt column.
    
    Per the Foundation-Sec-8B protocol: the dataset's Prompt column
    is the full user message, no system prompt.
    """
    return [{"role": "user", "content": prompt_text}]


def format_prompt(tokenizer, messages: list[dict]) -> str:
    """Apply Qwen chat template to messages, returning the full prompt string."""
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


@app.cls(
    image=make_image(),
    gpu=GPU_TYPE,
    timeout=60 * 60 * 4,
    volumes={
        "/results": results_vol,
        "/hf_cache": hf_cache,
    },
)
class CTIEval:
    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[start] Loading {MODEL_ID} ...")
        t0 = time.perf_counter()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"[start] Model loaded in {time.perf_counter() - t0:.1f}s")

    @modal.exit()
    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    def _load_dataset(self, task: str) -> list[dict]:
        from datasets import load_dataset

        ds = load_dataset(DATASET_ID, task, split="test")
        return [dict(item) for item in ds]

    def _run_trial(self, task: str, items: list[dict], seed: int) -> dict:
        """Run one trial (all items, one seed). Returns per-item results."""
        import numpy as np

        rng = random.Random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        prompts = []
        gt_list = []
        parseable = 0

        for item in items:
            prompt_text = item["Prompt"]
            messages = build_messages(prompt_text)
            full_prompt = format_prompt(self.tokenizer, messages)
            prompts.append(full_prompt)
            gt_list.append(item["GT"].strip().upper())

        results = []
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i : i + BATCH_SIZE]
            batch_gt = gt_list[i : i + BATCH_SIZE]

            enc = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3072,
            )
            input_ids = enc["input_ids"].to(self.model.device)
            attention_mask = enc["attention_mask"].to(self.model.device)
            pad_id = self.tokenizer.pad_token_id

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    pad_token_id=pad_id,
                    repetition_penalty=1.1,
                )

            for j, (out, gt) in enumerate(zip(outputs, batch_gt)):
                input_len = input_ids[j].numel()
                generated = out[input_len:]
                response = self.tokenizer.decode(generated, skip_special_tokens=True)

                if task == "cti-mcq":
                    pred = extract_mcq_answer(response)
                else:
                    pred = extract_cwe_answer(response)

                is_parseable = bool(pred)
                if is_parseable:
                    parseable += 1

                is_correct = pred == gt
                results.append({
                    "index": i + j,
                    "gt": gt,
                    "pred": pred,
                    "correct": is_correct,
                    "parseable": is_parseable,
                    "response_preview": response[:200],
                })

            del enc, input_ids, attention_mask, outputs
            torch.cuda.empty_cache()

        correct = sum(1 for r in results if r["correct"])
        return {
            "seed": seed,
            "correct": correct,
            "total": len(results),
            "accuracy": correct / len(results),
            "parseable": parseable,
            "results": results,
        }

    @modal.method()
    def run_all(self, tasks: list[str]) -> dict:
        """Run all tasks + trials in a single container. Model loaded once."""
        import numpy as np

        results: dict[str, Any] = {
            "model": MODEL_ID,
            "quantization": "4-bit NF4 (bitsandbytes)",
            "gpu": GPU_TYPE,
            "tasks": {},
        }

        for task in tasks:
            print(f"[eval] Task: {task}")
            items = self._load_dataset(task)
            print(f"[eval] Loaded {len(items)} items")

            all_results = []
            for trial_idx in range(NUM_TRIALS):
                seed = 42 + trial_idx
                print(f"[eval] Trial {trial_idx + 1}/{NUM_TRIALS} (seed={seed}) ...")
                t0 = time.perf_counter()
                trial = self._run_trial(task, items, seed)
                elapsed = time.perf_counter() - t0
                print(
                    f"[eval]   Trial {trial_idx + 1}: acc={trial['accuracy']:.4f} "
                    f"({trial['correct']}/{trial['total']}), parseable={trial['parseable']}, "
                    f"took={elapsed:.1f}s"
                )
                all_results.append(trial)

            accuracies = [t["accuracy"] for t in all_results]
            mean_acc = float(np.mean(accuracies))
            std_acc = float(np.std(accuracies))
            total_correct = sum(t["correct"] for t in all_results)
            total_items = sum(t["total"] for t in all_results)
            total_parseable = sum(t["parseable"] for t in all_results)

            ref = REF_SCORES.get(task, {})
            ref_acc = ref.get("accuracy")
            ref_std = ref.get("std")
            delta_vs_ref = mean_acc - ref_acc if ref_acc is not None else None

            print(f"[result] {task}: {mean_acc:.4f} ± {std_acc:.4f}")
            if ref_acc is not None:
                print(f"[result]   vs FP16 ref ({ref_acc:.4f} ± {ref_std:.4f}): Δ={delta_vs_ref:+.4f}")
            print(f"[result]   parseable rate: {total_parseable / total_items:.4f}")

            results["tasks"][task] = {
                "task": task,
                "accuracy": round(mean_acc, 4),
                "correct": total_correct,
                "total": total_items,
                "std": round(std_acc, 4),
                "parseable_rate": round(total_parseable / total_items, 4),
                "parseable": total_parseable,
                "trial_results": [
                    {"seed": t["seed"], "accuracy": t["accuracy"], "correct": t["correct"], "total": t["total"]}
                    for t in all_results
                ],
                "model": MODEL_ID,
                "protocol": {
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "num_trials": NUM_TRIALS,
                    "batch_size": BATCH_SIZE,
                    "prompt_source": "dataset_Prompt_column",
                    "system_prompt": "none",
                    "chat_template": "applied",
                },
                "reference_fp16": {"accuracy": ref_acc, "std": ref_std},
                "delta_vs_fp16": round(delta_vs_ref, 4) if delta_vs_ref is not None else None,
            }

        output_path = "/results/eval_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        results_vol.commit()
        print(f"[save] Results saved to {output_path} and committed.")

        return results


@app.local_entrypoint()
def main(task: str = "all"):
    """
    Run CTI-Bench evaluation for CyberSecQwen-4B-4bit.

    Usage:
      modal run modal_cti_eval.py --task all     # CTI-MCQ + CTI-RCM
      modal run modal_cti_eval.py --task cti-mcq
      modal run modal_cti_eval.py --task cti-rcm
    """
    tasks = ["cti-mcq", "cti-rcm"] if task == "all" else [task]

    print("=" * 60)
    print("  CyberSecQwen-4B-4bit — CTI-Bench Evaluation on Modal")
    print("=" * 60)
    print(f"  Model:      {MODEL_ID}")
    print(f"  GPU:        {GPU_TYPE}")
    print(f"  Tasks:      {tasks}")
    print(f"  Trials:     {NUM_TRIALS}")
    print(f"  Temp:       {TEMPERATURE}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("=" * 60)

    bench = CTIEval()
    results = bench.run_all.remote(tasks)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Task':<12} {'4-bit Acc':>12} {'FP16 Ref':>12} {'Δ':>10} {'Std':>10}")
    print("-" * 60)

    for t in tasks:
        tr = results["tasks"][t]
        ref = REF_SCORES.get(t, {})
        ref_acc = ref.get("accuracy")
        delta = tr.get("delta_vs_fp16")
        print(
            f"{t:<12} {tr['accuracy']:>11.4f} "
            f"{(ref_acc if ref_acc else 'N/A'):>12} "
            f"{(f'{delta:+.4f}' if delta is not None else 'N/A'):>10} "
            f"{tr['std']:>10.4f}"
        )
    print("=" * 60)
    print("\nResults persisted to Modal volume at /results/eval_results.json")
