#!/usr/bin/env python3
"""
CTIBench evaluation: compare FP16 (original) vs 4-bit NF4 (quantized)
on CyberSecQwen-4B across CTI-MCQ and CTI-RCM tasks.

Tasks:
  - cti-mcq : CTI knowledge MCQ → Accuracy
  - cti-rcm : CVE→CWE mapping → Accuracy
"""

import json
import re
import os
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ORIGINAL   = "lablab-ai-amd-developer-hackathon/CyberSecQwen-4B"
MODEL_QUANTIZED  = "ree2raz/CyberSecQwen-4B-4bit"
DATASET          = "AI4Sec/cti-bench"
TASKS            = ["cti-mcq", "cti-rcm"]
MAX_NEW_TOKENS   = 128
BATCH_SIZE       = 1  # single sample at a time to avoid OOM
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_fp16(model_id: str):
    print(f"[FP16] Loading {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_model_4bit(model_id: str):
    from transformers import BitsAndBytesConfig
    print(f"[4bit] Loading {model_id} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def extract_mcq_answer(response: str) -> str:
    """Extract A/B/C/D from model response."""
    response = response.strip().upper()
    # Look for a standalone letter in brackets or after "Answer:"
    m = re.search(r'\b([A-D])\b', response)
    if m:
        return m.group(1)
    return ""


def extract_cwe_answer(response: str) -> str:
    """Extract CWE-XXX from model response."""
    m = re.search(r'(CWE-\d+)', response.upper())
    if m:
        return m.group(1)
    return ""


def eval_task(model, tokenizer, task: str):
    ds = load_dataset(DATASET, task, split="test")
    total = len(ds)
    correct = 0

    print(f"  Evaluating {total} samples for {task}...")
    for i, item in enumerate(ds):
        prompt_text = item["Prompt"]
        gt = item["GT"].strip().upper()

        if task == "cti-mcq":
            question = item["Question"]
            opts = [f"A) {item['Option A']}", f"B) {item['Option B']}",
                    f"C) {item['Option C']}", f"D) {item['Option D']}"]
            full_prompt = f"{prompt_text}\nQuestion: {question}\n" + "\n".join(opts) + "\nAnswer:"

            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_mcq_answer(response)

        elif task == "cti-rcm":
            desc = item["Description"]
            prompt_full = f"{prompt_text}\n\nCVE Description: {desc}\n\nCWE Mapping:"
            inputs = tokenizer(prompt_full, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_cwe_answer(response)

        is_correct = (pred == gt)
        if is_correct:
            correct += 1

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{total}] current acc: {correct/(i+1):.4f}")

        # Clear cache every sample
        del inputs, out, response
        gc.collect()
        torch.cuda.empty_cache()

    acc = correct / total
    return {"accuracy": round(acc, 4), "correct": correct, "total": total}


def main():
    results = {}
    output_file = os.environ.get("EVAL_OUTPUT", "eval_results.json")

    # ── FP16 evaluation ──────────────────────────────────────────────────────
    model_fp16, tok_fp16 = load_model_fp16(MODEL_ORIGINAL)
    results["original_fp16"] = {}
    for task in TASKS:
        r = eval_task(model_fp16, tok_fp16, task)
        results["original_fp16"][task] = r
        print(f"  → {task}: {r['accuracy']*100:.2f}% ({r['correct']}/{r['total']})")
    del model_fp16, tok_fp16
    gc.collect(); torch.cuda.empty_cache()

    # ── 4-bit evaluation ─────────────────────────────────────────────────────
    model_4b, tok_4b = load_model_4bit(MODEL_QUANTIZED)
    results["quantized_4bit"] = {}
    for task in TASKS:
        r = eval_task(model_4b, tok_4b, task)
        results["quantized_4bit"][task] = r
        print(f"  → {task}: {r['accuracy']*100:.2f}% ({r['correct']}/{r['total']})")
    del model_4b, tok_4b

    # ── Delta summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Task':<12} {'FP16':>8} {'4-bit':>8} {'Δ':>8} {'Degradation':>12}")
    print("-"*60)
    for task in TASKS:
        fp16 = results["original_fp16"][task]["accuracy"]
        q4   = results["quantized_4bit"][task]["accuracy"]
        delta = q4 - fp16
        pct   = (delta / fp16) * 100 if fp16 > 0 else 0
        print(f"{task:<12} {fp16*100:>7.2f}% {q4*100:>7.2f}% {delta:>+7.2f}  {pct:>+10.2f}%")

    print("="*60)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()