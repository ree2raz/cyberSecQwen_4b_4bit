#!/usr/bin/env python3
"""Quantize CyberSecQwen-4B to 4-bit NF4 using bitsandbytes, push to Hub."""

import os
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "lablab-ai-amd-developer-hackathon/CyberSecQwen-4B"
OUTPUT_REPO_ID = "ree2raz/CyberSecQwen-4B-4bit"  # ← change to your HF username/org

print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Building 4-bit NF4 quantization config...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)

print(f"Loading model with 4-bit quantization config...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True,
)

print(f"Pushing quantized model to HuggingFace Hub: {OUTPUT_REPO_ID}...")
model.push_to_hub(OUTPUT_REPO_ID, safe_serialization=True)
tokenizer.push_to_hub(OUTPUT_REPO_ID)

print(f"\nQuantization complete! Pushed to:")
print(f"  https://huggingface.co/{OUTPUT_REPO_ID}")
print(f"Original: ~8GB | Expected: ~2GB at 4-bit NF4")