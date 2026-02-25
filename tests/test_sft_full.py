"""
test_4_sft_training.py — Test the full SFT training pipeline.
Downloads model weights (~15GB), needs GPU for actual training.
Uses --max-samples 3 and 1 epoch for a quick smoke test.

Run from project root:
  uv run notebooks/test_4_sft_training.py
"""


import os
import sys
import shutil


# Ensure we run from project root for relative paths
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import torch
from torch.utils.data import random_split
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from asa.data import Qwen2AudioCollator, SFTDataset

MODEL_ID = "Qwen/Qwen2-Audio-7B"
MAX_SAMPLES = 6
BATCH_SIZE = 1
OUTPUT_DIR = "results/sft_test"

print("=== Test 4: Full SFT Training Pipeline ===\n")

# ── 1. Processor ─────────────────────────────────────────────────────────
print(f"Step 1: Loading processor from {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("  ✅ Processor loaded\n")

# ── 2. Dataset + Collator ────────────────────────────────────────────────
print("Step 2: Creating dataset...")
full_dataset = SFTDataset(
    json_path="data/processed/train_nisqa_llama_10k.json",
    data_root="data",
    max_samples=MAX_SAMPLES,
)

# Split into train/val
val_size = max(1, int(len(full_dataset) * 0.33))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

collator = Qwen2AudioCollator(processor)

# Quick collator test
print("\n  Testing collator on 2 samples...")
batch = collator([full_dataset[0], full_dataset[1]])
for k, v in batch.items():
    print(f"    {k:25s} shape={str(tuple(v.shape)):20s} dtype={v.dtype}")
print("  ✅ Collator works\n")

# ── 3. Model ─────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("❌ No GPU available — this test requires a GPU.")
    sys.exit(1)


print(f"Step 3: Loading model {MODEL_ID}...")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32 if not torch.cuda.is_available() else torch.float16
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=dtype,
)
print(f"  ✅ Model loaded (dtype={dtype})\n")

# ── 4. Training ──────────────────────────────────────────────────────────
print("Step 4: Setting up Trainer...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    num_train_epochs=1,
    max_steps=2,  # Only 2 steps for smoke test
    bf16=(dtype == torch.bfloat16),
    fp16=(dtype == torch.float16),
    logging_steps=1,
    save_strategy="no",
    eval_strategy="steps",
    eval_steps=2,
    optim="adamw_torch",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    deepspeed=None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    processing_class=processor,
)

print("  Starting training (2 steps)...")
trainer.train()

print(f"\n✅ Full SFT pipeline works! (2 training steps completed)")
print(f"  To run real training: python -m asa.supervised_finetune --max-samples 100")

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
print("  Cleaned up test output directory.")