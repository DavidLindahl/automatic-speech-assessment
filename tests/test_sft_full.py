"""
test_4_sft_training.py — Test the full SFT training pipeline.
Downloads model weights (~15GB), needs GPU for actual training.
Uses --max-samples 6 and 2 steps for a quick smoke test.

Run from project root:
  uv run notebooks/test_4_sft_training.py
"""

import os
import sys
import shutil

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

local_rank = int(os.environ.get("LOCAL_RANK", 0))
is_main = local_rank == 0

if is_main:
    print("=== Test 4: Full SFT Training Pipeline ===\n")

# ── 1. Processor ─────────────────────────────────────────────────────────
if is_main:
    print(f"Step 1: Loading processor from {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
if is_main:
    print("  ✅ Processor loaded\n")

# ── 2. Dataset + Collator ────────────────────────────────────────────────
if is_main:
    print("Step 2: Creating dataset...")
full_dataset = SFTDataset(
    json_path="data/processed/train_nisqa_llama_10k.json",
    data_root="data",
    max_samples=MAX_SAMPLES,
)

val_size = max(1, int(len(full_dataset) * 0.33))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
if is_main:
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

collator = Qwen2AudioCollator(processor)

if is_main:
    print("\n  Testing collator on 2 samples...")
batch = collator([full_dataset[0], full_dataset[1]])
if is_main:
    for k, v in batch.items():
        print(f"    {k:25s} shape={str(tuple(v.shape)):20s} dtype={v.dtype}")
    print("  ✅ Collator works\n")

# ── 3. Model ─────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    if is_main:
        print("❌ No GPU available — this test requires a GPU.")
    sys.exit(1)

if is_main:
    print(f"Step 3: Loading model {MODEL_ID}...")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=dtype,
)
model.config.use_cache = False
if is_main:
    print(f"  ✅ Model loaded (dtype={dtype})\n")

# ── 4. Training ──────────────────────────────────────────────────────────
if is_main:
    print("Step 4: Setting up Trainer...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    num_train_epochs=1,
    max_steps=2,
    bf16=(dtype == torch.bfloat16),
    fp16=(dtype == torch.float16),
    logging_steps=1,
    save_strategy="no",
    save_total_limit=2,
    save_only_model=True,
    eval_strategy="steps",
    eval_steps=2,
    optim="adamw_torch",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    deepspeed=None,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    processing_class=processor,
)

if is_main:
    print("  Starting training (2 steps)...")
trainer.train()

if is_main:
    print(f"\n✅ Full SFT pipeline works! (2 training steps completed)")
    print(f"  To run real training: torchrun --nproc_per_node=2 src/asa/supervised-finetune.py --bf16 --deepspeed configs/ds_zero2.json")

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
if is_main:
    print("  Cleaned up test output directory.")