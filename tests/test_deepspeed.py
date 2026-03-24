"""
test_5_deepspeed.py — Test DeepSpeed ZeRO-2 training with 2 GPUs.
Runs 2 training steps to verify the full distributed pipeline.

Launch via job script:
  bsub < jobs/test_deepspeed.sh
"""

import os
import sys
import shutil

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist
from torch.utils.data import random_split
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from asa.datasets import SFTDataset
from asa.collators import Qwen2AudioCollator

MODEL_ID = "Qwen/Qwen2-Audio-7B"
MAX_SAMPLES = 6
BATCH_SIZE = 1
OUTPUT_DIR = "results/deepspeed_test"

# Only print from main process in distributed training
local_rank = int(os.environ.get("LOCAL_RANK", 0))
is_main = local_rank == 0

if is_main:
    print("=== Test 5: DeepSpeed ZeRO-2 Training ===\n")

if not torch.cuda.is_available():
    print("❌ No GPU available.")
    import pytest
    pytest.skip("No GPU available", allow_module_level=True)

if is_main:
    print(f"  GPUs available: {torch.cuda.device_count()}")
    print(f"  Local rank: {local_rank}")

# ── 1. Processor ─────────────────────────────────────────────────────────
if is_main:
    print(f"\nStep 1: Loading processor from {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
if is_main:
    print("  ✅ Processor loaded")

# ── 2. Dataset ───────────────────────────────────────────────────────────
if is_main:
    print("\nStep 2: Creating dataset...")
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
    print("  ✅ Dataset ready")

# ── 3. Model ─────────────────────────────────────────────────────────────
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
if is_main:
    print(f"\nStep 3: Loading model {MODEL_ID} (dtype={dtype})...")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=dtype,
)
model.config.use_cache = False
if is_main:
    print("  ✅ Model loaded")

# ── 4. Training with DeepSpeed ───────────────────────────────────────────
if is_main:
    print("\nStep 4: Running 2 training steps with DeepSpeed ZeRO-2...")
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
    eval_strategy="steps",
    eval_steps=2,
    optim="adamw_torch",
    gradient_checkpointing=True,
    deepspeed="configs/ds_zero2.json",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    processing_class=processor,
)

trainer.train()

if is_main:
    print("\n✅ DeepSpeed ZeRO-2 training works! (2 steps completed)")
    print("  To run real training: bsub < jobs/sft_hpc.sh")

    # Clean up only from main process
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    print("  Cleaned up test output directory.")

# Clean up distributed process group
if dist.is_initialized():
    dist.destroy_process_group()
