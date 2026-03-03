"""
test_4_sft_training.py — Test the full SFT training pipeline.
Downloads model weights (~15GB), needs GPU for actual training.
Runs a forward + backward pass to validate the pipeline.

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
)

from asa.data import Qwen2AudioCollator, SFTDataset

MODEL_ID = "Qwen/Qwen2-Audio-7B"
MAX_SAMPLES = 6

print("=== Test 4: Full SFT Training Pipeline ===\n")

# ── 1. Processor ─────────────────────────────────────────────────────────
print(f"Step 1: Loading processor from {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID, fix_mistral_regex=True)
print("  ✅ Processor loaded\n")

# ── 2. Dataset + Collator ────────────────────────────────────────────────
print("Step 2: Creating dataset...")
full_dataset = SFTDataset(
    json_path="data/processed/train_nisqa_llama_10k.json",
    data_root="data",
    max_samples=MAX_SAMPLES,
)

val_size = max(1, int(len(full_dataset) * 0.33))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

collator = Qwen2AudioCollator(processor)

print("\n  Testing collator on 2 samples...")
batch = collator([full_dataset[0], full_dataset[1]])
for k, v in batch.items():
    print(f"    {k:25s} shape={str(tuple(v.shape)):20s} dtype={v.dtype}")
print("  ✅ Collator works\n")

# ── 3. Model ─────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("❌ No GPU available — this test requires a GPU.")
    sys.exit(1)

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Step 3: Loading model {MODEL_ID} (dtype={dtype})...")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)
model.to("cuda")
print("  ✅ Model loaded\n")

# ── 4. Forward + Backward pass ──────────────────────────────────────────
print("Step 4: Testing forward + backward pass...")
model.train()
model.gradient_checkpointing_enable()

batch = collator([train_dataset[0]])
batch = {k: v.to("cuda") for k, v in batch.items()}

with torch.amp.autocast("cuda", dtype=dtype):
    output = model(**batch)

print(f"  Loss: {output.loss.item():.4f}")

output.loss.backward()
print("  ✅ Forward + backward pass works!\n")

# ── 5. Label masking check ──────────────────────────────────────────────
print("Step 5: Verifying label masking...")
labels = batch["labels"][0]
total = labels.shape[0]
masked = (labels == -100).sum().item()
active = total - masked
print(f"  Total tokens:  {total}")
print(f"  Masked (-100): {masked}  (prompt + padding)")
print(f"  Active:        {active}  (response tokens)")
assert active > 0, "No active labels — loss would be meaningless!"
print("  ✅ Label masking correct\n")

print("✅ Full SFT pipeline validated!")
print("  To run real training with DeepSpeed:")
print("  bsub < jobs/sft_hpc.sh")