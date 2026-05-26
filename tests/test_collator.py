"""
test_3_collator.py — Test the Qwen2AudioCollator.
Downloads the processor (~10MB tokenizer, no GPU needed) and tests batching.
"""

from transformers import AutoProcessor
from asa.data import SFTDataset, Qwen2AudioCollator

MODEL_ID = "Qwen/Qwen2-Audio-7B"

print("=== Test 3: Qwen2AudioCollator ===\n")

# Load processor (downloads tokenizer, no model weights)
print(f"Loading processor from {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID, fix_mistral_regex=True)

# Load 2 samples
ds = SFTDataset(
    json_path="data/processed/sft/train_nisqa_llama_10k.json",
    data_root="data",
    max_samples=3,
)

# Collate into a batch
collator = Qwen2AudioCollator(processor)
batch = collator([ds[0], ds[1], ds[2]])

print("\nBatch contents:")
for key, tensor in batch.items():
    print(f"  {key:25s} shape={str(tuple(tensor.shape)):20s} dtype={tensor.dtype}")

# Verify label masking
labels = batch["labels"][0]
num_masked = (labels == -100).sum().item()
num_total = labels.shape[0]
print("\nLabel masking (sample 0):")
print(f"  Total tokens:  {num_total}")
print(f"  Masked (-100): {num_masked}  (prompt + padding)")
print(f"  Active:        {num_total - num_masked}  (response tokens)")
print(f"  First 20 labels: {labels[:20].tolist()}")

print("\n✅ Collator works!")
