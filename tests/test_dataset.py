"""
test_2_dataset.py — Test the SFTDataset class.
Verifies JSONL parsing, audio path resolution, and sample structure.
"""

from asa.datasets import SFTDataset

print("=== Test 2: SFTDataset ===\n")

ds = SFTDataset(
    json_path="data/processed/train_nisqa_llama_10k.json",
    data_root="data",
    max_samples=5,
)

print(f"Dataset size: {len(ds)}\n")

for i in range(len(ds)):
    sample = ds[i]
    print(f"── Sample {i} ──")
    print(f"  Prompt:    {sample['prompt']}")
    print(f"  Response:  {sample['response'][:80]}...")
    print(f"  Audio:     shape={sample['audio'].shape}, SR={sample['sampling_rate']}")
    print(f"  Duration:  {len(sample['audio']) / sample['sampling_rate']:.1f}s")
    print()

print("✅ SFTDataset works!")
