"""
test_dataset_ab.py — Pytest script for SFTDatasetAB and Qwen2AudioCollatorAB.
Downloads the processor (~10MB tokenizer, no GPU needed) and tests batching.
"""

from transformers import AutoProcessor
from asa.datasets import SFTDatasetAB
from asa.collators import Qwen2AudioCollatorAB

MODEL_ID = "Qwen/Qwen2-Audio-7B"


def test_dataset_and_collator_ab():
    print(f"\nLoading processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # 1. Test Dataset Loading
    print("Loading dataset...")
    ds = SFTDatasetAB(
        json_path="data/processed/train_nisqa_abtest_llama_10k.json",
        data_root="data",
        max_samples=2,
    )

    assert len(ds) == 2, f"Expected 2 samples, got {len(ds)}"

    for i in range(len(ds)):
        sample = ds[i]
        assert "prompt" in sample, "Sample is missing 'prompt'"
        assert "response" in sample, "Sample is missing 'response'"
        assert "audio_a" in sample, "Sample is missing 'audio_a'"
        assert "audio_b" in sample, "Sample is missing 'audio_b'"
        assert "<|AUDIO|>" in sample["prompt"], "Prompt missing Qwen special tokens"

        print(f"── Sample {i} ──")
        print(f"  Audio A:   shape={sample['audio_a'].shape}")
        print(f"  Audio B:   shape={sample['audio_b'].shape}")

    # 2. Test Collation and Masking
    print("Collating into a batch...")
    collator = Qwen2AudioCollatorAB(processor)
    batch = collator([ds[0], ds[1]])

    # Verify expected keys and dimensions
    assert "input_ids" in batch, "Batch missing 'input_ids'"
    assert "attention_mask" in batch, "Batch missing 'attention_mask'"
    assert "input_features" in batch, "Batch missing 'input_features'"
    assert "labels" in batch, "Batch missing 'labels'"

    print("\nBatch contents:")
    for key, tensor in batch.items():
        print(f"  {key:25s} shape={str(tuple(tensor.shape)):20s} dtype={tensor.dtype}")

    batch_size = batch["input_ids"].shape[0]
    assert batch_size == 2, f"Expected batch size 2, got {batch_size}"

    # Specifically check the audio features shape. We passed 2 samples with 2 audios each = 4 total audios.
    # Qwen2-Audio processor features dim 0 should be total number of audios across the batch.
    num_audios_in_batch = batch["input_features"].shape[0]
    assert (
        num_audios_in_batch == 4
    ), f"Expected 4 audio features, got {num_audios_in_batch}"

    # Verify label masking
    labels = batch["labels"][0]
    num_masked = (labels == -100).sum().item()
    num_total = labels.shape[0]
    active_tokens = num_total - num_masked

    print("\nLabel masking (sample 0):")
    print(f"  Total tokens:  {num_total}")
    print(f"  Masked (-100): {num_masked}  (prompt + padding)")
    print(f"  Active:        {active_tokens}  (response tokens)")

    assert active_tokens > 0, "No active response tokens found in labels!"
    assert num_masked > 0, "No masked tokens found (prompt should be masked)!"
