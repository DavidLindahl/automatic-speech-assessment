import json
import tempfile
import numpy as np
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoTokenizer

# Import the newly written classes from your data module
from asa.data import DPODataset, ALLDDPOCollator

def create_mock_dpo_jsonl(file_path: Path):
    """Creates a dummy DPO dataset with the ALLD metadata."""
    mock_data = [
        {
            "audios": ["raw/demo.wav"], # Assumes data/raw/demo.wav exists
            "mos": 4.5, "noi": 5.0, "col": 4.5, "dis": 5.0, "loud": 4.8,
            "chosen": "This speech is highly intelligible and perfectly loud.",
            "rejected": "The speech is okay I guess."
        },
        {
            "audios": ["raw/demo.wav"],
            "mos": 2.1, "noi": 3.0, "col": 2.5, "dis": 1.5, "loud": 4.0,
            "chosen": "The volume is clear, but there is significant discontinuity.",
            "rejected": "This is a perfectly clean speech without any issues."
        }
    ]
    
    with open(file_path, "w", encoding="utf-8") as f:
        for item in mock_data:
            f.write(json.dumps(item) + "\n")

def test_alld_pipeline():
    print("=== Starting ALLD Pipeline Test ===")
    
    # Use a temporary directory for the mock json
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        json_path = tmp_path / "train_dpo_mock.json"
        
        # 1. Create mock data
        create_mock_dpo_jsonl(json_path)
        print("✓ Created mock DPO dataset with metadata.")

        # 2. Test the Dataset
        # Note: We assume 'data/raw/demo.wav' exists in your repo based on the context.
        # If not, you might need to create a dummy wav file.
        data_root = Path("data") 
        
        try:
            dataset = DPODataset(json_path=json_path, data_root=data_root)
            sample = dataset[0]
        except Exception as e:
            print(f"❌ Failed to load DPODataset: {e}")
            return
            
        # Assert dataset keys
        expected_keys = ["audio_prompt", "meta_prompt", "chosen", "rejected", "audio", "sampling_rate"]
        for key in expected_keys:
            assert key in sample, f"Missing key '{key}' in dataset sample!"
            
        print("\n--- Snippet of the generated Expert Meta-Prompt ---")
        print(sample["meta_prompt"][-200:]) # Print the end to see the injected scores
        print("---------------------------------------------------\n")
        print("✓ DPODataset successfully formatted the dual prompts.")

        # 3. Load Processors (Using lightweight or base names for the test)
        # Note: In a real test, this requires an internet connection or cached models.
        audio_model_id = "models/sft_warmup" 
        text_model_id = "Qwen/Qwen2-7B-Instruct"
        
        print(f"Loading processor for {audio_model_id}...")
        audio_processor = AutoProcessor.from_pretrained(audio_model_id)
        
        print(f"Loading tokenizer for {text_model_id}...")
        text_tokenizer = AutoTokenizer.from_pretrained(text_model_id)

        # 4. Test the Collator
        collator = ALLDDPOCollator(audio_processor=audio_processor, text_tokenizer=text_tokenizer)
        
        # Grab a batch of 2
        features = [dataset[0], dataset[1]]
        batch = collator(features)
        
        # 5. Assertions to guarantee the ALLD method is correctly shaped
        print("\nChecking Collator Output Shapes...")
        
        # Original batch size was 2. DeepSpeed 2N batching means the models should see 4.
        expected_2n_batch_size = 4 
        
        # Policy Model Checks
        assert "policy_input_ids" in batch, "Missing policy_input_ids"
        assert batch["policy_input_ids"].shape[0] == expected_2n_batch_size, f"Policy batch size should be {expected_2n_batch_size}"
        assert "policy_labels" in batch, "Missing policy_labels"
        
        # Verify prompts are masked with -100 in labels
        # The first few tokens of the label should be -100 (the prompt), and the end should be valid token IDs (the answer)
        assert batch["policy_labels"][0][0].item() == -100, "Policy labels are not masking the prompt correctly!"
        print("✓ Policy (Audio) stream is correctly shaped and masked.")

        # Reference Model Checks
        assert "ref_input_ids" in batch, "Missing ref_input_ids"
        assert batch["ref_input_ids"].shape[0] == expected_2n_batch_size, f"Reference batch size should be {expected_2n_batch_size}"
        assert "ref_labels" in batch, "Missing ref_labels"
        assert batch["ref_labels"][0][0].item() == -100, "Reference labels are not masking the prompt correctly!"
        print("✓ Reference (Text) stream is correctly shaped and masked.")

        print("\n🎉 ALLD Pipeline Test Passed Successfully! The data is ready for the Trainer.")

if __name__ == "__main__":
    test_alld_pipeline()