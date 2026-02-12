from pathlib import Path
import torch
from asa.data import MyDataset


def test_dataset():
    # Use the corpus root because CSV paths are relative to it

    data_path = Path("data") / "raw" / "NISQA_Corpus"
    print(f"Initializing MyDataset with {data_path}...")
    try:
        dataset = MyDataset(data_path)
    except FileNotFoundError as e:
        print(f"Failed to initialize dataset: {e}")
        return

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty!")
        return

    print("Getting sample 0...")
    try:
        sample = dataset[0]
        print("Sample 0 keys:", sample.keys())
        print("Audio shape:", sample["audio"].shape)
        print("Filename:", sample["filename"])
        print("Sample rate:", sample["sample_rate"])

        # Verify types
        assert isinstance(sample["audio"], torch.Tensor)

        print("\nVerification SUCCESS!")

    except Exception as e:
        print(f"Failed to get sample: {e}")


if __name__ == "__main__":
    test_dataset()
