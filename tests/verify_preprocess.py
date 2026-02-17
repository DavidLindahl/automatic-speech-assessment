from pathlib import Path
import sys

# Add src to path
sys.path.append("src")
from asa.data import MyDataset


def test_preprocess():
    print("Running manual verification of preprocess...")

    # Locate a CSV to test with
    data_path = Path("data/raw/NISQA_Corpus")
    if not data_path.exists():
        print(f"Data path {data_path} not found. Skipping full dataset test.")
        # Create a mock CSV for testing logic if real data missing
        return

    # Initialize dataset
    try:
        ds = MyDataset(data_path)
    except FileNotFoundError as e:
        print(f"Dataset init failed: {e}")
        return

    print(f"Dataset length: {len(ds)}")

    # Test text generation logic directly if possible, or just run one sample
    # Access private method for testing logic
    print("\nTesting text generation logic:")
    text = ds._generate_response(mos=4.2, noi=4.5, col=4.5, dis=4.5, loud=4.2)
    print(f"Input: High scores -> Output: {text}")
    assert "clean" in text and "loud" in text

    text = ds._generate_response(mos=1.2, noi=1.2, col=1.2, dis=1.2, loud=1.2)
    print(f"Input: Low scores -> Output: {text}")
    assert "poor" in text or "noisy" in text

    # Test __getitem__ on first sample
    print("\nTesting __getitem__ on index 0:")
    try:
        sample = ds[0]
        print("Keys:", sample.keys())
        print("MOS:", sample["mos"])
        print("Filename:", sample["filename"])
    except Exception as e:
        print(f"__getitem__ failed: {e}")


if __name__ == "__main__":
    test_preprocess()
