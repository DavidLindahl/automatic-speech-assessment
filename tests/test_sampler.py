import pytest
import json
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
from asa.sampler import sample_data

@pytest.fixture
def mock_csv_data():
    """Mock the NISQA dataset CSVs with dummy data needed for A/B testing logic."""
    data = []
    # Create ~20 rows so pandas has enough to sample from
    for i in range(20):
        # Alternate MOS values to ensure gap > 0.5 exists
        mos = 4.5 if i % 2 == 0 else 2.0
        data.append({
            "filepath_deg": f"demo_{i}.wav",
            "filename_deg": f"demo_{i}.wav",
            "mos": mos,
            "noi": 3.0,
            "col": 4.0,
            "dis": 3.5,
            "loud": 4.5
        })
    return pd.DataFrame(data)

@patch("asa.sampler.load_csv")
def test_sample_data(mock_load_csv, mock_csv_data, tmp_path):
    """Test the sampler logic for allocating A/B and MOS paths correctly into train and test out bins."""
    # We want load_csv to return our small dataframe
    mock_load_csv.return_value = mock_csv_data

    # Call the sampler with very small target numbers so the loop finishes instantly
    with patch("asa.sampler.random.seed"):
        # We need to temporarily patch exactly the variables we want, but they're hardcoded in sample_data logic!
        # The script hardcodes n_ab_pairs = 500, n_mos_train = 500. This is bad for testing.
        # But we can patch the pandas .iloc length checks to just swallow the exception or modify the function.
        pass

    # Actually, we should test the utility functions or rewrite the test to supply enough data
    # Instead of patching lengths, let's provide a massive dataframe 
    large_df = pd.concat([mock_csv_data] * 100).reset_index(drop=True)
    mock_load_csv.return_value = large_df

    # Run the sampler
    sample_data(data_root=Path("fake_root"), output_dir=tmp_path, seed=42)

    ab_file = tmp_path / "ab_dataset.json"
    mos_file = tmp_path / "mos_dataset.json"

    assert ab_file.exists(), "A/B dataset JSON was not generated"
    assert mos_file.exists(), "MOS dataset JSON was not generated"

    with open(ab_file, 'r') as f:
        ab_data = json.load(f)
    with open(mos_file, 'r') as f:
        mos_data = json.load(f)

    assert len(ab_data) > 0, "No A/B pairs found"
    assert len(mos_data) > 0, "No MOS records found"

    # Verify formatting requirements
    assert "winner" in ab_data[0]
    assert "pair_id" in ab_data[0]
    assert "meta_a" in ab_data[0]
    assert "meta" in mos_data[0]
