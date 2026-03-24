import os
import json
import pytest
from unittest.mock import patch, MagicMock
from asa.caption_generator import process_single_file

@patch("asa.caption_generator.call_gemini_api")
def test_process_single_file(mock_gemini, tmp_path):
    """Test generating descriptions mapping input metadata to string captions using mocked Google API."""
    input_file = tmp_path / "mos_dataset.json"
    output_file = tmp_path / "train_nisqa_llama_10k.json"

    # Minimal MOS dataset matching expected structure
    mock_input = [{
        "audio_path": "fake.wav",
        "split": "train",
        "utt_id": "test_item",
        "meta": {"mos": 4.0, "noi": 4.5, "col": 4.0, "loud": 4.5}
    }]
    with open(input_file, "w") as f:
        json.dump(mock_input, f)

    # Mock API to return some descriptive text
    mock_gemini.return_value = 'This audio is exceptionally clean.'

    # Process file
    process_single_file(str(input_file), str(output_file))

    # Assert output exists and has expected injected captions
    assert output_file.exists()
    
    with open(output_file, 'r') as f:
        results = [json.loads(line) for line in f]

    assert len(results) == 1
    assert "response" in results[0]
    assert results[0]["response"] == "This audio is exceptionally clean."
    assert "query" in results[0]
    assert results[0]["audios"][0] == "fake.wav"
    assert results[0]["mos"] == 4.0
