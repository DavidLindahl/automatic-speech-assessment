"""Integration test for the ALLD DPO data pipeline."""

from pathlib import Path
import json
import tempfile

import numpy as np
import pytest
import soundfile as sf
from transformers import AutoProcessor, AutoTokenizer

from asa.data import ALLDDPOCollator, DPODataset

pytestmark = pytest.mark.slow


def create_mock_dpo_jsonl(file_path: Path, audio_path: Path) -> None:
    """Create a dummy DPO dataset with metadata and local audio paths."""
    mock_data = [
        {
            "audios": [str(audio_path)],
            "mos": 4.5,
            "noi": 5.0,
            "col": 4.5,
            "loud": 4.8,
            "chosen": "This speech is highly intelligible and perfectly loud.",
            "rejected": "The speech is okay I guess.",
        },
        {
            "audios": [str(audio_path)],
            "mos": 2.1,
            "noi": 3.0,
            "col": 2.5,
            "loud": 4.0,
            "chosen": "The volume is clear, but there is significant discontinuity.",
            "rejected": "This is a perfectly clean speech without any issues.",
        },
    ]

    with file_path.open("w", encoding="utf-8") as handle:
        for item in mock_data:
            handle.write(json.dumps(item) + "\n")


def test_alld_pipeline() -> None:
    """Validate dataset formatting and dual-stream collator output shapes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        json_path = tmp_path / "train_dpo_mock.json"
        wav_path = tmp_path / "demo.wav"

        # 1 second silent waveform at 16 kHz.
        sf.write(wav_path, np.zeros(16000, dtype=np.float32), 16000)
        create_mock_dpo_jsonl(json_path, wav_path)

        dataset = DPODataset(json_path=json_path, data_root=tmp_path)
        sample = dataset[0]

        expected_keys = {
            "audio_prompt",
            "meta_prompt",
            "chosen",
            "rejected",
            "audio",
            "sampling_rate",
        }
        assert expected_keys.issubset(sample.keys())
        assert "(4) dis:" not in sample["meta_prompt"].lower()

        audio_model_dir = Path("models/sft_warmup")
        if not audio_model_dir.exists():
            pytest.skip(
                f"Missing local checkpoint required by this integration test: {audio_model_dir}"
            )

        audio_processor = AutoProcessor.from_pretrained(
            str(audio_model_dir),
            local_files_only=True,
            fix_mistral_regex=True,
        )

        try:
            text_tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2-7B-Instruct",
                local_files_only=True,
                fix_mistral_regex=True,
            )
        except OSError:
            pytest.skip("Missing cached tokenizer for Qwen/Qwen2-7B-Instruct.")

        collator = ALLDDPOCollator(
            audio_processor=audio_processor,
            text_tokenizer=text_tokenizer,
        )
        batch = collator([dataset[0], dataset[1]])

        expected_2n_batch_size = 4
        assert batch["policy_input_ids"].shape[0] == expected_2n_batch_size
        assert batch["ref_input_ids"].shape[0] == expected_2n_batch_size
        assert batch["policy_labels"][0][0].item() == -100
        assert batch["ref_labels"][0][0].item() == -100

        # Right-padding must be forced so _build_labels masks the prompt as an
        # exact prefix. Left-padding here re-introduces the DPO collapse.
        assert audio_processor.tokenizer.padding_side == "right"
        assert text_tokenizer.padding_side == "right"

        # The supervised reference labels must decode to exactly the response
        # (chosen / rejected) text, with NO prompt tokens leaking through. The
        # 2N layout is [chosen_0, chosen_1, rejected_0, rejected_1].
        responses = [
            dataset[0]["chosen"],
            dataset[1]["chosen"],
            dataset[0]["rejected"],
            dataset[1]["rejected"],
        ]
        for row, expected_response in enumerate(responses):
            label_row = batch["ref_labels"][row]
            supervised_ids = batch["ref_input_ids"][row][label_row != -100]
            supervised_text = text_tokenizer.decode(
                supervised_ids, skip_special_tokens=True
            ).strip()
            # Decoded supervised span must be the response, not prompt content.
            assert expected_response.strip().startswith(
                supervised_text[:20]
            ) or supervised_text.startswith(expected_response.strip()[:20]), (
                f"row {row}: supervised text {supervised_text!r} does not match "
                f"response {expected_response!r} — prompt is leaking into labels"
            )
            # No prompt marker may appear in the supervised span.
            assert "meta information" not in supervised_text.lower()
            assert "--- Current Task ---" not in supervised_text
