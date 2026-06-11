"""Unit tests for DPODataset._build_meta_prompt (MOS vs temporal branch).

These exercise only the prompt-selection logic, so they construct a DPODataset
without loading audio: __init__ reads the JSONL, and _build_meta_prompt is pure
given a record + the joined dims index.
"""

import json
from pathlib import Path

import pytest

from asa.datasets import DPODataset


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")


def _dataset(tmp_path: Path, records: list[dict], dims: list[dict] | None):
    data_path = tmp_path / "dpo.jsonl"
    _write_jsonl(data_path, records)
    dims_path = None
    if dims is not None:
        dims_path = tmp_path / "dims.jsonl"
        _write_jsonl(dims_path, dims)
    return DPODataset(
        json_path=data_path, data_root=tmp_path, dims_source_json=dims_path
    )


def test_mos_record_uses_mos_expert(tmp_path):
    """A record with the 4 MOS dims and no segments takes the MOS path."""
    rec = {
        "audios": ["a/deg1.wav"],
        "chosen": "good caption.",
        "rejected": "bad caption.",
        "mos": 2.1,
        "noi": 3.0,
        "col": 2.5,
        "loud": 4.0,
    }
    ds = _dataset(tmp_path, [rec], dims=None)
    prompt = ds._build_meta_prompt(rec)
    assert "{mos: 2.1, noi: 3.0, col: 2.5, loud: 4.0}" in prompt
    assert "start:" not in prompt  # MOS prompt has no interval


def test_temporal_record_uses_temporal_expert_with_joined_dims(tmp_path):
    """A temporal record (has mix_deg_segments, lacks noi/col/loud) joins dims."""
    rec = {
        "audios": ["processed/temporal/mix_deg1.wav"],
        "filename_deg": "deg1.wav",
        "chosen": "The degradation in the clip is between <|3.57|> and <|4.72|> and is noisy.",
        "rejected": "The degradation in the clip is between <|1.10|> and <|2.00|> and is noisy.",
        "mos": 1.4,
        "mix_deg_segments": [{"start": 3.572, "end": 4.718}],
    }
    dims = [{"audios": ["x/deg1.wav"], "mos": 1.4, "noi": 1.4, "col": 2.6, "loud": 3.0}]
    ds = _dataset(tmp_path, [rec], dims=dims)
    prompt = ds._build_meta_prompt(rec)
    # full palette + rounded interval, temporal task framing
    assert "{mos: 1.4, noi: 1.4, col: 2.6, loud: 3.0, start: 3.57, end: 4.72}" in prompt
    assert "The degradation in the clip is between START and END" in prompt


def test_temporal_record_without_dims_raises(tmp_path):
    """A temporal record with no dims source must fail loudly, not silently."""
    rec = {
        "audios": ["processed/temporal/mix_deg1.wav"],
        "filename_deg": "deg1.wav",
        "chosen": "c",
        "rejected": "r",
        "mos": 1.4,
        "mix_deg_segments": [{"start": 1.0, "end": 2.0}],
    }
    ds = _dataset(tmp_path, [rec], dims=None)
    with pytest.raises(KeyError):
        ds._build_meta_prompt(rec)
