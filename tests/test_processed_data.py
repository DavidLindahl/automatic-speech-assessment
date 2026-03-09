"""Tests for processed dataset loading and path resolution."""

from __future__ import annotations

import json
from pathlib import Path

from asa.preflight import run_preflight_checks
from asa.processed_data import load_processed_records, resolve_audio_path, write_processed_records


def test_load_processed_records_accepts_json_array(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text(json.dumps([{"id": 1}, {"id": 2}]), encoding="utf-8")

    assert load_processed_records(path) == [{"id": 1}, {"id": 2}]


def test_load_processed_records_accepts_object_stream(tmp_path: Path) -> None:
    path = tmp_path / "stream.json"
    path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")

    assert load_processed_records(path) == [{"id": 1}, {"id": 2}]


def test_write_processed_records_emits_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "written.json"
    write_processed_records(path, [{"id": 1}, {"id": 2}])

    assert path.read_text(encoding="utf-8") == '{"id": 1}\n{"id": 2}\n'


def test_resolve_audio_path_handles_hpc_and_repo_layouts(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    audio_path = data_root / "raw" / "NISQA_Corpus" / "NISQA_TRAIN_SIM" / "sample.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"wav")

    hpc_path = "/work3/s234817/NISQA_Corpus/NISQA_TRAIN_SIM/sample.wav"
    workspace_path = "/workspace/data/nisqa/NISQA_Corpus/NISQA_TRAIN_SIM/sample.wav"

    assert resolve_audio_path(hpc_path, data_root) == audio_path
    assert resolve_audio_path(workspace_path, data_root) == audio_path


def test_preflight_reports_missing_dpo_metadata(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    raw_dir = tmp_path / "data" / "raw" / "NISQA_Corpus" / "NISQA_TRAIN_SIM"
    processed_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    (raw_dir / "sample.wav").write_bytes(b"wav")
    write_processed_records(
        processed_dir / "train_dpo_10k.json",
        [
            {
                "audios": ["/work3/s234817/NISQA_Corpus/NISQA_TRAIN_SIM/sample.wav"],
                "mos": 4.0,
                "noi": 4.0,
                "col": 4.0,
                "chosen": "chosen",
                "rejected": "rejected",
            }
        ],
    )

    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        findings = run_preflight_checks("dpo", Path("data"), 1, [])
    finally:
        os.chdir(original_cwd)

    assert any("missing keys loud" in finding for finding in findings)
