"""Utilities for processed training and evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


DPO_METADATA_FIELDS = ("mos", "noi", "col", "loud")


def load_processed_records(path: str | Path) -> list[dict[str, Any]]:
    """Load processed records from JSON array, JSONL, or object-stream JSON.

    Args:
        path: Input dataset path.

    Returns:
        Parsed records as a list of dictionaries.

    Raises:
        TypeError: If the file content is not an object/list of objects.
    """
    dataset_path = Path(path)
    text = dataset_path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []

    if stripped[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise TypeError(f"Expected a JSON array in {dataset_path}")
        return [ensure_record(item, dataset_path) for item in payload]

    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index] in " \t\n\r":
            index += 1
        if index >= len(text):
            break
        obj, end_index = decoder.raw_decode(text, index)
        records.append(ensure_record(obj, dataset_path))
        index = end_index
    return records


def write_processed_records(
    path: str | Path, records: Iterable[dict[str, Any]]
) -> None:
    """Write processed records in canonical JSONL form.

    Args:
        path: Output dataset path.
        records: Records to write.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_audio_path(raw_path: str, data_root: str | Path) -> Path:
    """Resolve an audio path across local and HPC dataset layouts.

    Args:
        raw_path: Stored audio path from a processed record.
        data_root: Root directory that contains the repo-local `raw/` tree.

    Returns:
        Resolved local path.
    """
    root = Path(data_root)
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    normalized = raw_path.replace("\\", "/")
    if "NISQA_Corpus/" in normalized:
        relative = normalized.split("NISQA_Corpus/", maxsplit=1)[1]
        return root / "raw" / "NISQA_Corpus" / relative

    if normalized.startswith("/workspace/data/nisqa/"):
        return root / "raw" / normalized.removeprefix("/workspace/data/nisqa/")

    if normalized.startswith("/data/raw/"):
        return root / normalized.removeprefix("/data/")

    return root / normalized.lstrip("/")


def ensure_record(item: Any, path: Path) -> dict[str, Any]:
    """Validate that a parsed payload item is a JSON object."""
    if not isinstance(item, dict):
        raise TypeError(
            f"Expected JSON object records in {path}, got {type(item).__name__}"
        )
    return item
