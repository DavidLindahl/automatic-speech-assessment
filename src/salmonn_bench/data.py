"""Dataset loading and path resolution for SALMONN benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load records from JSONL/object-stream/JSON-array files.

    Args:
        path: Dataset path.

    Returns:
        List of dictionary records.
    """
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []

    if stripped[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise TypeError(f"Expected list payload in {path}")
        return [ensure_record(item, path) for item in payload]

    records: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        while index < len(text) and text[index] in " \t\n\r":
            index += 1
        if index >= len(text):
            break
        obj, end_index = decoder.raw_decode(text, index)
        records.append(ensure_record(obj, path))
        index = end_index
    return records


def ensure_record(item: Any, path: Path) -> dict[str, Any]:
    """Validate a parsed record payload item."""
    if not isinstance(item, dict):
        raise TypeError(f"Expected object records in {path}, got {type(item).__name__}")
    return item


def resolve_audio_path(raw_path: str, data_root: Path) -> Path:
    """Resolve audio path across local and HPC-style path formats.

    Args:
        raw_path: Path from dataset record.
        data_root: Root that contains local raw datasets.

    Returns:
        Existing or translated local path candidate.
    """
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("/workspace/data/nisqa/"):
        return data_root / "raw" / normalized.removeprefix("/workspace/data/nisqa/")

    if "NISQA_Corpus/" in normalized:
        rel = normalized.split("NISQA_Corpus/", maxsplit=1)[1]
        return data_root / "raw" / "NISQA_Corpus" / rel

    if normalized.startswith("/data/raw/"):
        return data_root / normalized.removeprefix("/data/")

    return data_root / normalized.lstrip("/")
