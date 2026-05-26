"""Build temporal SFT JSONL by reusing existing NISQA captions.

This script merges:
1. A temporal mix manifest (with mix filenames and degradation intervals)
2. Existing caption records from ``train_nisqa_llama_10k.json``

It outputs JSONL records ready for SFT, with timestamp supervision embedded in
the response text using ``<|start|>`` and ``<|end|>`` tokens.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import typer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from asa.processed_data import write_processed_records


DEGRADATION_LABELS: dict[str, str] = {
    "codec1": "codec artifacts",
    "codec2": "codec artifacts",
    "plcMode1": "packet-loss concealment artifacts",
    "plcMode2": "packet-loss concealment artifacts",
    "bgn": "background noise",
    "wbgn": "background noise",
    "p50mnru": "background noise",
    "filter": "band-limiting filter distortion",
    "arb_filter": "band-limiting filter distortion",
    "timeclipping": "time clipping artifacts",
    "clipping": "clipping distortion",
}

DEFAULT_QUERY = (
    "Please describe and evaluate the synthetic speech<audio>. "
    "Also localize the degraded region and report timestamps as <|start|> and <|end|>."
)

app = typer.Typer()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records.

    Args:
        path: Path to JSONL file.

    Returns:
        Parsed records.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def parse_json_field(value: object, fallback: Any) -> Any:
    """Parse a JSON-serialized value from CSV fields.

    Args:
        value: Raw field value.
        fallback: Value returned if parsing fails.

    Returns:
        Parsed Python object or fallback.
    """
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def normalize_caption(text: str) -> str:
    """Normalize a caption to a single clean sentence block."""
    collapsed = " ".join(text.strip().split())
    if not collapsed:
        return "The speech has mixed quality."
    return collapsed


def normalize_degradation_types(raw_types: list[str]) -> list[str]:
    """Normalize raw degradation tags into readable phrases.

    Args:
        raw_types: Raw degradation tags from the manifest.

    Returns:
        Deduplicated list of normalized phrases.
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_types:
        key = str(raw).strip()
        if not key:
            continue
        phrase = DEGRADATION_LABELS.get(key, key.replace("_", " "))
        if phrase not in seen:
            seen.add(phrase)
            normalized.append(phrase)
    return normalized


def format_degradation_phrase(types: list[str]) -> str:
    """Format human-readable degradation type phrase for response text."""
    if not types:
        return "localized degradation"
    if len(types) == 1:
        return types[0]
    if len(types) == 2:
        return f"{types[0]} and {types[1]}"
    return f"{', '.join(types[:-1])}, and {types[-1]}"


def pick_primary_segment(segments: list[dict[str, Any]]) -> tuple[float, float]:
    """Pick one segment for supervision.

    The current mix generator creates one segment, but this keeps behavior robust
    if multiple segments are present by choosing the longest one.
    """
    if not segments:
        return 0.0, 0.1

    valid: list[tuple[float, float]] = []
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        if end > start:
            valid.append((start, end))
    if not valid:
        return 0.0, 0.1

    start, end = max(valid, key=lambda item: item[1] - item[0])
    return start, end


def build_temporal_response(
    base_caption: str,
    start_time: float,
    end_time: float,
    degradation_phrase: str,
) -> str:
    """Compose target response that includes both quality text and localization."""
    prefix = normalize_caption(base_caption)
    if not prefix.endswith((".", "!", "?")):
        prefix = f"{prefix}."

    return (
        f"{prefix} The quality is interrupted by {degradation_phrase} "
        f"occurring between <|{start_time:.2f}|> and <|{end_time:.2f}|>."
    )


def build_caption_index(
    caption_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index base caption records by degraded filename basename."""
    index: dict[str, dict[str, Any]] = {}
    for record in caption_records:
        audios = record.get("audios", [])
        if not isinstance(audios, list) or not audios:
            continue
        deg_name = Path(str(audios[0])).name
        if deg_name:
            index[deg_name] = record
    return index


@app.command()
def main(
    manifest_path: Path = typer.Option(
        Path("data/processed/nisqa_sim_mix_lowmos_active_3000/manifest.csv"),
        help="Temporal mix manifest CSV.",
    ),
    caption_jsonl: Path = typer.Option(
        Path("data/processed/train_nisqa_llama_10k.json"),
        help="Base caption JSONL used for caption reuse.",
    ),
    mixes_dir: Path = typer.Option(
        Path("data/processed/nisqa_sim_mix_lowmos_active_3000"),
        help="Directory containing mixed WAV files from manifest.",
    ),
    output_jsonl: Path = typer.Option(
        Path("data/processed/train_nisqa_temporal_mix_3000.json"),
        help="Output JSONL path.",
    ),
    data_root: Path = typer.Option(
        Path("data"),
        help="Root directory used to build portable audio paths.",
    ),
    query: str = typer.Option(
        DEFAULT_QUERY,
        help="Query text stored in each record.",
    ),
    include_clean_path: bool = typer.Option(
        True,
        "--include-clean-path/--no-include-clean-path",
        help="Keep clean_path field from base caption record.",
    ),
) -> None:
    """Build temporal SFT JSONL for NISQA temporal mixes."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not caption_jsonl.exists():
        raise FileNotFoundError(f"Caption JSONL not found: {caption_jsonl}")
    if not mixes_dir.exists():
        raise FileNotFoundError(f"Mix directory not found: {mixes_dir}")

    manifest_df = pd.read_csv(manifest_path)
    data_root_resolved = data_root.resolve()
    caption_records = load_jsonl(caption_jsonl)
    caption_index = build_caption_index(caption_records)

    output_records: list[dict[str, Any]] = []
    missing_base = 0
    missing_mix = 0

    for _, row in manifest_df.sort_values("index").iterrows():
        filename_deg = str(row["filename_deg"])
        mix_filename = str(row["mix_filename"])
        mix_path = (mixes_dir / mix_filename).resolve()
        if not mix_path.exists():
            missing_mix += 1
            continue

        base_record = caption_index.get(filename_deg)
        if base_record is None:
            missing_base += 1
            continue

        try:
            audio_value = str(mix_path.relative_to(data_root_resolved))
        except ValueError:
            audio_value = str(mix_path)

        segments = parse_json_field(row.get("mix_deg_segments", "[]"), [])
        if not isinstance(segments, list):
            segments = []
        start_time, end_time = pick_primary_segment(segments)

        raw_types = parse_json_field(row.get("source_degradation_types", "[]"), [])
        if not isinstance(raw_types, list):
            raw_types = []
        normalized_types = normalize_degradation_types(
            [str(value) for value in raw_types]
        )
        degradation_phrase = format_degradation_phrase(normalized_types)

        base_response = str(base_record.get("response", "")).strip()
        response = build_temporal_response(
            base_caption=base_response,
            start_time=start_time,
            end_time=end_time,
            degradation_phrase=degradation_phrase,
        )

        record: dict[str, Any] = {
            "id": f"nisqa_temporal_{int(row['index']):05d}",
            "audios": [audio_value],
            "response": response,
            "query": query,
            "mos": float(row["mos"]) if pd.notna(row.get("mos", None)) else None,
            "filename_deg": filename_deg,
            "filename_ref": str(row.get("filename_ref", "")),
            "mix_filename": mix_filename,
            "duration_seconds": float(row["duration_seconds"])
            if pd.notna(row.get("duration_seconds", None))
            else None,
            "mix_deg_segments": segments,
            "source_degradation_types": normalized_types,
        }

        if include_clean_path and "clean_path" in base_record:
            record["clean_path"] = base_record["clean_path"]

        output_records.append(record)

    write_processed_records(output_jsonl, output_records)

    print(f"Manifest rows: {len(manifest_df)}")
    print(f"Base captions: {len(caption_records)}")
    print(f"Wrote records: {len(output_records)}")
    print(f"Missing base caption rows: {missing_base}")
    print(f"Missing mix files: {missing_mix}")
    print(f"Output: {output_jsonl}")


if __name__ == "__main__":
    app()
