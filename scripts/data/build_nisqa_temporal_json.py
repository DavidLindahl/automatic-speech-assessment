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
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from asa.processed_data import load_processed_records, write_processed_records


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
    "Please describe and evaluate the synthetic speech, and find timestamps "
    "for the degredation<audio>"
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
    label_style: str = "clear-speech-localization",
) -> str:
    """Compose target response that includes both quality text and localization."""
    if label_style == "clear-speech-localization":
        return (
            "The overall speech is clear, but the quality is interrupted by "
            f"{degradation_phrase} occurring between <|{start_time:.2f}|> "
            f"and <|{end_time:.2f}|>."
        )

    if label_style == "localization-only":
        return (
            f"A localized degradation occurs between <|{start_time:.2f}|> "
            f"and <|{end_time:.2f}|>."
        )

    if label_style != "caption-plus-localization":
        raise ValueError(
            "label_style must be 'clear-speech-localization', 'localization-only', "
            "or 'caption-plus-localization'"
        )

    prefix = normalize_caption(base_caption)
    if not prefix.endswith((".", "!", "?")):
        prefix = f"{prefix}."

    return (
        f"{prefix} The quality is interrupted by {degradation_phrase} "
        f"occurring between <|{start_time:.2f}|> and <|{end_time:.2f}|>."
    )


def parse_existing_list_field(value: Any) -> list[Any]:
    """Parse a list field that may already be a list or JSON-serialized text."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def relabel_existing_temporal_records(
    records: list[dict[str, Any]],
    query: str,
    label_style: str,
) -> list[dict[str, Any]]:
    """Rebuild temporal query/response labels from stored segments and types.

    This is useful when the mixed audio and manifest-derived metadata are already
    present in an existing JSONL, but the textual targets need a new style.
    """
    output_records: list[dict[str, Any]] = []
    for record in records:
        segments = parse_existing_list_field(record.get("mix_deg_segments", []))
        start_time, end_time = pick_primary_segment(segments)

        raw_types = parse_existing_list_field(
            record.get("source_degradation_types", [])
        )
        normalized_types = normalize_degradation_types(
            [str(value) for value in raw_types]
        )
        degradation_phrase = format_degradation_phrase(normalized_types)

        updated = dict(record)
        updated["query"] = query
        updated["source_degradation_types"] = normalized_types
        updated["response"] = build_temporal_response(
            base_caption=str(record.get("response", "")).strip(),
            start_time=start_time,
            end_time=end_time,
            degradation_phrase=degradation_phrase,
            label_style=label_style,
        )
        output_records.append(updated)
    return output_records


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
    input_jsonl: Path | None = typer.Option(
        None,
        "--input-jsonl",
        help=(
            "Existing temporal JSONL to relabel from stored mix_deg_segments and "
            "source_degradation_types. If set, manifest/caption inputs are skipped."
        ),
    ),
    manifest_path: Path = typer.Option(
        Path("data/processed/nisqa_sim_mix_lowmos_active_3000/manifest.csv"),
        help="Temporal mix manifest CSV.",
    ),
    caption_jsonl: Path = typer.Option(
        Path("data/processed/sft/train_nisqa_llama_10k.json"),
        help="Base caption JSONL used for caption reuse.",
    ),
    mixes_dir: Path = typer.Option(
        Path("data/processed/nisqa_sim_mix_lowmos_active_3000"),
        help="Directory containing mixed WAV files from manifest.",
    ),
    output_jsonl: Path = typer.Option(
        Path("data/processed/temporal/train_nisqa_temporal_mix_3000.json"),
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
    label_style: str = typer.Option(
        "clear-speech-localization",
        help=(
            "Target text style: 'clear-speech-localization' for the current "
            "metadata timestamp labels, 'localization-only' for short timestamp "
            "labels, or 'caption-plus-localization' for the old caption-prefixed "
            "labels."
        ),
    ),
    include_clean_path: bool = typer.Option(
        True,
        "--include-clean-path/--no-include-clean-path",
        help="Keep clean_path field from base caption record.",
    ),
) -> None:
    """Build temporal SFT JSONL for NISQA temporal mixes."""
    if label_style not in {
        "clear-speech-localization",
        "localization-only",
        "caption-plus-localization",
    }:
        raise ValueError(
            "label_style must be 'clear-speech-localization', 'localization-only', "
            "or 'caption-plus-localization'"
        )

    if input_jsonl is not None:
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Input temporal JSONL not found: {input_jsonl}")
        records = load_processed_records(input_jsonl)
        output_records = relabel_existing_temporal_records(
            records=records,
            query=query,
            label_style=label_style,
        )
        write_processed_records(output_jsonl, output_records)
        print(f"Input records: {len(records)}")
        print(f"Wrote records: {len(output_records)}")
        print(f"Label style: {label_style}")
        print(f"Output: {output_jsonl}")
        return

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
            label_style=label_style,
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
    print(f"Label style: {label_style}")
    print(f"Output: {output_jsonl}")


if __name__ == "__main__":
    app()
