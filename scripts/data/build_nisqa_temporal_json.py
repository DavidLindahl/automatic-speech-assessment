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
import re
import sys
from typing import Any

import pandas as pd
import typer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from asa.processed_data import load_processed_records, write_processed_records
from asa.temporal_tokens import encode_time


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
    "Please describe and evaluate the synthetic speech, and identify when the "
    "degradation occurs.<audio>"
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


# Matches a leading subject phrase up to and including the first "is"/"has" verb,
# e.g. "This synthesized speech is ...", "The synthesized speech has ...", or
# "This speech is ...". The lazy prefix takes the first verb; the word boundary
# avoids matching the "is" inside "This". Used to drop the boilerplate opener so
# the global caption can be re-attached after a leading temporal clause.
_CAPTION_VERB_RE = re.compile(r"^.*?\b(is|has)\b\s+", re.IGNORECASE)


def splice_caption_from_verb(caption: str) -> str:
    """Strip the boilerplate subject opener, keeping from the first is/has verb.

    The global captions open with a fixed subject phrase ("This synthesized
    speech is/has ...") that is meaningless once a temporal clause leads the
    sentence. This keeps the descriptive content and the MOS score verbatim by
    returning the verb plus everything after it.

    Args:
        caption: A normalized global caption sentence block.

    Returns:
        The caption from its first is/has verb onward. If no is/has verb is
        present, the full caption is returned unchanged as a safe fallback.
    """
    collapsed = normalize_caption(caption)
    match = _CAPTION_VERB_RE.match(collapsed)
    if match is None:
        return collapsed
    verb = match.group(1).lower()
    rest = collapsed[match.end() :]
    return f"{verb} {rest}"


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
    if label_style == "global-caption-localization":
        # Global captioning task (same as the MOS-style global set) with a leading
        # temporal clause, free-text <|seconds|> timestamps. The boilerplate
        # opener is stripped and the global caption re-attached from its is/has
        # verb, so the descriptive text and the MOS score are kept verbatim.
        spliced = splice_caption_from_verb(base_caption)
        return (
            f"The degradation in the clip is between <|{start_time:.2f}|> "
            f"and <|{end_time:.2f}|> and {spliced}"
        )

    if label_style == "global-caption-anchoroffset":
        # Same as global-caption-localization but with TimeAudio-style discrete
        # <aN><fK> time tokens instead of free-text timestamps.
        spliced = splice_caption_from_verb(base_caption)
        return (
            "The degradation in the clip is between "
            f"{encode_time(start_time)} and {encode_time(end_time)} and {spliced}"
        )

    if label_style in {
        "global-caption-timelast",
        "global-caption-timelast-anchoroffset",
    }:
        # Caption-first twins of the two global-caption styles: the full global
        # caption leads verbatim and the temporal clause closes the response, so
        # the model commits to the timestamps at the position with the most
        # self-conditioning instead of as its first content tokens. Information
        # content matches the timestamp-first styles (no degradation category);
        # only the order changes, which makes the pair a clean order ablation.
        # The caption is kept whole (no verb splice) because it opens the
        # sentence, so its original subject phrase stays grammatical.
        prefix = normalize_caption(base_caption)
        if not prefix.endswith((".", "!", "?")):
            prefix = f"{prefix}."
        if label_style == "global-caption-timelast":
            clause = (
                "The degradation in the clip is between "
                f"<|{start_time:.2f}|> and <|{end_time:.2f}|>."
            )
        else:
            clause = (
                "The degradation in the clip is between "
                f"{encode_time(start_time)} and {encode_time(end_time)}."
            )
        return f"{prefix} {clause}"

    if label_style == "anchor-offset-localization":
        # TimeAudio-style discrete time tokens, timestamp-only (no category).
        # Matches the target shape: "... there is distortion between <a><f> ...".
        return (
            "The overall speech is clear, but there is distortion between "
            f"{encode_time(start_time)} and {encode_time(end_time)}."
        )

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
            "label_style must be 'global-caption-localization', "
            "'global-caption-anchoroffset', 'global-caption-timelast', "
            "'global-caption-timelast-anchoroffset', "
            "'anchor-offset-localization', 'clear-speech-localization', "
            "'localization-only', or 'caption-plus-localization'"
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
    caption_index: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Rebuild temporal query/response labels from stored segments and types.

    This is useful when the mixed audio and manifest-derived metadata are already
    present in an existing JSONL, but the textual targets need a new style.

    The record's own ``response`` may already carry a temporal clause (e.g. the
    global-caption styles), so reusing it as the base caption would duplicate
    the clause. When ``caption_index`` is provided (degraded-filename basename
    to base caption record, as built by :func:`build_caption_index`), the base
    caption is taken from there instead and the original caption text is
    restored verbatim. Records missing from the index fall back to their own
    response and are counted.

    Args:
        records: Existing temporal JSONL records.
        query: Query text written into every output record.
        label_style: Target text style for :func:`build_temporal_response`.
        caption_index: Optional caption lookup by ``filename_deg``.

    Returns:
        Tuple of relabeled records and the number of caption-index misses.
    """
    output_records: list[dict[str, Any]] = []
    caption_misses = 0
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

        base_caption = str(record.get("response", "")).strip()
        if caption_index is not None:
            base_record = caption_index.get(str(record.get("filename_deg", "")))
            if base_record is not None:
                base_caption = str(base_record.get("response", "")).strip()
            else:
                caption_misses += 1

        updated = dict(record)
        updated["query"] = query
        updated["source_degradation_types"] = normalized_types
        updated["response"] = build_temporal_response(
            base_caption=base_caption,
            start_time=start_time,
            end_time=end_time,
            degradation_phrase=degradation_phrase,
            label_style=label_style,
        )
        output_records.append(updated)
    return output_records, caption_misses


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
            "Target text style: 'global-caption-localization' for the global MOS "
            "caption with a leading temporal clause and free-text <|seconds|> "
            "timestamps, 'global-caption-anchoroffset' for the same with discrete "
            "<aN><fK> time tokens, 'global-caption-timelast' / "
            "'global-caption-timelast-anchoroffset' for the caption-first twins "
            "(full caption verbatim, temporal clause appended last; the order "
            "ablation), 'anchor-offset-localization' for TimeAudio-style "
            "discrete <a><f> time tokens (timestamp-only, no category), "
            "'clear-speech-localization' for the current metadata timestamp "
            "labels, 'localization-only' for short timestamp labels, or "
            "'caption-plus-localization' for the old caption-prefixed labels."
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
        "global-caption-localization",
        "global-caption-anchoroffset",
        "global-caption-timelast",
        "global-caption-timelast-anchoroffset",
        "anchor-offset-localization",
        "clear-speech-localization",
        "localization-only",
        "caption-plus-localization",
    }:
        raise ValueError(
            "label_style must be 'global-caption-localization', "
            "'global-caption-anchoroffset', 'global-caption-timelast', "
            "'global-caption-timelast-anchoroffset', "
            "'anchor-offset-localization', 'clear-speech-localization', "
            "'localization-only', or 'caption-plus-localization'"
        )

    if input_jsonl is not None:
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Input temporal JSONL not found: {input_jsonl}")
        records = load_processed_records(input_jsonl)
        # Join the original captions when available so relabeling a styled
        # JSONL (whose responses already carry a temporal clause) restores the
        # caption verbatim instead of nesting clauses.
        caption_index = None
        if caption_jsonl.exists():
            caption_index = build_caption_index(load_jsonl(caption_jsonl))
            print(f"Caption join: {caption_jsonl} ({len(caption_index)} captions)")
        output_records, caption_misses = relabel_existing_temporal_records(
            records=records,
            query=query,
            label_style=label_style,
            caption_index=caption_index,
        )
        write_processed_records(output_jsonl, output_records)
        print(f"Input records: {len(records)}")
        print(f"Wrote records: {len(output_records)}")
        if caption_index is not None:
            print(f"Caption-index misses (fell back to record response): {caption_misses}")
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
