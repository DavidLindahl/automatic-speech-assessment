"""Generate temporal-factor (pair-type A) DPO pairs for the factorized temporal DPO.

This is the localization-pressure half of the Phase-2 factorized temporal DPO
(see research/temporal-loss-design.md, line 53). It is the complement of the
caption-pressure pairs in train_dpo_gc_plain.json (pair-type B), which were built
by sampling the SFT model and therefore jitter caption AND interval together.

Hard rule (memo line 62): never jitter caption and time in the same rejected. So
here the caption stays VERBATIM and only the two timestamp tokens are shifted by a
graded amount. This gives the model a clean "your interval is wrong, the caption
is right" gradient that targets the t-IoU collapse the frame probe diagnosed.

No model inference. The chosen target already exists in the source records (the
gc-plain DPO file carries it as the `chosen` field, and it always matches the gold
`mix_deg_segments` interval). We rewrite only the `<|start|>` / `<|end|>` tokens.

Jitter: for each record we emit one rejected per offset magnitude in {0.5, 1, 2,
4} s, with a deterministic sign per (record, magnitude) so the dataset is
reproducible without Math.random. Each shifted interval is clamped to
[0, duration] and rejected if it ends up degenerate (start >= end) or numerically
identical to the gold interval after clamping.

Output schema mirrors the gc-plain DPO file (same keys) so the existing DPO
training collator consumes it unchanged.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

import typer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from asa.temporal_tokens import encode_time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = typer.Typer(help="Build temporal-factor (interval-only-jitter) DPO pairs.")

# Free-text timestamp tokens, e.g. "<|3.57|>".
TS_PATTERN = re.compile(r"<\|(\d+(?:\.\d+)?)\|>")
# TimeAudio anchor/offset pairs, e.g. "<a3><f6>" (one decoded value each).
ANCHOROFFSET_PATTERN = re.compile(r"<a\d+>\s*<f\d+>")

# Graded jitter magnitudes in seconds (memo line 53).
DEFAULT_OFFSETS = (0.5, 1.0, 2.0, 4.0)


def detect_timestamp_format(text: str) -> Optional[str]:
    """Return 'anchoroffset', 'freetext', or None for the target's time format.

    Checks anchor/offset first because a malformed mix would otherwise be
    misread as free-text; a well-formed target carries exactly one format.
    """
    if len(ANCHOROFFSET_PATTERN.findall(text)) >= 2:
        return "anchoroffset"
    if len(TS_PATTERN.findall(text)) >= 2:
        return "freetext"
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dict records."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def gold_interval(record: dict[str, Any]) -> Optional[tuple[float, float]]:
    """Return the single ground-truth (start, end) interval, or None if absent."""
    segments = record.get("mix_deg_segments") or []
    if not segments:
        return None
    seg = segments[0]
    try:
        start = float(seg["start"])
        end = float(seg["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


def jitter_direction(record_id: str, magnitude: float) -> int:
    """Deterministic +1/-1 sign for a (record, magnitude) pair.

    Math.random is unavailable in this environment and we want the dataset to be
    reproducible, so the sign is a stable hash of the record id and magnitude.
    """
    key = f"{record_id}|{magnitude}"
    return 1 if (hash(key) & 1) == 0 else -1


def shift_interval(
    start: float,
    end: float,
    offset: float,
    duration: float,
) -> Optional[tuple[float, float]]:
    """Shift both endpoints by `offset` (signed), clamp to [0, duration].

    Returns None if the result is degenerate or collapses onto the gold interval.
    """
    new_start = max(0.0, min(duration, start + offset))
    new_end = max(0.0, min(duration, end + offset))
    if new_end - new_start < 0.05:
        return None
    if abs(new_start - start) < 0.01 and abs(new_end - end) < 0.01:
        return None
    return new_start, new_end


def rewrite_timestamps(
    chosen: str, new_start: float, new_end: float
) -> Optional[str]:
    """Replace the two timestamp tokens in `chosen`, leaving all other text intact.

    Handles both target formats: free-text ``<|s|>`` and TimeAudio
    ``<aN><fK>``. The format is detected from ``chosen`` itself, so the same
    generator serves the plain and the anchor-offset arms. Exactly two
    timestamp tokens must be present, or the record is skipped (returns None).
    """
    fmt = detect_timestamp_format(chosen)
    if fmt == "anchoroffset":
        pattern = ANCHOROFFSET_PATTERN
        replacements = [encode_time(new_start), encode_time(new_end)]
    elif fmt == "freetext":
        pattern = TS_PATTERN
        replacements = [f"<|{new_start:.2f}|>", f"<|{new_end:.2f}|>"]
    else:
        return None

    matches = list(pattern.finditer(chosen))
    if len(matches) != 2:
        return None
    out = []
    cursor = 0
    for match, replacement in zip(matches, replacements):
        out.append(chosen[cursor : match.start()])
        out.append(replacement)
        cursor = match.end()
    out.append(chosen[cursor:])
    return "".join(out)


@app.command()
def generate(
    input_json: Path = typer.Option(
        Path("data/processed/dpo/train_dpo_gc_plain.json"),
        help=(
            "Source with the gold target + `mix_deg_segments`. Accepts either a "
            "DPO file (gold in `chosen`) or an SFT training file (gold in "
            "`response`); `chosen` takes precedence when both are present, and "
            "the output always carries the gold as `chosen`."
        ),
    ),
    output_json: Path = typer.Option(
        Path("data/processed/dpo/train_dpo_gc_temporal_factor.json"),
        help="Output JSONL of temporal-factor (interval-only-jitter) pairs.",
    ),
    offsets: str = typer.Option(
        "0.5,1,2,4", help="Comma-separated jitter magnitudes in seconds."
    ),
    max_records: Optional[int] = typer.Option(
        None, help="Cap source records (debugging)."
    ),
) -> None:
    """Emit one rejected per jitter magnitude per record; caption stays verbatim."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    magnitudes = tuple(float(x) for x in offsets.split(",") if x.strip())
    if not magnitudes:
        magnitudes = DEFAULT_OFFSETS

    logging.info("Loading source records from %s", input_json)
    records = load_jsonl(input_json)
    if max_records:
        records = records[:max_records]
    logging.info("Loaded %d source records; offsets=%s", len(records), magnitudes)

    written = 0
    skipped_no_interval = 0
    skipped_no_chosen = 0
    skipped_degenerate = 0

    with output_json.open("w", encoding="utf-8") as out:
        for record in records:
            # Gold target: a DPO file carries it as `chosen`; an SFT training
            # file carries it as `response`. Prefer `chosen` when present.
            chosen = record.get("chosen") or record.get("response")
            interval = gold_interval(record)
            duration = float(record.get("duration_seconds") or 0.0)
            if not chosen:
                skipped_no_chosen += 1
                continue
            if interval is None or duration <= 0.0:
                skipped_no_interval += 1
                continue
            start, end = interval
            rid = str(record.get("id", record.get("mix_filename", "")))

            for magnitude in magnitudes:
                sign = jitter_direction(rid, magnitude)
                shifted = shift_interval(start, end, sign * magnitude, duration)
                if shifted is None:
                    # Try the opposite direction before giving up (e.g. clamped at an edge).
                    shifted = shift_interval(start, end, -sign * magnitude, duration)
                if shifted is None:
                    skipped_degenerate += 1
                    continue
                rejected = rewrite_timestamps(chosen, shifted[0], shifted[1])
                if rejected is None or rejected == chosen:
                    skipped_degenerate += 1
                    continue

                pair = dict(record)
                pair["chosen"] = chosen
                pair["rejected"] = rejected
                pair["temporal_jitter_seconds"] = round(magnitude, 3)
                pair["rejected_interval"] = [
                    round(shifted[0], 3),
                    round(shifted[1], 3),
                ]
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                written += 1

    logging.info("=" * 50)
    logging.info("Temporal-factor DPO pairs written: %d", written)
    logging.info("Skipped (no gold interval / duration): %d", skipped_no_interval)
    logging.info("Skipped (no chosen target): %d", skipped_no_chosen)
    logging.info("Skipped (degenerate / clamped onto gold): %d", skipped_degenerate)
    logging.info("Output: %s", output_json)
    logging.info("=" * 50)


if __name__ == "__main__":
    app()
