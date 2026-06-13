"""Splice ARM A's single self-sample into three single-factor DPO cycle datasets.

This is the downstream string surgery for the factorized cyclic DPO on ARM A
(see research/temporal-loss-design.md). The expensive part, sampling ARM A once
per clip, is already done (train_dpo_armA_sampled.json, job 28647628). Here we
recombine the sampled (flawed) output with the gold target factor by factor so
each DPO cycle gets a clean, single-factor preference gradient.

The temporal target has a fixed shape:

    <caption ... that mentions the MOS number> The degradation in the clip is
    between <aN><fK> and <aN><fK>.

So three independently-corruptible factors live in one string:
  1. the MOS number (inside the caption sentence),
  2. the descriptive caption (the head, which carries the MOS with it),
  3. the timestamp interval (the trailing clause).

Hard rule (memo line 62): never corrupt two factors in the same rejected. The
raw sample jitters all three at once, so it cannot be used as a DPO rejected
directly. We split chosen and rejected at the timestamp clause, then build:

  MOS cycle      chosen = gold; rejected = gold head with ONLY the MOS number
                 swapped to the sampled MOS, gold timestamps. Pure rating
                 pressure. Dropped when the sampled MOS equals gold (no signal).

  caption cycle  chosen = gold; rejected = the sampled head (caption + its MOS)
                 with gold timestamps. Description pressure. MOS rides inside the
                 caption by construction, so caption and MOS move together here
                 deliberately, this is the "caption+MOS" cycle of the plan.

  timestamp      chosen = gold; rejected = gold head with ONLY the sampled
  cycle          timestamps. Localization pressure. This is the SAMPLED source
                 for the timestamp cycle; the synthetic-jitter set
                 (train_dpo_armA_jitter.json) is the other source, and the two
                 are run head-to-head as a 2-way A/B.

Every output record keeps the source schema (audios, query, mos, mix_deg_segments,
...) and overwrites only `chosen` / `rejected`, so the DPO collator consumes them
unchanged. No model inference.
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = typer.Typer(help="Build single-factor DPO cycle datasets from ARM A's self-sample.")

# The trailing timestamp clause, e.g.
# " The degradation in the clip is between <a3><f6> and <a5><f1>."
# Captured so the head (everything before it) splits off cleanly. Anchored to the
# end of the string: the clause is always last in the caption-last format.
TS_CLAUSE_RE = re.compile(
    r"\s*The degradation in the clip is between\s*"
    r"<a\d+>\s*<f\d+>\s*and\s*<a\d+>\s*<f\d+>\s*\.?\s*$"
)

# The MOS number inside a caption sentence. Mirrors evaluate.extract_mos: an
# explicit "MOS ... <number>" mention, number captured. re.IGNORECASE so "mos"
# variants match too. Used to locate and rewrite the rating in place.
MOS_IN_TEXT_RE = re.compile(r"(MOS(?:[^0-9]+))(\d+(?:\.\d+)?)", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dict records."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_head_and_clause(target: str) -> Optional[tuple[str, str]]:
    """Split a target into (head, timestamp_clause).

    The head is the caption (which carries the MOS); the clause is the trailing
    " The degradation in the clip is between <a><f> and <a><f>." Returns None if
    the target does not end in a well-formed timestamp clause, or the head is
    empty. A None means the record is unusable and is dropped by the caller.
    """
    match = TS_CLAUSE_RE.search(target)
    if match is None:
        return None
    head = target[: match.start()].rstrip()
    clause = target[match.start():].strip()
    if not head:
        return None
    return head, clause


def swap_mos(head: str, new_mos: float) -> Optional[str]:
    """Rewrite the MOS number in `head` to `new_mos`, leaving all else intact.

    The number is formatted to one decimal to match the label style ("1.2",
    "3.0"). Returns None if no MOS mention is present (so the record is dropped
    rather than silently left uncorrupted).
    """
    match = MOS_IN_TEXT_RE.search(head)
    if match is None:
        return None
    formatted = f"{new_mos:.1f}"
    start, end = match.span(2)
    return head[:start] + formatted + head[end:]


def extract_mos_value(text: str) -> Optional[float]:
    """Extract the MOS number from text, or None.

    Same precedence as evaluate.extract_mos (explicit MOS mention first, else the
    last number), but returns None on total failure instead of 0.0 so a missing
    MOS cannot masquerade as a real 0.0 rating.
    """
    match = re.search(r"MOS(?:[^0-9]+)(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    numbers = re.findall(r"(\d+(?:\.\d+)?)", text)
    if numbers:
        return float(numbers[-1])
    return None


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as JSONL (one object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def base_pair(record: dict[str, Any], chosen: str, rejected: str) -> dict[str, Any]:
    """Clone a source record and overwrite only chosen/rejected.

    Keeps every other key (audios, query, mos, mix_deg_segments, ...) so the DPO
    collator reads the spliced record exactly like the original.
    """
    pair = dict(record)
    pair["chosen"] = chosen
    pair["rejected"] = rejected
    return pair


@app.command()
def build(
    input_json: Path = typer.Option(
        Path("data/processed/dpo/train_dpo_armA_sampled.json"),
        help="ARM A's self-sample: gold in `chosen`, sampled in `rejected`.",
    ),
    out_mos: Path = typer.Option(
        Path("data/processed/dpo/train_dpo_armA_cycle_mos.json"),
        help="MOS cycle output (sampled MOS only).",
    ),
    out_caption: Path = typer.Option(
        Path("data/processed/dpo/train_dpo_armA_cycle_caption.json"),
        help="Caption+MOS cycle output (sampled head, gold timestamps).",
    ),
    out_timestamp: Path = typer.Option(
        Path("data/processed/dpo/train_dpo_armA_cycle_timestamp_sampled.json"),
        help="Timestamp cycle output, SAMPLED source (sampled timestamps only).",
    ),
) -> None:
    """Read the self-sample and write the three single-factor cycle datasets."""
    logging.info("Loading self-sample from %s", input_json)
    records = load_jsonl(input_json)
    logging.info("Loaded %d records", len(records))

    mos_pairs: list[dict[str, Any]] = []
    caption_pairs: list[dict[str, Any]] = []
    timestamp_pairs: list[dict[str, Any]] = []

    skip_no_chosen = 0
    skip_chosen_split = 0
    skip_rejected_split = 0
    skip_mos_same = 0
    skip_mos_missing = 0
    skip_caption_same = 0
    skip_ts_same = 0

    for record in records:
        gold = record.get("chosen") or record.get("response")
        sampled = record.get("rejected")
        if not gold or not sampled:
            skip_no_chosen += 1
            continue

        gold_split = split_head_and_clause(gold)
        if gold_split is None:
            skip_chosen_split += 1
            continue
        gold_head, gold_clause = gold_split

        sampled_split = split_head_and_clause(sampled)
        if sampled_split is None:
            # The sample degenerated (token salad, lowercase clause, stray sign).
            # Drop it: we cannot trust any factor of an unparseable rejected.
            skip_rejected_split += 1
            continue
        sampled_head, sampled_clause = sampled_split

        # --- Timestamp cycle (sampled source): gold head + sampled clause. ---
        if sampled_clause == gold_clause:
            skip_ts_same += 1
        else:
            rejected = f"{gold_head} {sampled_clause}"
            timestamp_pairs.append(base_pair(record, gold, rejected))

        # --- Caption+MOS cycle: sampled head + gold clause. ---
        if sampled_head == gold_head:
            skip_caption_same += 1
        else:
            rejected = f"{sampled_head} {gold_clause}"
            caption_pairs.append(base_pair(record, gold, rejected))

        # --- MOS cycle: gold head with ONLY the MOS swapped to the sampled MOS. ---
        sampled_mos = extract_mos_value(sampled_head)
        gold_mos = extract_mos_value(gold_head)
        if sampled_mos is None or gold_mos is None:
            skip_mos_missing += 1
        elif abs(sampled_mos - gold_mos) < 0.05:
            skip_mos_same += 1
        else:
            corrupted_head = swap_mos(gold_head, sampled_mos)
            if corrupted_head is None or corrupted_head == gold_head:
                skip_mos_missing += 1
            else:
                rejected = f"{corrupted_head} {gold_clause}"
                mos_pairs.append(base_pair(record, gold, rejected))

    write_jsonl(out_mos, mos_pairs)
    write_jsonl(out_caption, caption_pairs)
    write_jsonl(out_timestamp, timestamp_pairs)

    logging.info("=" * 60)
    logging.info("Cycle splice complete from %d source records", len(records))
    logging.info("MOS cycle:        %d pairs -> %s", len(mos_pairs), out_mos)
    logging.info("Caption+MOS cycle: %d pairs -> %s", len(caption_pairs), out_caption)
    logging.info(
        "Timestamp cycle (sampled): %d pairs -> %s",
        len(timestamp_pairs),
        out_timestamp,
    )
    logging.info("-" * 60)
    logging.info("Skipped (no chosen/rejected):     %d", skip_no_chosen)
    logging.info("Skipped (gold unparseable):       %d", skip_chosen_split)
    logging.info("Skipped (sampled unparseable):    %d", skip_rejected_split)
    logging.info("MOS-cycle drops (MOS unchanged):  %d", skip_mos_same)
    logging.info("MOS-cycle drops (MOS missing):    %d", skip_mos_missing)
    logging.info("Caption-cycle drops (head same):  %d", skip_caption_same)
    logging.info("Timestamp-cycle drops (ts same):  %d", skip_ts_same)
    logging.info("=" * 60)


if __name__ == "__main__":
    app()
