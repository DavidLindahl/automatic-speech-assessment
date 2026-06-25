"""Build a MEDIUM-difficulty timestamp-cycle DPO set (fixed Goldilocks shift).

Diagnosis behind this run (see runs/INDEX + temporal-loss-design): on the strong
caption-last SFT base (IoU 0.88), temporal ALLD never beats SFT because the
preference signal is ~0. The model's own sampled intervals land within 0.4 s of
gold on 77% of clips, so the DPO reward margin stays flat at 0 (no gradient). The
graded-jitter set went the other way (0.5-4 s shifts) and collapsed the model into
degenerate repetition, because the 2-4 s shifts push the interval onto wrong
active regions.

This builds the Goldilocks middle: a single FIXED interval shift in the
0.5-1.0 s band (default 0.75 s) applied to the gold interval, big enough to be a
learnable preference signal but small enough to stay near the true window and not
trigger the collapse. Single-factor: caption and MOS stay gold (the chosen
target's head verbatim); only the trailing <a><f> clause is rewritten with the
shifted interval. One rejected per mix.

Tests whether there is ANY negative difficulty at which ALLD beats SFT, or whether
the model collapses the moment the signal becomes learnable.

Source: train_dpo_armA_sampled.json (its `chosen` is the gold caption-last target;
`mix_deg_segments` is the ground-truth interval in seconds; `duration_seconds`).

Usage:
  python scripts/data/build_dpo_medium_shift.py \
    --in-json  data/processed/dpo/train_dpo_armA_sampled.json \
    --out-json data/processed/dpo/train_dpo_armA_medshift.json \
    --shift 0.75
"""

import json
import re
import sys
from pathlib import Path

import typer

# Make `asa` importable when run as a plain script (repo `src/` layout).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Reuse the canonical encoder so the <a><f> tokens stay byte-compatible.
from asa.temporal_tokens import encode_time  # noqa: E402

TS_CLAUSE_RE = re.compile(
    r"\s*The degradation in the clip is between\s*"
    r"<a\d+>\s*<f\d+>\s*and\s*<a\d+>\s*<f\d+>\s*\.?\s*$"
)

app = typer.Typer(help="Build a medium-difficulty (fixed-shift) timestamp-cycle DPO set.")


def split_head(target: str):
    """Return (head, clause) splitting at the trailing timestamp clause, or None."""
    m = TS_CLAUSE_RE.search(target)
    if not m or m.start() == 0:
        return None
    return target[: m.start()], target[m.start():]


def shifted_interval(start: float, end: float, dur: float, shift: float):
    """Shift [start,end] by +shift (or -shift if +shift would exceed dur), clamped.

    Returns (s2, e2) or None if the result is degenerate or lands back on gold.
    """
    # Prefer shifting later; if that runs off the clip, shift earlier.
    if end + shift <= dur:
        s2, e2 = start + shift, end + shift
    elif start - shift >= 0.0:
        s2, e2 = start - shift, end - shift
    else:
        return None  # clip too short to move by `shift` either way
    s2 = max(0.0, min(s2, dur))
    e2 = max(0.0, min(e2, dur))
    if e2 - s2 < 0.2:  # degenerate after clamp
        return None
    # Must differ from gold by at least ~0.1 s after 0.1 s rounding, else no signal.
    if abs(s2 - start) < 0.1 and abs(e2 - end) < 0.1:
        return None
    return s2, e2


@app.command()
def main(
    in_json: Path = typer.Option(..., "--in-json", help="armA_sampled JSONL (chosen=gold target)."),
    out_json: Path = typer.Option(..., "--out-json", help="Output medium-shift DPO JSONL."),
    shift: float = typer.Option(0.75, "--shift", help="Fixed interval shift in seconds (Goldilocks band)."),
) -> None:
    records = [json.loads(line) for line in in_json.read_text().splitlines() if line.strip()]

    out = []
    dropped_nointerval = 0
    dropped_noseg = 0
    dropped_shift = 0
    for r in records:
        chosen = r.get("chosen")
        segs = r.get("mix_deg_segments")
        dur = r.get("duration_seconds")
        if not chosen or not segs or dur is None:
            dropped_noseg += 1
            continue
        split = split_head(chosen)
        if split is None:
            dropped_nointerval += 1
            continue
        head, _gold_clause = split
        gstart = float(segs[0]["start"])
        gend = float(segs[0]["end"])
        sh = shifted_interval(gstart, gend, float(dur), shift)
        if sh is None:
            dropped_shift += 1
            continue
        s2, e2 = sh
        rejected_clause = (
            f" The degradation in the clip is between "
            f"{encode_time(s2)} and {encode_time(e2)}."
        )
        rejected = head + rejected_clause
        rec = dict(r)
        rec["rejected"] = rejected
        rec["medshift_seconds"] = shift
        rec["rejected_interval"] = [round(s2, 2), round(e2, 2)]
        out.append(rec)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text("\n".join(json.dumps(rec) for rec in out) + "\n")
    typer.echo(
        f"Wrote {len(out)} pairs to {out_json} (shift={shift}s). "
        f"Dropped: no-seg {dropped_noseg}, no-clause {dropped_nointerval}, "
        f"unshiftable {dropped_shift}."
    )
    # Sanity: chosen head must be byte-identical to rejected head (single-factor).
    bad = sum(1 for rec in out if split_head(rec["chosen"])[0] != split_head(rec["rejected"])[0])
    typer.echo(f"Single-factor check: {bad} pairs where the head differs (must be 0).")


if __name__ == "__main__":
    app()
