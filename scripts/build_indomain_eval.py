"""Build an in-domain NISQA evaluation split from the SFT training data.

NISQA TRAIN_SIM filenames look like ``c<NNNNN>_<base-utterance-id>.wav``,
where ``c<NNNNN>`` is a noise/distortion condition and ``<base>`` is the
underlying clean source utterance. To prevent leakage where the test set
contains the same source utterance under a different condition, we hold
out by **base utterance ID**, not by filename.

Output is written as JSONL (one record per line) so it matches the
existing test_FOR.json / test_LIVE.json / test_P501.json format.

Usage::

    uv run python scripts/build_indomain_eval.py \
        --input  data/processed/train_nisqa_llama_10k.json \
        --train-out data/processed/train_nisqa_llama_indomain.json \
        --eval-out  data/processed/test_nisqa_indomain.json \
        --eval-frac 0.05 \
        --seed 42
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)

CONDITION_PREFIX_RE = re.compile(r"^c\d+_")


def _load_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _base_utterance_id(record: dict) -> str:
    audio = record["audios"][0]
    name = Path(audio).stem
    return CONDITION_PREFIX_RE.sub("", name)


@app.command()
def main(
    input: Path = typer.Option(
        ..., help="Input JSONL training file (e.g. train_nisqa_llama_10k.json)."
    ),
    train_out: Path = typer.Option(..., help="Output path for filtered training set."),
    eval_out: Path = typer.Option(..., help="Output path for in-domain eval set."),
    eval_frac: float = typer.Option(
        0.05, help="Fraction of base utterances to hold out for eval."
    ),
    seed: int = typer.Option(42, help="RNG seed for the split."),
) -> None:
    """Build an in-domain eval split with no base-utterance leakage."""
    records = _load_jsonl(input)
    if not records:
        typer.echo(f"No records in {input}")
        raise typer.Exit(1)

    by_base: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_base[_base_utterance_id(r)].append(r)

    bases = sorted(by_base.keys())
    rng = random.Random(seed)
    rng.shuffle(bases)
    n_eval_bases = max(1, int(round(len(bases) * eval_frac)))
    eval_bases = set(bases[:n_eval_bases])

    train_records: list[dict] = []
    eval_records: list[dict] = []
    for base, group in by_base.items():
        if base in eval_bases:
            eval_records.extend(group)
        else:
            train_records.extend(group)

    _write_jsonl(train_out, train_records)
    _write_jsonl(eval_out, eval_records)

    typer.echo("=" * 60)
    typer.echo(f"Input:                  {input}")
    typer.echo(f"Total records:          {len(records)}")
    typer.echo(f"Unique base utterances: {len(bases)}")
    typer.echo(f"Eval base utterances:   {len(eval_bases)} ({eval_frac:.1%})")
    typer.echo(f"Train records out:      {len(train_records)}  -> {train_out}")
    typer.echo(f"Eval records out:       {len(eval_records)}  -> {eval_out}")
    typer.echo(
        f"No base-utterance leakage: "
        f"{len(set(map(_base_utterance_id, eval_records)) & set(map(_base_utterance_id, train_records))) == 0}"
    )
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
