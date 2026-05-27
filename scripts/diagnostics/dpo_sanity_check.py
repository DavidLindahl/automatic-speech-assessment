#!/usr/bin/env python3
"""Lightweight sanity checks for evaluation outputs.

Designed to catch collapse patterns like constant MOS predictions,
near-duplicate responses, malformed MOS strings, and repeated text.
This is intentionally post-hoc and replication-safe: it does not alter
training or data construction, it only inspects saved eval results.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import pstdev

import typer

app = typer.Typer(help="Inspect eval outputs for collapse / reward-hacking patterns.")


MALFORMED_PATTERNS = [
    re.compile(r"\b\d+\.\d+\.\d+\b"),
    re.compile(r"(This speech has .*?)(\1)+", re.IGNORECASE),
]


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


@app.command()
def inspect(
    result_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    duplicate_threshold: float = typer.Option(
        0.97, help="Similarity threshold for near-duplicate grouping."
    ),
    top_k: int = typer.Option(5, help="How many common response templates to show."),
) -> None:
    files = sorted(result_dir.glob("*_results.json"))
    if not files:
        raise typer.BadParameter(f"No *_results.json files found under {result_dir}")

    overall_flags: list[str] = []

    for path in files:
        payload = json.loads(path.read_text())
        rows = payload["results"]
        preds = [row.get("predicted_mos") for row in rows]
        texts = [str(row.get("predicted_response", "")) for row in rows]
        norm_texts = [normalize_text(t) for t in texts]

        unique_mos = len(set(preds))
        mos_std = pstdev(preds) if len(preds) > 1 else 0.0
        counts = Counter(norm_texts)
        top_counts = counts.most_common(top_k)
        dominant_text_share = top_counts[0][1] / len(rows)

        malformed = sum(any(p.search(t) for p in MALFORMED_PATTERNS) for t in texts)
        repeated_prefix = sum(t.lower().count("this speech has") > 1 for t in texts)

        # crude near-duplicate rate against the most common template
        template = top_counts[0][0]
        near_dup = sum(
            SequenceMatcher(None, template, t).ratio() >= duplicate_threshold
            for t in norm_texts
        )
        near_dup_share = near_dup / len(rows)

        flags = []
        if unique_mos <= 2:
            flags.append(f"very low MOS diversity ({unique_mos} unique values)")
        if mos_std < 0.15:
            flags.append(f"predicted MOS variance is tiny (std={mos_std:.3f})")
        if dominant_text_share > 0.5:
            flags.append(f"one response template dominates ({dominant_text_share:.1%})")
        if near_dup_share > 0.8:
            flags.append(f"near-duplicate rate is extreme ({near_dup_share:.1%})")
        if malformed > 0:
            flags.append(f"malformed responses detected ({malformed})")
        if repeated_prefix > 0:
            flags.append(f"loop/repetition pattern detected ({repeated_prefix})")

        typer.echo(f"\n=== {path.name} ===")
        typer.echo(f"samples: {len(rows)}")
        typer.echo(f"metrics: {payload.get('metrics', {})}")
        typer.echo(f"unique predicted MOS: {unique_mos}")
        typer.echo(f"predicted MOS std: {mos_std:.4f}")
        typer.echo(f"malformed responses: {malformed}")
        typer.echo(f"loop-like repeated prefix count: {repeated_prefix}")
        typer.echo("top response templates:")
        for text, count in top_counts:
            preview = text[:140].replace("\n", " ")
            typer.echo(f"  - {count:>4}x | {preview}")

        if flags:
            typer.echo("flags:")
            for flag in flags:
                typer.echo(f"  - {flag}")
            overall_flags.extend([f"{path.name}: {flag}" for flag in flags])
        else:
            typer.echo("flags: none")

    typer.echo("\n=== overall verdict ===")
    if overall_flags:
        typer.echo("suspicious / collapsed behaviour detected")
        for flag in overall_flags:
            typer.echo(f"- {flag}")
    else:
        typer.echo("no obvious collapse patterns detected")


if __name__ == "__main__":
    app()
