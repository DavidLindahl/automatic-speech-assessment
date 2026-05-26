"""Audit textual diversity of `response` fields in a JSONL/JSON dataset.

Reports unique-response count, top-K most-duplicated responses, and the
length distribution (in whitespace tokens). Used to verify that the SFT
training data has enough surface variety for the model to learn from.

Usage::

    uv run python scripts/analysis/audit_response_diversity.py \
        data/processed/train_nisqa_llama_10k.json \
        --top-k 10
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def _load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


@app.command()
def main(
    dataset_path: Path = typer.Argument(..., help="Path to JSON or JSONL records."),
    field: str = typer.Option("response", help="Field to audit."),
    top_k: int = typer.Option(10, help="Show K most-duplicated responses."),
    snippet_chars: int = typer.Option(120, help="Truncate previews to this length."),
) -> None:
    """Audit response diversity in a dataset."""
    records = _load_records(dataset_path)
    if not records:
        typer.echo(f"No records found in {dataset_path}")
        raise typer.Exit(1)

    responses = [str(r.get(field, "")) for r in records]
    n = len(responses)
    unique = len(set(responses))
    counts = Counter(responses)

    lengths = [len(r.split()) for r in responses]
    p10 = _percentile(lengths, 0.10)
    p50 = _percentile(lengths, 0.50)
    p90 = _percentile(lengths, 0.90)
    avg_len = sum(lengths) / n

    typer.echo("=" * 60)
    typer.echo(f"Dataset:  {dataset_path}")
    typer.echo(f"Field:    {field}")
    typer.echo(f"Samples:  {n}")
    typer.echo(f"Unique:   {unique} ({unique / n:.1%})")
    typer.echo(
        f"Length tokens (whitespace): mean={avg_len:.1f} "
        f"p10={p10} p50={p50} p90={p90}"
    )
    typer.echo("-" * 60)
    typer.echo(f"Top {top_k} most-duplicated responses:")
    for i, (text, cnt) in enumerate(counts.most_common(top_k), 1):
        preview = text.replace("\n", " ")
        if len(preview) > snippet_chars:
            preview = preview[:snippet_chars] + "..."
        typer.echo(f"  {i:2d}. n={cnt:4d} ({cnt / n:5.1%})  {preview}")
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
