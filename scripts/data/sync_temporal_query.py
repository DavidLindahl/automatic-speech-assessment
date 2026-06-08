"""Rewrite the ``query`` field of temporal datasets in place.

The temporal test sets (FOR / LIVE / P501) were built with an older query
string (which also had a "degredation" typo). When the training sets switch to a
new query, the evaluator, which feeds each record's stored query to the model as
its prompt, must use the same query or train and eval prompts diverge.

This utility rewrites only the ``query`` field; responses, segments, and audio
references are left untouched. The default query matches ``DEFAULT_QUERY`` in
``build_nisqa_temporal_json.py``.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, List

import typer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from asa.processed_data import load_processed_records, write_processed_records

DEFAULT_QUERY = (
    "Please describe and evaluate the synthetic speech, and identify when the "
    "degradation occurs.<audio>"
)

app = typer.Typer()


@app.command()
def main(
    dataset_paths: List[Path] = typer.Option(
        ..., "--dataset-path", help="Temporal JSON/JSONL to rewrite in place."
    ),
    query: str = typer.Option(
        DEFAULT_QUERY,
        help="New query string to store in every record.",
    ),
) -> None:
    """Rewrite the query field of each given dataset in place."""
    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        records = load_processed_records(dataset_path)
        updated: list[dict[str, Any]] = []
        for record in records:
            new_record = dict(record)
            new_record["query"] = query
            updated.append(new_record)

        write_processed_records(dataset_path, updated)
        print(f"Rewrote query on {len(updated)} records: {dataset_path}")

    print(f"Query set to: {query}")


if __name__ == "__main__":
    app()
