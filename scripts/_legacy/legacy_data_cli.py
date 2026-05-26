#!/usr/bin/env python3
"""Legacy NISQA preprocessing CLI (download + caption generation).

Out of the active path since the 2026-04-13 pivot to temporal-localization on
NISQA-SIM mixes. Preserved for archival reproducibility of the older MOS-only
captioning pipeline. Standalone — run via:

    python scripts/_legacy/legacy_data_cli.py download ...
    python scripts/_legacy/legacy_data_cli.py generate-captions ...
"""

import os
import sys
from pathlib import Path

import typer


# Self-bootstrap so the legacy CLI can import asa.* without requiring the
# package to be installed.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))


app = typer.Typer(help=__doc__)


@app.command()
def download(
    bucket_name: str = "nisqa-dataset",
    source_blob_name: str = ".",
    destination_path: Path = Path("data/raw"),
):
    """Download the NISQA corpus from Google Cloud Storage."""
    from google.cloud import storage

    print(
        f"Downloading from gs://{bucket_name}/{source_blob_name} to {destination_path}..."
    )

    destination_path.mkdir(parents=True, exist_ok=True)

    try:
        client = storage.Client()
    except Exception:
        print("No credentials found. Using anonymous access...")
        client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(
        prefix=source_blob_name if source_blob_name != "." else None
    )
    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        relative_path = os.path.relpath(blob.name, source_blob_name)
        local_path = destination_path / relative_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {blob.name} to {local_path}...")
        blob.download_to_filename(str(local_path))


@app.command()
def generate_captions(
    data_path: Path = typer.Argument(Path("data/raw")),
    output_folder: Path = typer.Argument(Path("data/processed")),
) -> None:
    """Stage 1: preprocess the NISQA corpus into MOS + A/B JSONs and run captioning."""
    print("Preprocessing data for SFT/DPO...")

    # 1. Sampler — splits NISQA into MOS / AB datasets
    print("\n--- Step 1: Sampling Data ---")
    from asa.sampler import sample_data

    nisqa_corpus_path = data_path / "NISQA_Corpus"
    sample_data(nisqa_corpus_path, output_folder)

    # 2. Caption generation — uses Gemini to descriptive-caption each sample
    print("\n--- Step 2: Generating Captions ---")
    # Imported locally (and bootstrapped via sys.path above) because the
    # caption generator now lives next to this script under scripts/_legacy/.
    sys.path.append(str(Path(__file__).resolve().parent))
    from caption_generator import process_single_file  # type: ignore[import-not-found]

    mos_input = output_folder / "mos_dataset.json"
    mos_output = output_folder / "train_nisqa_llama_10k.json"
    if mos_input.exists():
        if not mos_output.exists():
            print(f"Generating captions for MOS dataset: {mos_input} -> {mos_output}")
            process_single_file(str(mos_input), str(mos_output))
        else:
            print(
                f"Captions for MOS dataset already exist at {mos_output}, skipping."
            )

    # AB dataset path kept for archival reproducibility even though the AB
    # paradigm was cut on 2026-04-13. Only fires if `ab_dataset.json` exists.
    ab_input = output_folder / "ab_dataset.json"
    ab_output = output_folder / "train_nisqa_abtest_llama_10k.json"
    if ab_input.exists():
        if not ab_output.exists():
            print(f"Generating captions for A/B dataset: {ab_input} -> {ab_output}")
            process_single_file(str(ab_input), str(ab_output))
        else:
            print(
                f"Captions for A/B dataset already exist at {ab_output}, skipping."
            )


if __name__ == "__main__":
    app()
