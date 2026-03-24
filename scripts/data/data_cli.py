"""
data_cli.py - CLI tool for downloading and preprocessing data.
"""

import os
from pathlib import Path
import typer

app = typer.Typer()


@app.command()
def download(
    bucket_name: str = "nisqa-dataset",
    source_blob_name: str = ".",
    destination_path: Path = Path("data/raw"),
):
    """Downloads data from Google Cloud Storage to a local directory."""
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
    """Stage 1: Preprocesses dataset for Supervised Fine-Tuning (SFT) and DPO."""
    print("Preprocessing data for SFT/DPO...")

    print("\n--- Step 1: Sampling Data ---")
    from asa.sampler import sample_data

    nisqa_corpus_path = data_path / "NISQA_Corpus"
    sample_data(nisqa_corpus_path, output_folder)

    print("\n--- Step 2: Generating Captions ---")
    from asa.caption_generator import process_single_file

    mos_input = output_folder / "mos_dataset.json"
    mos_output = output_folder / "train_nisqa_llama_10k.json"
    if mos_input.exists():
        if not mos_output.exists():
            print(f"Generating captions for MOS dataset: {mos_input} -> {mos_output}")
            process_single_file(str(mos_input), str(mos_output))
        else:
            print(
                f"Captions for MOS dataset already exist at {mos_output}, skipping generation."
            )

    ab_input = output_folder / "ab_dataset.json"
    ab_output = output_folder / "train_nisqa_abtest_llama_10k.json"
    if ab_input.exists():
        if not ab_output.exists():
            print(f"Generating captions for A/B dataset: {ab_input} -> {ab_output}")
            process_single_file(str(ab_input), str(ab_output))
        else:
            print(
                f"Captions for A/B dataset already exist at {ab_output}, skipping generation."
            )


if __name__ == "__main__":
    app()
