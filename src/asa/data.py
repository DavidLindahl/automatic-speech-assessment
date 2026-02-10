from pathlib import Path
import os


import typer
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

app = typer.Typer()


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

        # Find all file CSVs
        self.csv_files = list(data_path.rglob("*_file.csv"))
        if not self.csv_files:
            raise FileNotFoundError(f"No *_file.csv found in {data_path}")

        # Load and concatenate
        self.df = pd.concat([pd.read_csv(f) for f in self.csv_files], ignore_index=True)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        row = self.df.iloc[index]

        # Construct path: data_path / filepath_deg
        # filepath_deg in CSV is usually relative to the corpus root
        # e.g. NISQA_TEST_LIVETALK/deg/c01_f1.wav
        audio_path = self.data_path / row["filepath_deg"]

        if not audio_path.exists():
            # Try finding it relative to the CSV file's directory if the above fails
            # But based on inspection, it seems relative to corpus root.
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sample_rate = torchaudio.load(audio_path)

        return {
            "audio": waveform,
            "mos": row["mos"],
            "sample_rate": sample_rate,
            "filename": row["filename_deg"],
        }

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)

        for i in range(len(self)):
            sample = self[i]
            # Save as pt file
            torch.save(sample, output_folder / f"{i}.pt")


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

    # Ensure destination exists
    destination_path.mkdir(parents=True, exist_ok=True)

    # Try to create a client (checks for credentials first)
    try:
        client = storage.Client()
    except Exception:
        # If no credentials found, fall back to anonymous client
        print("No credentials found. Using anonymous access...")
        client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(
        prefix=source_blob_name if source_blob_name != "." else None
    )
    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        # Construct local path
        relative_path = os.path.relpath(blob.name, source_blob_name)
        local_path = destination_path / relative_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {blob.name} to {local_path}...")
        blob.download_to_filename(str(local_path))


@app.command()
def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    app()
