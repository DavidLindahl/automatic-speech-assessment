"""Download and materialize the QualiSpeech dataset locally.

This script downloads the Hugging Face dataset repository snapshot, extracts the
audio archive, and verifies that the key metadata files are available for local
use.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import shutil
import zipfile

from huggingface_hub import snapshot_download


DATASET_ID = "tsinghua-ee/QualiSpeech"
DEFAULT_REQUIRED_FILES = ("train.csv", "val.csv", "test.csv")
DEFAULT_ARCHIVE = "wav_part1.zip"


def download_dataset(output_dir: Path) -> Path:
    """Download the QualiSpeech repository snapshot.

    Args:
        output_dir: Directory where the snapshot should be stored.

    Returns:
        The path to the downloaded snapshot.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            local_dir=str(output_dir),
        )
    )


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    """Extract the dataset audio archive into the target directory."""

    if not archive_path.exists():
        raise FileNotFoundError(f"Missing archive: {archive_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zip_file:
        zip_file.extractall(target_dir)


def validate_dataset(dataset_dir: Path) -> None:
    """Validate that the downloaded dataset contains the expected files."""

    missing_files = [name for name in DEFAULT_REQUIRED_FILES if not (dataset_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

    archive_path = dataset_dir / DEFAULT_ARCHIVE
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing required archive: {archive_path}")


def main() -> None:
    """Run the downloader and materialize the dataset locally."""

    parser = ArgumentParser(description="Download QualiSpeech for local use.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/QualiSpeech"),
        help="Directory where the dataset snapshot should be stored.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download the archive without extracting wav_part1.zip.",
    )
    args = parser.parse_args()

    dataset_dir = download_dataset(args.output_dir)
    validate_dataset(dataset_dir)

    if not args.skip_extract:
        extract_archive(dataset_dir / DEFAULT_ARCHIVE, dataset_dir)

    wav_count = sum(1 for path in dataset_dir.rglob("*.wav"))

    print(f"Downloaded dataset to: {dataset_dir}")
    print(f"Extracted WAV files: {wav_count}")
    print("Ready for local inspection.")


if __name__ == "__main__":
    main()
