"""Prepare a small synthetic temporal dataset for a fast smoke test.

This script generates a train/test split of localized synthetic degradations on
clean speech, then distills the metadata into the JSONL format expected by the
existing SFT pipeline.
"""

import json
import random
from pathlib import Path

import torch
import torchaudio
import typer

from asa.distill_temporal_targets import generate_targets
from asa.generate_temporal_data import (
    apply_clipping,
    apply_packet_loss,
    load_clean_speech_dataset,
    load_noise_dataset,
    overlay_noise,
)

app = typer.Typer()


def _generate_split(
    num_samples: int,
    output_dir: Path,
    metadata_path: Path,
) -> None:
    """Generate one split of temporal synthetic data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    clean_iter = load_clean_speech_dataset()
    noise_iter = load_noise_dataset()
    metadata: list[dict[str, object]] = []

    for idx in range(num_samples):
        try:
            clean_sample = next(clean_iter)
        except StopIteration:
            clean_iter = load_clean_speech_dataset()
            clean_sample = next(clean_iter)

        clean_audio = clean_sample["audio"]
        clean_tensor = torch.tensor(clean_audio["array"]).view(1, -1).float()
        sr = 16000
        if clean_audio["sampling_rate"] != sr:
            clean_tensor = torchaudio.functional.resample(
                clean_tensor,
                clean_audio["sampling_rate"],
                sr,
            )

        degradation_choice = random.choice(["noise", "packet_loss", "clipping"])
        if degradation_choice == "noise":
            try:
                noise_sample = next(noise_iter)
            except StopIteration:
                noise_iter = load_noise_dataset()
                noise_sample = next(noise_iter)
            noise_audio = noise_sample["audio"]
            noise_tensor = torch.tensor(noise_audio["array"]).view(1, -1).float()
            degraded_waveform, start_time, end_time = overlay_noise(
                clean_tensor,
                sr,
                noise_tensor,
                noise_audio["sampling_rate"],
                target_sr=sr,
            )
            degradation_type = str(noise_sample.get("category", "noise")).replace("_", " ")
        elif degradation_choice == "packet_loss":
            degraded_waveform, start_time, end_time = apply_packet_loss(clean_tensor, sr)
            degradation_type = "packet loss"
        else:
            degraded_waveform, start_time, end_time = apply_clipping(
                clean_tensor,
                sr,
                threshold=random.uniform(0.05, 0.15),
            )
            degradation_type = "clipping distortion"

        file_path = output_dir / f"temporal_smoke_{idx:04d}.wav"
        torchaudio.save(str(file_path), degraded_waveform, sr)
        metadata.append(
            {
                "audio_path": str(file_path).replace("\\", "/"),
                "degradation_type": degradation_type,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
            }
        )

    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


@app.command()
def main(
    train_samples: int = typer.Option(1000, help="Number of train examples."),
    test_samples: int = typer.Option(200, help="Number of test examples."),
    seed: int = typer.Option(13, help="Random seed."),
    base_dir: Path = typer.Option(
        Path("data/raw/temporal_smoke"),
        help="Base directory for generated waveforms.",
    ),
    processed_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directory for generated metadata and JSONL files.",
    ),
) -> None:
    """Generate a train/test temporal smoke-test dataset."""
    random.seed(seed)
    torch.manual_seed(seed)

    train_audio_dir = base_dir / "train"
    test_audio_dir = base_dir / "test"

    train_metadata = processed_dir / "temporal_smoke_train_metadata.json"
    test_metadata = processed_dir / "temporal_smoke_test_metadata.json"
    train_jsonl = processed_dir / "train_temporal_smoke.jsonl"
    test_jsonl = processed_dir / "test_temporal_smoke.jsonl"

    _generate_split(train_samples, train_audio_dir, train_metadata)
    _generate_split(test_samples, test_audio_dir, test_metadata)

    generate_targets(str(train_metadata), str(train_jsonl))
    generate_targets(str(test_metadata), str(test_jsonl))

    typer.echo(f"Wrote train metadata: {train_metadata}")
    typer.echo(f"Wrote test metadata: {test_metadata}")
    typer.echo(f"Wrote train JSONL: {train_jsonl}")
    typer.echo(f"Wrote test JSONL: {test_jsonl}")


if __name__ == "__main__":
    app()
