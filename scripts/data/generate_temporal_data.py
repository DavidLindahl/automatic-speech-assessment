import os
import json
import random
import torch
import torchaudio
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def load_clean_speech_dataset(split="validation", streaming=True):
    """Loads LibriSpeech validation set using HuggingFace datasets."""
    print("Loading LibriSpeech...")
    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )
    return iter(ds)


def load_noise_dataset(split="train", streaming=True):
    """Loads ESC-50 noise dataset using HuggingFace datasets."""
    print("Loading ESC-50...")
    ds = load_dataset(
        "ashraq/esc50", split=split, streaming=streaming, trust_remote_code=True
    )
    return iter(ds)


from asa.generate_temporal_data import overlay_noise, apply_packet_loss, apply_clipping


def main(
    num_samples: int = 1000,
    output_dir: str = "data/raw/temporal_mixes",
    metadata_path: str = "data/processed/temporal_metadata_raw.json",
):
    """Generates synthetic data with random temporal degradations for Temporal-ALLD."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

    clean_iter = load_clean_speech_dataset()
    noise_iter = load_noise_dataset()

    metadata = []

    print(f"Generating {num_samples} temporal mixes...")
    for i in tqdm(range(num_samples)):
        try:
            clean_sample = next(clean_iter)
        except StopIteration:
            clean_iter = load_clean_speech_dataset()
            clean_sample = next(clean_iter)

        clean_audio = clean_sample["audio"]
        clean_tensor = torch.tensor(clean_audio["array"]).view(1, -1).float()
        sr = 16000

        # Resample clean to 16k immediately
        if clean_audio["sampling_rate"] != sr:
            clean_tensor = torchaudio.functional.resample(
                clean_tensor, clean_audio["sampling_rate"], sr
            )

        # Randomly choose degradation
        deg_choice = random.choice(["noise", "packet_loss", "clipping"])

        if deg_choice == "noise":
            try:
                noise_sample = next(noise_iter)
            except StopIteration:
                noise_iter = load_noise_dataset()
                noise_sample = next(noise_iter)
            noise_audio = noise_sample["audio"]
            noise_type = noise_sample.get("category", "noise").replace("_", " ")
            noise_tensor = torch.tensor(noise_audio["array"]).view(1, -1).float()

            degraded_waveform, start_time, end_time = overlay_noise(
                clean_tensor,
                sr,
                noise_tensor,
                noise_audio["sampling_rate"],
                target_sr=sr,
            )
            degradation_type = noise_type
        elif deg_choice == "packet_loss":
            degraded_waveform, start_time, end_time = apply_packet_loss(
                clean_tensor, sr
            )
            degradation_type = "packet loss"
        else:  # clipping
            degraded_waveform, start_time, end_time = apply_clipping(
                clean_tensor, sr, threshold=random.uniform(0.05, 0.15)
            )
            degradation_type = "clipping distortion"

        filename = f"temporal_mix_{i:04d}.wav"
        file_path = os.path.join(output_dir, filename)

        # Save output
        torchaudio.save(file_path, degraded_waveform, sr)

        metadata.append(
            {
                "audio_path": file_path,
                "degradation_type": degradation_type,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
            }
        )

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    import typer

    typer.run(main)
