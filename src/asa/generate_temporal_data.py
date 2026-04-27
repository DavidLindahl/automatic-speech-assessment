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
    ds = load_dataset("librispeech_asr", "clean", split=split, streaming=streaming, trust_remote_code=True)
    return iter(ds)

def load_noise_dataset(split="train", streaming=True):
    """Loads ESC-50 noise dataset using HuggingFace datasets."""
    print("Loading ESC-50...")
    ds = load_dataset("ashraq/esc50", split=split, streaming=streaming, trust_remote_code=True)
    return iter(ds)

def overlay_noise(clean_waveform, clean_sr, noise_waveform, noise_sr, min_duration=1.0, max_duration=3.0, target_sr=16000, snr_db=5.0):
    """Overlays noise onto clean speech at random timestamps."""
    # Resample to target_sr
    if clean_sr != target_sr:
        clean_waveform = torchaudio.functional.resample(clean_waveform, clean_sr, target_sr)
    if noise_sr != target_sr:
        noise_waveform = torchaudio.functional.resample(noise_waveform, noise_sr, target_sr)
        
    # Ensure Mono
    if clean_waveform.shape[0] > 1:
        clean_waveform = clean_waveform.mean(dim=0, keepdim=True)
    if noise_waveform.shape[0] > 1:
        noise_waveform = noise_waveform.mean(dim=0, keepdim=True)
        
    clean_len = clean_waveform.shape[1]
    noise_len = noise_waveform.shape[1]
    
    clean_duration = clean_len / target_sr
    
    # We want a noise duration of min_duration to max_duration
    overlay_duration = random.uniform(min_duration, max_duration)
    overlay_samples = int(overlay_duration * target_sr)
    
    # Fix overlay duration if it's too long
    overlay_samples = min(overlay_samples, clean_len, noise_len)
    
    if overlay_samples <= 0:
        return clean_waveform, 0.0, 0.0
    
    # Pick a random start time in the clean audio
    start_idx_clean = random.randint(0, clean_len - overlay_samples)
    start_time = start_idx_clean / target_sr
    
    # Pick a random start time in the noise audio
    start_idx_noise = random.randint(0, noise_len - overlay_samples)
    
    noise_segment = noise_waveform[:, start_idx_noise:start_idx_noise + overlay_samples]
    
    # Calculate energy and scale based on target SNR
    clean_energy = torch.mean(clean_waveform[:, start_idx_clean:start_idx_clean + overlay_samples] ** 2)
    noise_energy = torch.mean(noise_segment ** 2)
    
    if noise_energy > 0 and clean_energy > 0:
        target_noise_energy = clean_energy / (10 ** (snr_db / 10))
        scale = torch.sqrt(target_noise_energy / noise_energy)
        noise_segment = noise_segment * scale

    # Mix
    mixed_waveform = clean_waveform.clone()
    mixed_waveform[:, start_idx_clean:start_idx_clean + overlay_samples] += noise_segment
    
    # Normalize to avoid clipping
    max_val = torch.max(torch.abs(mixed_waveform))
    if max_val > 1.0:
        mixed_waveform = mixed_waveform / max_val
        
    end_time = start_time + (overlay_samples / target_sr)
    
    return mixed_waveform, start_time, end_time

def apply_packet_loss(waveform, sr, min_duration=0.1, max_duration=0.5):
    """Mutes a random segment of the audio to simulate packet loss."""
    clean_len = waveform.shape[1]
    duration = random.uniform(min_duration, max_duration)
    num_samples = min(int(duration * sr), clean_len)

    if num_samples <= 0:
        return waveform, 0.0, 0.0

    start_idx = random.randint(0, clean_len - num_samples)
    start_time = start_idx / sr
    end_time = (start_idx + num_samples) / sr

    degraded_waveform = waveform.clone()
    degraded_waveform[:, start_idx : start_idx + num_samples] = 0.0

    return degraded_waveform, start_time, end_time


def apply_clipping(waveform, sr, min_duration=0.5, max_duration=2.0, threshold=0.1):
    """Artificially clips a random segment of the audio."""
    clean_len = waveform.shape[1]
    duration = random.uniform(min_duration, max_duration)
    num_samples = min(int(duration * sr), clean_len)

    if num_samples <= 0:
        return waveform, 0.0, 0.0

    start_idx = random.randint(0, clean_len - num_samples)
    start_time = start_idx / sr
    end_time = (start_idx + num_samples) / sr

    degraded_waveform = waveform.clone()
    segment = degraded_waveform[:, start_idx : start_idx + num_samples]

    # Apply hard clipping
    segment = torch.clamp(segment, -threshold, threshold)
    # Scale back up slightly to emphasize the distortion if needed, or leave as is for "flat" clipping
    degraded_waveform[:, start_idx : start_idx + num_samples] = segment

    return degraded_waveform, start_time, end_time


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
                clean_tensor, sr, noise_tensor, noise_audio["sampling_rate"], target_sr=sr
            )
            degradation_type = noise_type
        elif deg_choice == "packet_loss":
            degraded_waveform, start_time, end_time = apply_packet_loss(clean_tensor, sr)
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
