# demo_run.py
import os
import pathlib
import time

import torch
import torchaudio
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


# ---- Config ----
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
AUDIO_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "demo.wav"
)
MAX_AUDIO_SECONDS = 5
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "You are a professional audio engineer and speech quality assessor. "
    "You evaluate speech recordings using the ITU-R BS.2399-0 Sound Wheel "
    "framework, covering six perceptual dimensions: Loudness, Dynamics, "
    "Timbre, Spatial, Transparency, and Artefacts. You always use precise "
    "Sound Wheel descriptors in your assessments."
)

USER_PROMPT = """\
Analyze the speech audio above using the ITU-R BS.2399-0 Sound Wheel framework.

Use the following Sound Wheel descriptors where relevant:
- LOUDNESS: loudness
- DYNAMICS: attack, bass precision, punch, powerful
- TIMBRE:
  - Treble: treble strength, brilliance, tinny
  - Midrange: midrange strength, nasal, canny
  - Bass: bass strength, bass depth, boomy
  - Colouration: boxy, timbral balance, full, homogeneous
- SPATIAL:
  - Spatial extent: depth, width, envelopment
  - Localization: spatial balance, distance, externalisation, localisability
  - Environment: reverbrance, clarity
- TRANSPARENCY: presence, clean, detail, natural
- ARTEFACTS:
  - Signal related: shrill, rubbing, rough, buzzing, clipping, distortion, compression
  - Noise: hissing, humming, bubbling, fluctuating

Respond with EXACTLY:
1. A concise 1-2 sentence expert assessment using the Sound Wheel descriptors listed above.
2. Quality label: Excellent / Good / Fair / Poor / Bad"""
# ----------------


def load_audio(
    filepath: pathlib.Path, target_sr: int, max_seconds: float
) -> torch.Tensor:
    """Load audio, resample to target_sr, mono, trim to max_seconds. Returns 1D float tensor."""
    waveform, sr = torchaudio.load(str(filepath))  # (channels, samples)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(
            waveform
        )
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    max_samples = int(max_seconds * target_sr)
    waveform = waveform[:, :max_samples]
    return waveform.squeeze(0).float()


def main():
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Missing {AUDIO_PATH.resolve()}.\n"
            f"Put a wav named 'demo.wav' in the repo root (same folder as demo_run.py)."
        )

    # Put HF cache somewhere safe (optional but recommended on HPC)
    os.environ.setdefault("HF_HOME", str(pathlib.Path.home() / "work" / "hf_cache"))

    print(f"PyTorch: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID, fix_mistral_regex=True)

    # CPU-friendly default; if you get OOM on CPU, request more RAM OR use GPU queue and float16.
    if device == "cuda":
        dtype = torch.float16
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        dtype = torch.float32
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.to("cpu")

    model.eval()
    print(f"Model dtype: {dtype}")

    target_sr = processor.feature_extractor.sampling_rate
    audio_tensor = load_audio(
        AUDIO_PATH, target_sr=target_sr, max_seconds=MAX_AUDIO_SECONDS
    )
    duration_s = audio_tensor.numel() / target_sr
    print(f"Loaded audio: {AUDIO_PATH} ({duration_s:.2f}s @ {target_sr} Hz)")

    # Build conversation (important: pass raw audio array, not audio_url path)
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_tensor.numpy()},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    # Debug: show the rendered prompt so we can verify system prompt is included
    print("\n--- Rendered chat template ---")
    print(text)
    print("--- End template ---\n")

    # NOTE: processor expects "audio=" for Qwen2-Audio in many setups; keep it as list with one element
    inputs = processor(
        text=text, audio=[audio_tensor.numpy()], return_tensors="pt", padding=True
    )

    # Move tensors to device (works for CPU or CUDA)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    elapsed = time.time() - t0

    # Strip prompt tokens
    out_ids = out_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n" + "=" * 80)
    print("Qwen2-Audio assessment")
    print("=" * 80)
    print(response)
    print("\n" + "=" * 80)
    print(f"Generated in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
