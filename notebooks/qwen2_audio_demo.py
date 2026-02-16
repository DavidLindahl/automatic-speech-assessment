import pathlib
import time

import pandas as pd
import torch
import torchaudio
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Paths
PROJECT_ROOT = pathlib.Path().resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "NISQA_Corpus" / "NISQA_TRAIN_SIM"
DEG_DIR = DATA_DIR / "deg"
CSV_PATH = DATA_DIR / "NISQA_TRAIN_SIM_file.csv"

assert CSV_PATH.exists(), f"CSV not found at {CSV_PATH}"
assert DEG_DIR.exists(), f"Audio directory not found at {DEG_DIR}"
print(f"Data directory: {DATA_DIR}")

df = pd.read_csv(CSV_PATH)
print(f"Total files: {len(df)}")
print(f"MOS range: {df['mos'].min():.2f} – {df['mos'].max():.2f}")

# Sample 2 files from each MOS bracket to get a diverse spread
mos_brackets = [(1.0, 2.0), (2.0, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 5.01)]
sampled = []
for low, high in mos_brackets:
    bracket = df[(df["mos"] >= low) & (df["mos"] < high)]
    n = min(2, len(bracket))
    sampled.append(bracket.sample(n, random_state=42))

sample_df = pd.concat(sampled).reset_index(drop=True)
print(f"\nSampled {len(sample_df)} files:")
sample_df[["filename_deg", "mos", "noi", "col", "dis", "loud"]]

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)

# Load on CPU — avoids the disk-offloading that makes generation extremely slow.
# float32 on CPU is actually faster than float16 on MPS when disk-offloading kicks in.
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model.eval()
print(f"Model loaded on: {model.device}")
print(f"Model dtype: {model.dtype}")

SYSTEM_PROMPT = """You are an expert audio quality assessor. Evaluate the speech audio using the \
ITU-R BS.2399-0 Sound Wheel framework. Consider ALL of the following perceptual dimensions:

1. LOUDNESS: perceived loudness (soft ↔ loud)
2. DYNAMICS: dynamic range (narrow ↔ wide), punch (weak ↔ strong), attack (slow ↔ fast), decay (slow ↔ fast)
3. TIMBRE: brightness (dark ↔ bright), warmth (cold ↔ warm), harshness (smooth ↔ harsh), \
sibilance (dull ↔ sibilant), fullness (thin ↔ full), boominess (tight ↔ boomy), \
boxiness (open ↔ boxy), presence (distant ↔ present), clarity (muddy ↔ clear), \
naturalness (unnatural ↔ natural), metallic, hollow, roughness, smoothness
4. SPATIAL: width (narrow ↔ wide), depth (shallow ↔ deep), localization (imprecise ↔ precise), \
envelopment (dry ↔ enveloping), distance (far ↔ close), stability (unstable ↔ stable)
5. TRANSPARENCY: openness (closed ↔ open), veiled (transparent ↔ veiled), diffusion (focused ↔ diffuse)
6. ARTEFACTS: noise (quiet ↔ noisy), distortion (clean ↔ distorted), clicking, hissing, buzzing, \
humming, rattling, interference, echo, reverberation, flutter, warble, gating, pumping, breathing

Provide your assessment as:
1. A short paragraph (3-5 sentences) describing the speech quality using the Sound Wheel descriptors above.
2. An overall quality label: Excellent / Good / Fair / Poor / Bad
"""

USER_PROMPT = "Listen to this speech audio and assess its quality using the Sound Wheel framework."

print("System prompt defined.")
print(f"Prompt length: {len(SYSTEM_PROMPT)} chars")

TARGET_SR = processor.feature_extractor.sampling_rate
MAX_AUDIO_SECONDS = 5  # Trim audio to reduce token count and speed up inference
MAX_NEW_TOKENS = 150  # Keep generation short

print(f"Target sample rate: {TARGET_SR} Hz")
print(f"Max audio length: {MAX_AUDIO_SECONDS}s")
print(f"Max new tokens: {MAX_NEW_TOKENS}")


def load_audio(filepath: pathlib.Path, target_sr: int, max_seconds: float):
    """Load a WAV file, resample, convert to mono, and trim to max_seconds."""
    waveform, sr = torchaudio.load(str(filepath))
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Trim to max_seconds
    max_samples = int(max_seconds * target_sr)
    waveform = waveform[:, :max_samples]
    return waveform.squeeze(0).numpy()


results = []

for idx, row in sample_df.iterrows():
    filename = row["filename_deg"]
    filepath = DEG_DIR / filename
    print(f"\n[{idx+1}/{len(sample_df)}] Processing: {filename} (MOS={row['mos']:.2f})")

    # Load audio (trimmed)
    audio_array = load_audio(filepath, TARGET_SR, MAX_AUDIO_SECONDS)
    duration = len(audio_array) / TARGET_SR
    print(f"  Audio loaded: {duration:.1f}s, {len(audio_array)} samples")

    # Build conversation
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(filepath)},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    # Tokenize — note: kwarg is 'audio' (not 'audios') in transformers >=5.x
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=text, audio=[audio_array], return_tensors="pt", padding=True
    )
    # Move inputs to same device as model
    inputs = inputs.to(model.device)

    # Generate
    t0 = time.time()
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    elapsed = time.time() - t0
    generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"  Generated in {elapsed:.1f}s")
    print(f"  Response: {response[:120]}...")
    results.append(
        {
            "filename": filename,
            "mos": row["mos"],
            "noi": row["noi"],
            "col": row["col"],
            "dis": row["dis"],
            "loud": row["loud"],
            "qwen2_description": response,
            "inference_time_s": round(elapsed, 1),
        }
    )

print("\n✅ Inference complete!")

results_df = pd.DataFrame(results)

# Sort by MOS score for readability
results_df = results_df.sort_values("mos").reset_index(drop=True)

# Display with full text
pd.set_option("display.max_colwidth", None)
results_df

print("=" * 80)
print("BRIEF ANALYSIS: Qwen2-Audio Descriptions vs Ground-Truth MOS")
print("=" * 80)

for _, row in results_df.iterrows():
    print(f"\n{'─' * 80}")
    print(f"File: {row['filename']}")
    print(
        f"Ground-truth MOS: {row['mos']:.2f}  |  NOI: {row['noi']:.2f}  |  "
        f"COL: {row['col']:.2f}  |  DIS: {row['dis']:.2f}  |  LOUD: {row['loud']:.2f}"
    )
    print(f"Inference time: {row['inference_time_s']}s")
    print("\nQwen2-Audio assessment:")
    print(f"  {row['qwen2_description']}")

print(f"\n{'═' * 80}")
print("Observation: Compare whether low-MOS files receive descriptions mentioning")
print("artefacts, noise, distortion, etc., while high-MOS files are described as")
print("natural, clear, and transparent.")
print(f"{'═' * 80}")
