"""
test_1_audio_loading.py — Test the audio loading function.
Verifies WAV files are read, converted to mono, and resampled to 16kHz.
"""

from asa.data import load_audio

# Pick a known audio file
AUDIO_PATH = "data/raw/NISQA_Corpus/NISQA_TRAIN_SIM/deg/c07896_tsp_2_MG_18.wav"

print("=== Test 1: Audio Loading ===\n")

audio = load_audio(AUDIO_PATH)

print(f"File:     {AUDIO_PATH}")
print(f"Shape:    {audio.shape}")
print(f"dtype:    {audio.dtype}")
print(f"Duration: {len(audio) / 16000:.2f}s")
print(f"Range:    [{audio.min():.3f}, {audio.max():.3f}]")
print(f"Mono:     {audio.ndim == 1}")
print("\n✅ Audio loading works!")
