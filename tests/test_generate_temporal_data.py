import pytest
import torch
from asa.generate_temporal_data import overlay_noise, apply_packet_loss, apply_clipping

def test_overlay_noise_shapes_and_bounds():
    # Create dummy clean speech (5 seconds, 16kHz)
    clean_sr = 16000
    clean_waveform = torch.randn(1, 5 * clean_sr)
    
    # Create dummy noise (2 seconds, 44.1kHz - testing resampling & stereo)
    noise_sr = 44100
    noise_waveform = torch.randn(2, 2 * noise_sr)
    
    mixed, start_time, end_time = overlay_noise(
        clean_waveform, clean_sr, 
        noise_waveform, noise_sr,
        min_duration=1.0, max_duration=1.5,
        target_sr=16000, snr_db=5.0
    )
    
    # Check output shape
    assert mixed.shape == clean_waveform.shape
    
    # Check types
    assert isinstance(start_time, float)
    assert isinstance(end_time, float)
    
    # Check that overlay duration is within bounds
    duration = end_time - start_time
    assert duration >= 1.0 and duration <= 1.5
    
    # Ensure no clipping
    assert torch.max(torch.abs(mixed)) <= 1.0

def test_overlay_noise_edge_cases():
    # Noise shorter than min bounds (0.5s)
    clean_sr = 16000
    clean_waveform = torch.randn(1, 5 * clean_sr)
    noise_sr = 16000
    noise_waveform = torch.randn(1, int(0.5 * noise_sr)) 
    
    mixed, start_time, end_time = overlay_noise(
        clean_waveform, clean_sr,
        noise_waveform, noise_sr,
        min_duration=1.0, max_duration=3.0,
        target_sr=16000
    )
    
    # The function truncates the overlay request to the min of (noise_len, max_dur)
    duration = end_time - start_time
    assert duration == pytest.approx(0.5, rel=1e-3)

def test_apply_packet_loss():
    sr = 16000
    waveform = torch.ones(1, 5 * sr) # All ones
    mixed, start_time, end_time = apply_packet_loss(waveform, sr, min_duration=0.5, max_duration=0.5)
    
    # Check that a segment is indeed muted
    start_idx = int(start_time * sr)
    end_idx = int(end_time * sr)
    assert torch.all(mixed[:, start_idx:end_idx] == 0)
    # Check that outside segments are still ones
    if start_idx > 0:
        assert torch.all(mixed[:, 0:start_idx] == 1)
    if end_idx < waveform.shape[1]:
        assert torch.all(mixed[:, end_idx:] == 1)

def test_apply_clipping():
    sr = 16000
    waveform = torch.randn(1, 5 * sr) * 2.0 # High amplitude
    threshold = 0.1
    mixed, start_time, end_time = apply_clipping(waveform, sr, min_duration=0.5, max_duration=0.5, threshold=threshold)
    
    start_idx = int(start_time * sr)
    end_idx = int(end_time * sr)
    # Check segment is clipped
    assert torch.max(torch.abs(mixed[:, start_idx:end_idx])) <= threshold + 1e-6
