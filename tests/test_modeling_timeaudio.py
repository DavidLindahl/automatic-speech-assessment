"""CPU-safe structural tests for the TimeAudio Qwen2-Audio subclass.

These build a tiny random-config model (no 7B download) and exercise:
- zero-init of the absolute-time embedding (off = identical to vanilla),
- the requires_grad gating that makes the embedding a clean ablation toggle,
- the per-frame additive time-embedding shape/masking logic,
- a full forward pass with the embedding on (loss is finite).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from transformers import (  # noqa: E402
    Qwen2AudioConfig,
    Qwen2AudioEncoderConfig,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config  # noqa: E402

from asa.modeling_timeaudio import (  # noqa: E402
    MAX_AUDIO_FRAMES,
    Qwen2AudioTimeForConditionalGeneration,
)


def _tiny_config(use_abs_time_embedding: bool) -> Qwen2AudioConfig:
    """Build a tiny Qwen2-Audio config for fast CPU instantiation."""
    audio_config = Qwen2AudioEncoderConfig(
        d_model=32,
        encoder_attention_heads=2,
        encoder_ffn_dim=64,
        encoder_layers=2,
        num_mel_bins=80,
        max_source_positions=64,
        scale_embedding=False,
    )
    text_config = Qwen2Config(
        hidden_size=48,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=1024,
        max_position_embeddings=512,
    )
    config = Qwen2AudioConfig(
        audio_config=audio_config,
        text_config=text_config,
        audio_token_index=10,
    )
    config.use_abs_time_embedding = use_abs_time_embedding
    return config


def test_abs_time_embedding_zero_initialized():
    model = Qwen2AudioTimeForConditionalGeneration(_tiny_config(True))
    assert model.abs_time_embedding.weight.shape[0] == MAX_AUDIO_FRAMES
    assert torch.count_nonzero(model.abs_time_embedding.weight) == 0


def test_flag_off_freezes_embedding():
    off = Qwen2AudioTimeForConditionalGeneration(_tiny_config(False))
    assert off.use_abs_time_embedding is False
    assert off.abs_time_embedding.weight.requires_grad is False

    on = Qwen2AudioTimeForConditionalGeneration(_tiny_config(True))
    assert on.use_abs_time_embedding is True
    assert on.abs_time_embedding.weight.requires_grad is True


def _tiny_batch(config):
    """Build a minimal valid forward batch for the tiny model.

    The audio tower requires the mel length to equal
    max_source_positions * conv1.stride * conv2.stride (= 128 here). We feed a
    full-length mel and a matching feature attention mask.
    """
    ac = config.audio_config
    mel_len = ac.max_source_positions * 1 * 2  # conv1 stride 1, conv2 stride 2
    input_features = torch.randn(1, ac.num_mel_bins, mel_len)
    feature_attention_mask = torch.ones(1, mel_len, dtype=torch.long)
    return input_features, feature_attention_mask


def test_forward_loss_finite_with_time_embedding():
    """Full forward with the embedding ON returns a finite loss.

    Exercises the fixed forward tail (inline loss + Qwen2AudioCausalLMOutput).
    """
    config = _tiny_config(True)
    model = Qwen2AudioTimeForConditionalGeneration(config)
    model.eval()

    input_features, feature_attention_mask = _tiny_batch(config)
    # Determine how many audio frames the tower will emit, then build a matching
    # token sequence with that many audio-placeholder tokens.
    _, output_lengths = model.audio_tower._get_feat_extract_output_lengths(
        feature_attention_mask.sum(-1)
    )
    n_audio = int(output_lengths[0].item())
    audio_id = config.audio_token_index
    text_ids = [5, 6, 7]
    input_ids = torch.tensor([[audio_id] * n_audio + text_ids])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[input_ids == audio_id] = -100

    out = model(
        input_ids=input_ids,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        attention_mask=attention_mask,
        labels=labels,
    )
    assert out.loss is not None
    assert torch.isfinite(out.loss).item()


def test_generate_smoke_with_time_embedding():
    """model.generate() survives prefill + a couple decode steps, flag ON.

    This is the eval path (run_inference -> generate), which hits forward()
    differently than training. Garbage weights are fine; we only assert the call
    mechanics (signature, time-embedding insertion on prefill, KV cache) work.
    """
    config = _tiny_config(True)
    model = Qwen2AudioTimeForConditionalGeneration(config)
    model.eval()

    input_features, feature_attention_mask = _tiny_batch(config)
    _, output_lengths = model.audio_tower._get_feat_extract_output_lengths(
        feature_attention_mask.sum(-1)
    )
    n_audio = int(output_lengths[0].item())
    audio_id = config.audio_token_index
    input_ids = torch.tensor([[audio_id] * n_audio + [5, 6]])
    attention_mask = torch.ones_like(input_ids)

    out = model.generate(
        input_ids=input_ids,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        attention_mask=attention_mask,
        max_new_tokens=3,
        do_sample=False,
    )
    # Generated at least the prompt back plus new tokens.
    assert out.shape[1] >= input_ids.shape[1]


def test_add_time_embedding_shapes_and_masking():
    model = Qwen2AudioTimeForConditionalGeneration(_tiny_config(True))
    hidden = model.config.text_config.hidden_size
    num_audios, max_tokens = 2, 5
    feats = torch.zeros(num_audios, max_tokens, hidden)
    lengths = torch.tensor([5, 3])

    # With zero-init weights, adding the embedding changes nothing yet.
    out_zero = model._add_time_embedding(feats, lengths)
    assert out_zero.shape == feats.shape
    assert torch.allclose(out_zero, feats)

    # Give the embedding a non-zero signal and confirm only valid frames move.
    with torch.no_grad():
        model.abs_time_embedding.weight.fill_(1.0)
    out = model._add_time_embedding(feats, lengths)
    # Row 0 has length 5: all frames get +1.
    assert torch.allclose(out[0], torch.ones(max_tokens, hidden))
    # Row 1 has length 3: frames 0-2 get +1, frames 3-4 stay 0 (padding).
    assert torch.allclose(out[1, :3], torch.ones(3, hidden))
    assert torch.allclose(out[1, 3:], torch.zeros(max_tokens - 3, hidden))
