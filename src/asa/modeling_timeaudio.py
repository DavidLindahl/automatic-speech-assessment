"""TimeAudio adaptations for Qwen2-Audio: absolute-time embedding + time tokens.

This module adds two mechanisms from TimeAudio (arXiv 2511.11039) to a stock
``Qwen2AudioForConditionalGeneration`` so the model can do temporal localization:

1. Absolute time-aware encoding (mechanism 2). A learnable, zero-initialized
   per-frame time embedding is added to the audio features right after the
   multi-modal projector, before they are scattered into the LLM input sequence.
   Qwen2-Audio already carries a *frozen*, Whisper-inherited absolute position
   embedding at the encoder input (``embed_positions``, ``requires_grad=False``,
   added before 32 attention layers). What we add is a *learnable, task-adapted*
   absolute-time signal near the LLM boundary, where it is not diluted by the
   encoder stack. Zero-init means the model starts identical to vanilla and
   learns the time signal gradually, so the flag is a clean on/off ablation.

   Placement note: TimeAudio injects time at the *encoder* output (dim 1280),
   before its Q-Former. Qwen2-Audio has no Q-Former (its projector is a single
   ``nn.Linear``), so we inject post-projector at the LLM hidden dim (4096). A
   post-linear additive embedding is at least as expressive as a pre-linear one;
   this is a deliberate divergence from TimeAudio's placement, not a replication.

2. Anchor/offset time tokens (mechanism 1). Handled by ``temporal_tokens`` and
   ``install_time_tokens`` below: the timestamp vocabulary is extended and the
   new rows are seeded from the model's existing numeral embeddings.

Frame rate is fixed by Qwen2-Audio's audio tower: 16 kHz mel at 100 fps, halved
by ``conv2`` (stride 2) and again by ``avg_pooler`` (stride 2) to 25 fps, i.e.
0.04 s per output frame. The time embedding is indexed by absolute *frame*
position, so index i corresponds to i * 0.04 s.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Qwen2AudioForConditionalGeneration
from transformers.models.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioCausalLMOutputWithPast,
)

from asa import temporal_tokens

# Output frames per second after the audio tower (conv2 stride 2, avg_pooler
# stride 2 over 100 fps mel). 1 / 25 = 0.04 s per frame.
AUDIO_FRAMES_PER_SECOND: int = 25
# Max addressable frames = Qwen2-Audio max_source_positions (1500 = 60 s).
MAX_AUDIO_FRAMES: int = 1500


class Qwen2AudioTimeForConditionalGeneration(Qwen2AudioForConditionalGeneration):
    """Qwen2-Audio with an optional learnable absolute-time frame embedding.

    The behavior is gated by ``config.use_abs_time_embedding``. When false the
    module is created but never added, so the model is bit-for-bit a stock
    Qwen2-Audio. This is what makes the time embedding a clean ablation toggle.
    """

    def __init__(self, config):
        super().__init__(config)
        self.use_abs_time_embedding = bool(
            getattr(config, "use_abs_time_embedding", False)
        )
        hidden_size = config.text_config.hidden_size
        # Zero-init so the model starts identical to vanilla; it learns the time
        # contribution during training. Indexed by absolute output-frame number.
        self.abs_time_embedding = nn.Embedding(MAX_AUDIO_FRAMES, hidden_size)
        nn.init.zeros_(self.abs_time_embedding.weight)
        if not self.use_abs_time_embedding:
            # Keep it inert (and out of the optimizer) when the ablation is off.
            self.abs_time_embedding.weight.requires_grad_(False)

    def _add_time_embedding(
        self,
        audio_features: torch.Tensor,
        audio_output_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Add the per-frame absolute-time embedding to projected audio features.

        Args:
            audio_features: ``(num_audios, max_audio_tokens, hidden)`` projector
                output, one row per audio, right-padded to ``max_audio_tokens``.
            audio_output_lengths: ``(num_audios,)`` valid frame count per audio.

        Returns:
            ``audio_features`` with the time embedding added on valid frames only.
        """
        num_audios, max_tokens, _ = audio_features.shape
        device = audio_features.device

        # audio_features are post-avg-pooler (the encoder applies avg_pooler
        # before returning last_hidden_state), so their length matches
        # audio_output_lengths and never exceeds the encoder's max positions.
        assert max_tokens <= MAX_AUDIO_FRAMES, (
            f"audio frames {max_tokens} exceed MAX_AUDIO_FRAMES {MAX_AUDIO_FRAMES}"
        )

        # Frame index per position: 0, 1, ..., max_tokens-1 (== absolute time at
        # 0.04 s/frame). Clamp to the embedding range as a guard.
        frame_index = torch.arange(max_tokens, device=device).clamp_(
            0, MAX_AUDIO_FRAMES - 1
        )
        time_emb = self.abs_time_embedding(frame_index)  # (max_tokens, hidden)
        time_emb = time_emb.unsqueeze(0).to(audio_features.dtype)

        # Only add on real (non-padding) frames. Padded tail frames are masked
        # out downstream, but masking here keeps the signal clean and avoids
        # nudging positions that will be discarded.
        valid = (
            torch.arange(max_tokens, device=device)[None, :]
            < audio_output_lengths.to(device)[:, None]
        )
        time_emb = time_emb * valid.unsqueeze(-1).to(audio_features.dtype)
        return audio_features + time_emb

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Stock Qwen2-Audio forward, with the time embedding added post-projector.

        This mirrors ``Qwen2AudioForConditionalGeneration.forward`` (transformers
        4.48.x). The only change is the ``_add_time_embedding`` call inserted
        right after ``multi_modal_projector`` and before the audio features are
        merged into ``inputs_embeds``. When ``use_abs_time_embedding`` is false we
        delegate entirely to ``super().forward`` so behavior is unchanged.
        """
        if not self.use_abs_time_embedding:
            return super().forward(
                input_ids=input_ids,
                input_features=input_features,
                attention_mask=attention_mask,
                feature_attention_mask=feature_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        target_device = self.audio_tower.device
        if input_features is not None:
            input_features = input_features.to(target_device)
            feature_attention_mask = feature_attention_mask.to(target_device)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if input_features is not None and input_ids.shape[1] != 1:
                audio_feat_lengths, audio_output_lengths = (
                    self.audio_tower._get_feat_extract_output_lengths(
                        feature_attention_mask.sum(-1)
                    )
                )
                batch_size, _, max_mel_seq_len = input_features.shape
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                seq_range = (
                    torch.arange(
                        0,
                        max_seq_len,
                        dtype=audio_feat_lengths.dtype,
                        device=audio_feat_lengths.device,
                    )
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )
                lengths_expand = audio_feat_lengths.unsqueeze(1).expand(
                    batch_size, max_seq_len
                )
                padding_mask = seq_range >= lengths_expand

                audio_attention_mask_ = padding_mask.view(
                    batch_size, 1, 1, max_seq_len
                ).expand(batch_size, 1, max_seq_len, max_seq_len)
                audio_attention_mask = audio_attention_mask_.to(
                    dtype=self.audio_tower.conv1.weight.dtype,
                    device=self.audio_tower.conv1.weight.device,
                )
                audio_attention_mask[audio_attention_mask_] = float("-inf")

                audio_outputs = self.audio_tower(
                    input_features, attention_mask=audio_attention_mask
                )
                selected_audio_feature = audio_outputs.last_hidden_state
                audio_features = self.multi_modal_projector(selected_audio_feature)

                # ---- TimeAudio mechanism 2: add learnable absolute-time emb ----
                audio_features = self._add_time_embedding(
                    audio_features, audio_output_lengths
                )
                # ----------------------------------------------------------------

                audio_tokens = input_ids == self.config.audio_token_index
                legacy_processing = (
                    audio_tokens[:, :-1] & audio_tokens[:, 1:]
                ).sum() == 0

                if legacy_processing:
                    (
                        inputs_embeds,
                        attention_mask,
                        labels,
                        position_ids,
                        _,
                    ) = self._merge_input_ids_with_audio_features(
                        audio_features,
                        audio_output_lengths,
                        inputs_embeds,
                        input_ids,
                        attention_mask,
                        labels,
                    )
                else:
                    num_audios, max_audio_tokens, embed_dim = audio_features.shape
                    audio_features_mask = torch.arange(
                        max_audio_tokens, device=audio_output_lengths.device
                    )[None, :]
                    audio_features_mask = (
                        audio_features_mask < audio_output_lengths[:, None]
                    )
                    audio_features = audio_features[audio_features_mask]

                    n_audio_tokens = (
                        (input_ids == self.config.audio_token_index).sum().item()
                    )
                    n_audio_features = audio_features.shape[0]

                    if n_audio_tokens != n_audio_features:
                        raise ValueError(
                            "Audio features and audio tokens do not match: "
                            f"tokens: {n_audio_tokens}, features {n_audio_features}"
                        )
                    special_audio_mask = (
                        input_ids == self.config.audio_token_index
                    ).to(inputs_embeds.device)
                    special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(
                        inputs_embeds
                    )
                    audio_features = audio_features.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(
                        special_audio_mask, audio_features
                    )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Loss is computed here, NOT inside language_model, mirroring stock
        # Qwen2AudioForConditionalGeneration.forward verbatim. The stock model
        # filters padding via the attention_mask before CrossEntropy; delegating
        # labels to language_model would instead rely only on -100 masking, a
        # different code path that would break the bit-for-bit ablation claim.
        logits = outputs[0]

        loss = None
        if labels is not None:
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2AudioCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
        )


def install_time_tokens(
    model: Union[Qwen2AudioForConditionalGeneration, nn.Module],
    processor,
    seed_from_numerals: bool = True,
) -> int:
    """Register anchor/offset time tokens and resize + seed the embeddings.

    Adds the time tokens as REGULAR added tokens (not special tokens) so they
    survive ``skip_special_tokens=True`` decoding. Then resizes the model token
    embeddings and, following TimeAudio Eq. 1, seeds the new rows from the
    model's existing numeral embeddings instead of leaving them random:

        anchor ``<aN>`` row  := embedding(numeral token "N")
        offset ``<fK>`` row  := (embedding("K") + embedding(".")) / 2

    Args:
        model: The Qwen2-Audio model (resized in place).
        processor: The Qwen2-Audio processor (its tokenizer is extended in place).
        seed_from_numerals: When false, new rows keep HF's default init.

    Returns:
        Number of tokens actually added (0 if they were already present).
    """
    tokenizer = processor.tokenizer
    tokens = temporal_tokens.all_time_tokens()
    num_added = tokenizer.add_tokens(tokens, special_tokens=False)

    # Resize BEFORE seeding so the new rows exist to write into.
    model.resize_token_embeddings(len(tokenizer))

    if seed_from_numerals and num_added > 0:
        _seed_time_token_embeddings(model, tokenizer)

    return num_added


def _encode_single_token_id(tokenizer, text: str) -> Optional[int]:
    """Return the token id if ``text`` encodes to exactly one token, else None."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


def _seed_time_token_embeddings(model, tokenizer) -> None:
    """Seed anchor/offset rows from numeral embeddings (TimeAudio Eq. 1).

    Both the input embedding and the output projection (``lm_head``) are seeded.
    TimeAudio writes the same numeral-derived vector into ``embed_tokens`` and
    ``lm_head`` for each new token. Qwen2-Audio's LM head is untied
    (``text_config.tie_word_embeddings=False``), so the output rows are a
    separate tensor that must be seeded explicitly; otherwise the model starts
    unable to emit the new tokens with any inductive bias. When the head happens
    to be tied, ``get_output_embeddings`` returns the same tensor and the second
    write is a harmless no-op.
    """
    embed = model.get_input_embeddings()
    weight = embed.weight

    output_embed = model.get_output_embeddings()
    output_weight = (
        output_embed.weight
        if output_embed is not None and output_embed.weight is not weight
        else None
    )

    # Precompute numeral embeddings 0..9 and the decimal point, when each maps to
    # a single token (true for Qwen's BPE digits and ".").
    digit_rows: dict[int, torch.Tensor] = {}
    for digit in range(10):
        tid = _encode_single_token_id(tokenizer, str(digit))
        if tid is not None:
            digit_rows[digit] = weight.data[tid].clone()
    dot_id = _encode_single_token_id(tokenizer, ".")
    dot_row = weight.data[dot_id].clone() if dot_id is not None else None

    def numeral_row(value: int) -> Optional[torch.Tensor]:
        """Embedding for a multi-digit integer = mean of its digit embeddings."""
        rows = [digit_rows[int(ch)] for ch in str(value) if int(ch) in digit_rows]
        if len(rows) != len(str(value)):
            return None
        return torch.stack(rows, dim=0).mean(dim=0)

    def write_seed(tok_id: int, seed: torch.Tensor) -> None:
        """Write the seed vector into both input and output rows for a token."""
        weight.data[tok_id] = seed.to(weight.dtype)
        if output_weight is not None:
            output_weight.data[tok_id] = seed.to(output_weight.dtype)

    with torch.no_grad():
        for second in range(temporal_tokens.MAX_SECONDS):
            tok_id = _encode_single_token_id(
                tokenizer, temporal_tokens.anchor_token(second)
            )
            seed = numeral_row(second)
            if tok_id is not None and seed is not None:
                write_seed(tok_id, seed)

        for tenth in range(temporal_tokens.NUM_OFFSETS):
            tok_id = _encode_single_token_id(
                tokenizer, temporal_tokens.offset_token(tenth)
            )
            digit = digit_rows.get(tenth)
            if tok_id is not None and digit is not None:
                if dot_row is not None:
                    seed = (digit + dot_row) / 2.0
                else:
                    seed = digit
                write_seed(tok_id, seed)
