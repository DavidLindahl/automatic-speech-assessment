"""Timestamp-weighted, distance-aware token loss for temporal SFT (Phase 1).

Why this exists: the 2026-06-09 temporal SFT arms collapsed to constant
intervals below the audio-blind baselines, and the frame probe (28627256)
proved the degradation location is linearly readable from the model's own
audio features. The bottleneck is the objective: uniform cross-entropy gives
timestamps ~8% of the loss mass and prices a 0.1 s miss exactly like a 50 s
miss, so the loss-optimal behavior under a hard perception task is to recite
the training-interval prior. This module changes both properties at the
timestamp positions only:

1. Weighting (lambda): the loss term of every position whose label is a time
   token is multiplied by ``time_weight``, shifting loss mass toward the
   tokens that carry the task.
2. Distance-aware soft targets (sigma): for anchor ``<aN>`` and offset
   ``<fK>`` labels the one-hot target is replaced by a Gaussian over the
   ordered token group (anchors 0..59, offsets 0..9), normalized within the
   group. The cross-entropy is still taken against the full-vocabulary
   log-softmax, so the model is also pushed to put mass on the group at all;
   the Gaussian only redistributes the target inside the group. Near misses
   now receive partial credit, which gives gradient descent a direction along
   the time axis (Momentor-style neighboring-token propagation).

Caption and MOS positions keep plain cross-entropy with weight 1, so the
joint target's other two outputs are supervised exactly as before. The total
is divided by the number of supervised tokens (not the weighted count), so
caption gradients keep their vanilla scale and time-token gradients are
scaled up by exactly ``time_weight``.

Only meaningful for the anchor/offset target format: free-text digit
timestamps have no ordered single-token vocabulary to smooth over.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from asa import temporal_tokens

IGNORE_INDEX: int = -100


def time_token_id_groups(tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ordered anchor and offset token-id tensors from an extended tokenizer.

    Args:
        tokenizer: Tokenizer that already carries the installed time tokens
            (see ``asa.modeling_timeaudio.install_time_tokens``).

    Returns:
        ``(anchor_ids, offset_ids)`` where ``anchor_ids[j]`` is the id of
        ``<aj>`` and ``offset_ids[k]`` the id of ``<fk>``.

    Raises:
        ValueError: When any time token is missing or not a single token,
            which means the tokenizer was not extended.
    """
    anchor_ids = []
    for second in range(temporal_tokens.MAX_SECONDS):
        ids = tokenizer.encode(
            temporal_tokens.anchor_token(second), add_special_tokens=False
        )
        if len(ids) != 1:
            raise ValueError(
                f"anchor token {temporal_tokens.anchor_token(second)} is not a "
                "single token; install_time_tokens must run before building "
                "the temporal loss"
            )
        anchor_ids.append(ids[0])
    offset_ids = []
    for tenth in range(temporal_tokens.NUM_OFFSETS):
        ids = tokenizer.encode(
            temporal_tokens.offset_token(tenth), add_special_tokens=False
        )
        if len(ids) != 1:
            raise ValueError(
                f"offset token {temporal_tokens.offset_token(tenth)} is not a "
                "single token; install_time_tokens must run before building "
                "the temporal loss"
            )
        offset_ids.append(ids[0])
    return torch.tensor(anchor_ids), torch.tensor(offset_ids)


def gaussian_group_targets(
    label_positions: torch.Tensor,
    group_size: int,
    sigma: float,
) -> torch.Tensor:
    """Gaussian soft-target distributions over an ordered token group.

    Args:
        label_positions: ``(N,)`` integer index of the true token within the
            group (e.g. 5 for ``<a5>``).
        group_size: Number of tokens in the ordered group.
        sigma: Gaussian width in group buckets. ``sigma <= 0`` returns one-hot.

    Returns:
        ``(N, group_size)`` rows summing to 1, peaked at the true index.
    """
    positions = torch.arange(
        group_size, dtype=torch.float32, device=label_positions.device
    ).unsqueeze(0)
    centers = label_positions.float().unsqueeze(1)
    if sigma <= 0:
        return (positions == centers).float()
    weights = torch.exp(-((positions - centers) ** 2) / (2.0 * sigma**2))
    return weights / weights.sum(dim=1, keepdim=True)


def _group_soft_loss(
    log_probs: torch.Tensor,
    flat_labels: torch.Tensor,
    group_ids: torch.Tensor,
    sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Soft cross-entropy at positions whose label is in an ordered group.

    Args:
        log_probs: ``(N, vocab)`` full-vocabulary log-softmax rows.
        flat_labels: ``(N,)`` label ids.
        group_ids: ``(G,)`` ordered token ids of the group.
        sigma: Gaussian width in buckets.

    Returns:
        ``(positions_mask, losses)``: a boolean ``(N,)`` mask of rows whose
        label belongs to the group, and the ``(M,)`` soft-CE losses for them.
    """
    group_ids = group_ids.to(flat_labels.device)
    mask = torch.isin(flat_labels, group_ids)
    if not bool(mask.any()):
        return mask, log_probs.new_zeros(0)
    # Map each in-group label id to its index within the ordered group.
    labels_in = flat_labels[mask]
    index_in_group = (labels_in.unsqueeze(1) == group_ids.unsqueeze(0)).float()
    label_idx = index_in_group.argmax(dim=1)
    targets = gaussian_group_targets(label_idx, len(group_ids), sigma).to(
        log_probs.device, log_probs.dtype
    )
    group_log_probs = log_probs[mask][:, group_ids]
    losses = -(targets * group_log_probs).sum(dim=1)
    return mask, losses


def temporal_weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    anchor_ids: torch.Tensor,
    offset_ids: torch.Tensor,
    time_weight: float = 1.0,
    soft_sigma: float = 0.0,
) -> torch.Tensor:
    """Causal-LM cross-entropy with weighted, distance-aware time tokens.

    Identical to mean next-token cross-entropy when ``time_weight == 1`` and
    ``soft_sigma == 0``. The total is normalized by the number of supervised
    tokens, so non-time gradients keep their vanilla scale regardless of the
    weight applied at time-token positions.

    Args:
        logits: ``(B, T, vocab)`` model logits, unshifted.
        labels: ``(B, T)`` labels with ``IGNORE_INDEX`` masking.
        anchor_ids: Ordered anchor token ids (``<a0>``..).
        offset_ids: Ordered offset token ids (``<f0>``..).
        time_weight: Multiplier on the loss of time-token positions.
        soft_sigma: Gaussian width (in buckets) for soft targets over the
            anchor and offset groups; 0 keeps one-hot targets.

    Returns:
        Scalar loss.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != IGNORE_INDEX
    flat_logits = shift_logits[mask].float()
    flat_labels = shift_labels[mask]
    if flat_labels.numel() == 0:
        return logits.sum() * 0.0

    log_probs = F.log_softmax(flat_logits, dim=-1)
    per_token = -log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)

    anchor_mask, anchor_losses = _group_soft_loss(
        log_probs, flat_labels, anchor_ids, soft_sigma
    )
    offset_mask, offset_losses = _group_soft_loss(
        log_probs, flat_labels, offset_ids, soft_sigma
    )

    weighted = per_token.clone()
    if soft_sigma > 0:
        weighted[anchor_mask] = anchor_losses
        weighted[offset_mask] = offset_losses
    time_mask = anchor_mask | offset_mask
    weighted = torch.where(time_mask, weighted * time_weight, weighted)

    return weighted.sum() / mask.sum()


class TemporalLossConfig:
    """Bundles the precomputed tensors and knobs for the custom trainer.

    Attributes:
        anchor_ids: Ordered anchor token ids.
        offset_ids: Ordered offset token ids.
        time_weight: Loss multiplier at time-token positions.
        soft_sigma: Gaussian soft-target width in buckets.
    """

    def __init__(
        self,
        anchor_ids: torch.Tensor,
        offset_ids: torch.Tensor,
        time_weight: float,
        soft_sigma: float,
    ) -> None:
        self.anchor_ids = anchor_ids
        self.offset_ids = offset_ids
        self.time_weight = time_weight
        self.soft_sigma = soft_sigma

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        time_weight: float,
        soft_sigma: float,
    ) -> "TemporalLossConfig":
        """Build the config by resolving time-token ids from the tokenizer."""
        anchor_ids, offset_ids = time_token_id_groups(tokenizer)
        return cls(anchor_ids, offset_ids, time_weight, soft_sigma)

    @property
    def active(self) -> bool:
        """Whether the custom loss differs from plain cross-entropy."""
        return self.time_weight != 1.0 or self.soft_sigma > 0.0


def compute_temporal_loss(
    model,
    inputs: dict,
    config: TemporalLossConfig,
    return_outputs: bool = False,
):
    """Drop-in ``Trainer.compute_loss`` body using the temporal loss.

    Pops the labels so the model forward skips its internal loss, then
    computes the weighted, distance-aware cross-entropy from the logits.
    Label-to-logit alignment holds because the processor expands the audio
    placeholder at tokenization time (the non-legacy path), so the sequence
    length is unchanged inside the model; the same alignment assumption is
    already load-bearing in the ALLD DPO trainer.

    Args:
        model: The Qwen2-Audio model.
        inputs: Collated batch including ``labels``.
        config: Temporal loss configuration.
        return_outputs: Mirror of the Trainer flag.

    Returns:
        The loss, or ``(loss, outputs)`` when ``return_outputs`` is set.
    """
    inputs = dict(inputs)
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    loss = temporal_weighted_ce(
        outputs.logits,
        labels,
        anchor_ids=config.anchor_ids,
        offset_ids=config.offset_ids,
        time_weight=config.time_weight,
        soft_sigma=config.soft_sigma,
    )
    if return_outputs:
        return loss, outputs
    return loss
