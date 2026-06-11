"""Frozen-feature frame probe for temporal localization (the Phase-0 gate).

Answers one question: is the degradation location linearly readable from the
audio features Qwen2-Audio hands to its LLM? The 2026-06-09 temporal SFT arms
collapsed to constant intervals below the audio-blind baselines, which is
consistent with two very different root causes:

1. The information reaches the LLM boundary but the cross-entropy objective
   never prices temporal distance, so the model settles for the interval
   prior (an objective problem; loss-level fixes can work).
2. The encoder features do not carry the degradation location at all (a
   representation problem; no text-side loss can fix it).

This script decides between them. It runs the FROZEN audio tower + projector
over training mixes, taps per-frame features at two points (encoder output,
1280-d, and post-projector, 4096-d, the exact tensor the LLM reads), and fits
a linear probe per tap: is this frame inside the degraded window? Ground
truth comes from the mix construction. It reports the validation frame AUC,
and then recovers one interval per clip from the probe scores (smooth,
threshold, longest run) and scores it with temporal IoU against the same
audio-blind baselines the eval uses.

Reading the result:
- Probe-recovered mean t-IoU well above the best-constant baseline: cause 1.
  Proceed with loss-level fixes; the probe number is the ceiling a perfect
  text-side readout could approach.
- Probe near or below the baselines: cause 2. Pivot to longer degradation
  windows, coarser targets, or encoder-side supervision.

The TimeAudio absolute-time embedding is deliberately NOT added to the
projector tap: it is a function of frame position only, identical for every
clip, so including it would let the probe exploit positional priors instead
of acoustics. The probe is per-frame and position-blind by design.

The train/val split is grouped by source reference file (``filename_ref``)
so the placement-augmented reuses of one REF never straddle the split.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import typer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from asa.audio import load_audio
from asa.processed_data import load_processed_records, resolve_audio_path

# Qwen2-Audio audio tower output rate: 16 kHz mel at 100 fps, conv2 stride 2,
# avg_pooler stride 2 -> 25 fps, i.e. one feature frame per 0.04 s.
FRAME_SECONDS: float = 0.04

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(help="Linear frame probe on frozen Qwen2-Audio audio features.")


# ---------------------------------------------------------------------------
# Pure helpers (numpy only; unit-tested in tests/test_frame_probe.py)
# ---------------------------------------------------------------------------


def frame_labels(num_frames: int, start: float, end: float) -> np.ndarray:
    """Binary per-frame labels for one degradation interval.

    Frame ``i`` covers ``[i, i+1) * FRAME_SECONDS``; it is labeled degraded
    when its center falls inside ``[start, end)``.

    Args:
        num_frames: Number of valid feature frames for the clip.
        start: Degradation window start in seconds.
        end: Degradation window end in seconds.

    Returns:
        ``(num_frames,)`` float32 array of 0/1 labels.
    """
    centers = (np.arange(num_frames, dtype=np.float64) + 0.5) * FRAME_SECONDS
    return ((centers >= start) & (centers < end)).astype(np.float32)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding, same length as input.

    Args:
        values: 1-D array of scores.
        window: Window size in frames; values below 2 return the input.

    Returns:
        Smoothed array of the same shape.
    """
    if window < 2 or len(values) == 0:
        return values.astype(np.float64)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values.astype(np.float64), (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def longest_run(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Longest contiguous run of True values.

    Args:
        mask: 1-D boolean array.

    Returns:
        ``(start_idx, end_idx)`` of the longest run with ``end`` exclusive,
        or ``None`` when the mask has no True entries.
    """
    best: Optional[Tuple[int, int]] = None
    run_start: Optional[int] = None
    for i, flag in enumerate(list(mask) + [False]):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            if best is None or (i - run_start) > (best[1] - best[0]):
                best = (run_start, i)
            run_start = None
    return best


def recover_interval(
    scores: np.ndarray,
    threshold: float,
    smooth_window: int = 5,
) -> Tuple[float, float]:
    """Turn per-frame probe scores into one ``(start, end)`` interval.

    Smooths the scores, thresholds them, and takes the longest contiguous
    run. When no frame clears the threshold, falls back to the peak frame
    extended by half the smoothing window on each side, so the recovery
    always produces a (small) interval rather than failing.

    Args:
        scores: ``(num_frames,)`` probe scores in ``[0, 1]``.
        threshold: Score cutoff for the degraded mask.
        smooth_window: Moving-average window in frames.

    Returns:
        ``(start_seconds, end_seconds)`` interval.
    """
    smoothed = moving_average(scores, smooth_window)
    run = longest_run(smoothed >= threshold)
    if run is None:
        # Smoothing turns a single hot frame into a plateau; argmax alone
        # would return the plateau's left edge, so center on the plateau.
        peak_indices = np.flatnonzero(smoothed == smoothed.max())
        peak = int(peak_indices[len(peak_indices) // 2])
        half = max(1, smooth_window // 2)
        run = (max(0, peak - half), min(len(smoothed), peak + half + 1))
    return run[0] * FRAME_SECONDS, run[1] * FRAME_SECONDS


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney rank statistic (no sklearn dependency).

    Args:
        scores: 1-D array of probe scores.
        labels: 1-D array of binary labels aligned with ``scores``.

    Returns:
        AUC in ``[0, 1]``; 0.5 when either class is empty.
    """
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    rank_position = 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        mean_rank = (rank_position + rank_position + (j - i)) / 2.0
        ranks[order[i : j + 1]] = mean_rank
        rank_position += j - i + 1
        i = j + 1
    pos_rank_sum = float(ranks[labels].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def interval_iou_values(a0: float, a1: float, b0: float, b1: float) -> float:
    """Temporal IoU of two ``[start, end]`` intervals given as scalars."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def grouped_split(
    group_keys: Sequence[str],
    val_fraction: float,
) -> Tuple[List[int], List[int]]:
    """Deterministic grouped train/val split by source reference.

    Every k-th group (sorted) goes to validation, ``k = round(1/val_fraction)``,
    so placement-augmented reuses of the same REF never straddle the split and
    reruns produce identical assignments.

    Args:
        group_keys: One group key per sample (e.g. ``filename_ref``).
        val_fraction: Approximate fraction of groups assigned to validation.

    Returns:
        ``(train_indices, val_indices)`` into the original sample order.
    """
    every_k = max(2, int(round(1.0 / max(val_fraction, 1e-6))))
    unique_groups = sorted(set(group_keys))
    val_groups = {g for i, g in enumerate(unique_groups) if i % every_k == 0}
    train_idx = [i for i, g in enumerate(group_keys) if g not in val_groups]
    val_idx = [i for i, g in enumerate(group_keys) if g in val_groups]
    return train_idx, val_idx


def best_constant_interval(
    truths: List[Tuple[float, float]],
    start_step: float = 0.25,
    length_min: float = 0.5,
    length_max: float = 4.0,
    length_step: float = 0.25,
) -> Tuple[Tuple[float, float], float]:
    """Grid-search the strongest audio-blind constant interval (context only).

    Mirrors ``best_constant_baseline`` in ``scripts/eval/evaluate_temporal.py``
    without importing its torch-heavy module chain.

    Args:
        truths: Ground-truth ``(start, end)`` tuples.
        start_step: Start-time grid step in seconds.
        length_min: Smallest candidate window length.
        length_max: Largest candidate window length.
        length_step: Window-length grid step.

    Returns:
        ``((start, end), mean_iou)`` of the best constant guess.
    """
    if not truths:
        return (0.0, 0.0), 0.0
    max_start = max(end for _, end in truths)
    best = ((0.0, length_min), -1.0)
    start = 0.0
    while start <= max_start:
        length = length_min
        while length <= length_max + 1e-9:
            candidate = (start, start + length)
            score = float(
                np.mean(
                    [
                        interval_iou_values(candidate[0], candidate[1], s, e)
                        for s, e in truths
                    ]
                )
            )
            if score > best[1]:
                best = (candidate, score)
            length += length_step
        start += start_step
    return best


# ---------------------------------------------------------------------------
# Feature extraction (frozen audio tower + projector)
# ---------------------------------------------------------------------------


def load_frozen_audio_pathway(
    model_path: str,
    device: str,
) -> Tuple[torch.nn.Module, torch.nn.Module, Any]:
    """Load only the audio tower + projector of a Qwen2-Audio checkpoint.

    The language model is discarded after loading so GPU memory holds the
    ~600M-parameter audio pathway instead of the full 7B model. TimeAudio
    checkpoints load through the subclass (extended vocab + extra parameter)
    exactly as in ``asa.inference.load_model``; the abs-time embedding is not
    part of the extracted pathway (see module docstring).

    Args:
        model_path: Hub repo ID or local checkpoint directory.
        device: Torch device string for the audio pathway.

    Returns:
        ``(audio_tower, projector, feature_extractor)`` with modules in eval
        mode on ``device``.
    """
    from transformers import (
        AutoConfig,
        AutoProcessor,
        Qwen2AudioForConditionalGeneration,
    )

    from asa.modeling_timeaudio import Qwen2AudioTimeForConditionalGeneration

    try:
        cfg = AutoConfig.from_pretrained(model_path)
        is_timeaudio = bool(
            getattr(cfg, "use_abs_time_embedding", False)
            or getattr(cfg, "use_time_tokens", False)
        )
    except Exception:
        is_timeaudio = False
    model_cls = (
        Qwen2AudioTimeForConditionalGeneration
        if is_timeaudio
        else Qwen2AudioForConditionalGeneration
    )

    logging.info(
        "Loading %s via %s (audio pathway only)", model_path, model_cls.__name__
    )
    model = model_cls.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    audio_tower = model.audio_tower.to(device).eval()
    projector = model.multi_modal_projector.to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    feature_extractor = processor.feature_extractor
    # Drop the 7B language model; only the audio pathway is needed.
    del model
    return audio_tower, projector, feature_extractor


@torch.no_grad()
def extract_clip_features(
    audio_tower: torch.nn.Module,
    projector: torch.nn.Module,
    feature_extractor: Any,
    audio_paths: List[str],
    device: str,
    batch_size: int = 8,
) -> List[Dict[str, np.ndarray]]:
    """Per-frame features for each clip at the two probe taps.

    Mirrors the audio path of ``Qwen2AudioForConditionalGeneration.forward``
    (mel features, length computation, padding attention mask, encoder,
    projector) and slices each clip to its valid frame count.

    Args:
        audio_tower: Frozen Qwen2-Audio audio encoder.
        projector: Frozen multi-modal projector.
        feature_extractor: The checkpoint's Whisper-style feature extractor.
        audio_paths: Resolved paths to the mix WAV files.
        device: Torch device of the audio pathway.
        batch_size: Clips per forward pass.

    Returns:
        One dict per clip: ``encoder`` ``(L, 1280)`` float16, ``projector``
        ``(L, 4096)`` float16, with ``L`` the clip's valid frame count.
    """
    results: List[Dict[str, np.ndarray]] = []
    sr = feature_extractor.sampling_rate
    dtype = next(audio_tower.parameters()).dtype

    for start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[start : start + batch_size]
        audios = [load_audio(p, target_sr=sr) for p in batch_paths]
        features = feature_extractor(
            audios,
            sampling_rate=sr,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
        )
        input_features = features.input_features.to(device, dtype)
        feature_attention_mask = features.attention_mask.to(device)

        feat_lengths, output_lengths = audio_tower._get_feat_extract_output_lengths(
            feature_attention_mask.sum(-1)
        )
        batch, _, max_mel_seq_len = input_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        seq_range = (
            torch.arange(0, max_seq_len, dtype=feat_lengths.dtype, device=device)
            .unsqueeze(0)
            .expand(batch, max_seq_len)
        )
        lengths_expand = feat_lengths.unsqueeze(1).expand(batch, max_seq_len)
        padding_mask = seq_range >= lengths_expand
        audio_attention_mask_ = padding_mask.view(batch, 1, 1, max_seq_len).expand(
            batch, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(dtype)
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        encoder_out = audio_tower(
            input_features, attention_mask=audio_attention_mask
        ).last_hidden_state
        projector_out = projector(encoder_out)

        for i in range(batch):
            valid = int(output_lengths[i].item())
            results.append(
                {
                    "encoder": encoder_out[i, :valid]
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float16),
                    "projector": projector_out[i, :valid]
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float16),
                }
            )
        if (start // batch_size) % 20 == 0:
            logging.info(
                "Extracted features: %d / %d clips",
                min(start + batch_size, len(audio_paths)),
                len(audio_paths),
            )
    return results


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------


def fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    device: str,
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 8192,
) -> Tuple[torch.nn.Linear, np.ndarray, np.ndarray]:
    """Fit a logistic-regression probe with BCE on standardized features.

    Args:
        train_x: ``(N, d)`` float array of frame features.
        train_y: ``(N,)`` binary labels.
        device: Torch device for probe training.
        epochs: Passes over the training frames.
        lr: Adam learning rate.
        batch_size: Frames per probe step.

    Returns:
        ``(probe, mean, std)`` where mean/std are the train standardization
        statistics to apply at inference.
    """
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = (train_x.std(axis=0, dtype=np.float64) + 1e-6).astype(np.float32)

    probe = torch.nn.Linear(train_x.shape[1], 1).to(device)
    pos = max(float(train_y.sum()), 1.0)
    neg = max(float(len(train_y) - train_y.sum()), 1.0)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / pos], device=device)
    )
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    n = len(train_x)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = torch.from_numpy(
                ((train_x[idx].astype(np.float32) - mean) / std)
            ).to(device)
            yb = torch.from_numpy(train_y[idx].astype(np.float32)).to(device)
            optimizer.zero_grad()
            logits = probe(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(idx)
        logging.info("Probe epoch %d: loss %.4f", epoch + 1, epoch_loss / n)
    return probe, mean, std


@torch.no_grad()
def probe_scores(
    probe: torch.nn.Linear,
    mean: np.ndarray,
    std: np.ndarray,
    features: np.ndarray,
    device: str,
    batch_size: int = 16384,
) -> np.ndarray:
    """Sigmoid probe scores for a feature matrix.

    Args:
        probe: Trained linear probe.
        mean: Train-set feature means.
        std: Train-set feature stds.
        features: ``(N, d)`` frame features.
        device: Torch device of the probe.
        batch_size: Frames per scoring step.

    Returns:
        ``(N,)`` float64 scores in ``[0, 1]``.
    """
    out: List[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        xb = torch.from_numpy(
            ((features[start : start + batch_size].astype(np.float32) - mean) / std)
        ).to(device)
        out.append(torch.sigmoid(probe(xb).squeeze(-1)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pick the score threshold maximizing frame F1 on the given (train) set.

    Args:
        scores: Frame scores in ``[0, 1]``.
        labels: Binary frame labels.

    Returns:
        The best threshold from a fixed 0.05-step grid.
    """
    best_threshold = 0.5
    best_f1 = -1.0
    labels = labels.astype(bool)
    for threshold in np.arange(0.05, 0.96, 0.05):
        predicted = scores >= threshold
        tp = float(np.sum(predicted & labels))
        fp = float(np.sum(predicted & ~labels))
        fn = float(np.sum(~predicted & labels))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def probe(
    model_path: str = typer.Option(
        "Qwen/Qwen2-Audio-7B",
        help="Hub repo ID or local checkpoint whose audio pathway is probed.",
    ),
    json_path: Path = typer.Option(
        Path(
            "data/processed/temporal/"
            "train_nisqa_temporal_global_caption_aug_anchoroffset.json"
        ),
        help="Temporal JSONL with mix_deg_segments ground truth.",
    ),
    data_root: Path = typer.Option(
        Path("data"), help="Root directory used to resolve audio paths."
    ),
    max_samples: int = typer.Option(
        2000, help="Clips to probe (deterministic even spread over the JSONL)."
    ),
    val_fraction: float = typer.Option(
        0.2, help="Fraction of REF groups held out for validation."
    ),
    batch_size: int = typer.Option(8, help="Clips per feature-extraction batch."),
    epochs: int = typer.Option(3, help="Probe training epochs."),
    lr: float = typer.Option(1e-3, help="Probe learning rate."),
    smooth_window: int = typer.Option(
        5, help="Moving-average window (frames) for interval recovery."
    ),
    output_dir: Path = typer.Option(
        Path("results/analysis/frame_probe"),
        help="Directory for the metrics JSON and per-clip CSV.",
    ),
) -> None:
    """Run the frozen-feature frame probe and report the gate verdict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_processed_records(json_path)
    rows: List[Dict[str, Any]] = []
    for record in records:
        segments = record.get("mix_deg_segments") or []
        if isinstance(segments, str):
            try:
                segments = json.loads(segments)
            except json.JSONDecodeError:
                segments = []
        valid = [
            (float(s.get("start", 0.0)), float(s.get("end", 0.0)))
            for s in segments
            if isinstance(s, dict) and float(s.get("end", 0.0)) > float(s.get("start", 0.0))
        ]
        audios = record.get("audios") or []
        if not valid or not audios:
            continue
        start, end = max(valid, key=lambda item: item[1] - item[0])
        audio_path = resolve_audio_path(str(audios[0]), data_root)
        if not audio_path.exists():
            continue
        rows.append(
            {
                "audio_path": str(audio_path),
                "start": start,
                "end": end,
                "ref": str(record.get("filename_ref", record.get("id", ""))),
                "id": str(record.get("id", "")),
            }
        )

    if len(rows) > max_samples:
        keep = np.unique(np.linspace(0, len(rows) - 1, max_samples).astype(int))
        rows = [rows[i] for i in keep]
    logging.info("Probing %d clips from %s", len(rows), json_path)

    audio_tower, projector, feature_extractor = load_frozen_audio_pathway(
        model_path, device
    )
    features = extract_clip_features(
        audio_tower,
        projector,
        feature_extractor,
        [row["audio_path"] for row in rows],
        device,
        batch_size=batch_size,
    )

    labels = [
        frame_labels(feat["encoder"].shape[0], row["start"], row["end"])
        for row, feat in zip(rows, features)
    ]
    train_idx, val_idx = grouped_split([row["ref"] for row in rows], val_fraction)
    logging.info(
        "Grouped split: %d train clips, %d val clips", len(train_idx), len(val_idx)
    )

    val_truths = [(rows[i]["start"], rows[i]["end"]) for i in val_idx]
    constant_interval, constant_tiou = best_constant_interval(val_truths)
    whole_clip_ious = [
        interval_iou_values(
            0.0,
            features[i]["encoder"].shape[0] * FRAME_SECONDS,
            rows[i]["start"],
            rows[i]["end"],
        )
        for i in val_idx
    ]
    baselines = {
        "whole_clip_mean_tiou": float(np.mean(whole_clip_ious)),
        "best_constant_mean_tiou": float(constant_tiou),
        "best_constant_interval": list(constant_interval),
    }

    summary: Dict[str, Any] = {
        "model_path": model_path,
        "json_path": str(json_path),
        "clips": len(rows),
        "train_clips": len(train_idx),
        "val_clips": len(val_idx),
        "val_baselines_audio_blind": baselines,
        "taps": {},
    }
    per_clip_rows: List[Dict[str, Any]] = []

    for tap in ("encoder", "projector"):
        train_x = np.concatenate([features[i][tap] for i in train_idx], axis=0)
        train_y = np.concatenate([labels[i] for i in train_idx], axis=0)
        probe_model, mean, std = fit_linear_probe(
            train_x, train_y, device, epochs=epochs, lr=lr
        )

        train_scores = probe_scores(probe_model, mean, std, train_x, device)
        threshold = best_f1_threshold(train_scores, train_y)

        val_scores_flat: List[np.ndarray] = []
        val_labels_flat: List[np.ndarray] = []
        tious: List[float] = []
        for i in val_idx:
            scores = probe_scores(probe_model, mean, std, features[i][tap], device)
            val_scores_flat.append(scores)
            val_labels_flat.append(labels[i])
            rec_start, rec_end = recover_interval(scores, threshold, smooth_window)
            tiou = interval_iou_values(
                rec_start, rec_end, rows[i]["start"], rows[i]["end"]
            )
            tious.append(tiou)
            per_clip_rows.append(
                {
                    "tap": tap,
                    "id": rows[i]["id"],
                    "gt_start": rows[i]["start"],
                    "gt_end": rows[i]["end"],
                    "pred_start": rec_start,
                    "pred_end": rec_end,
                    "tiou": tiou,
                }
            )

        frame_auc = rank_auc(
            np.concatenate(val_scores_flat), np.concatenate(val_labels_flat)
        )
        tious_arr = np.array(tious)
        summary["taps"][tap] = {
            "val_frame_auc": float(frame_auc),
            "recovery_threshold": threshold,
            "val_mean_tiou": float(tious_arr.mean()),
            "val_median_tiou": float(np.median(tious_arr)),
            "val_hit_iou_ge_0_5": float((tious_arr >= 0.5).mean()),
        }
        logging.info(
            "[%s] frame AUC %.3f | recovered mean t-IoU %.3f (median %.3f, "
            "Hit@0.5 %.3f) vs audio-blind best-constant %.3f / whole-clip %.3f",
            tap,
            frame_auc,
            tious_arr.mean(),
            float(np.median(tious_arr)),
            float((tious_arr >= 0.5).mean()),
            constant_tiou,
            baselines["whole_clip_mean_tiou"],
        )

    projector_tiou = summary["taps"]["projector"]["val_mean_tiou"]
    margin = projector_tiou - max(
        baselines["best_constant_mean_tiou"], baselines["whole_clip_mean_tiou"]
    )
    summary["gate_verdict"] = (
        "INFORMATION PRESENT: probe beats audio-blind baselines; the SFT "
        "failure is an objective problem, loss-level fixes can work"
        if margin > 0.05
        else "INFORMATION WEAK OR ABSENT at this tap: text-side loss changes "
        "alone are unlikely to fix localization; pivot to task/data/encoder"
    )
    logging.info("GATE: %s (margin %.3f)", summary["gate_verdict"], margin)

    out_json = output_dir / "probe_results.json"
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    out_csv = output_dir / "probe_per_clip.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tap",
                "id",
                "gt_start",
                "gt_end",
                "pred_start",
                "pred_end",
                "tiou",
            ],
        )
        writer.writeheader()
        writer.writerows(per_clip_rows)
    logging.info("Saved %s and %s", out_json, out_csv)


if __name__ == "__main__":
    app()
