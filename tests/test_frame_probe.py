import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.analysis.probe_temporal_frames import (
    FRAME_SECONDS,
    best_constant_interval,
    frame_labels,
    grouped_split,
    interval_iou_values,
    longest_run,
    moving_average,
    rank_auc,
    recover_interval,
)


def test_frame_labels_marks_frames_inside_window() -> None:
    # 25 fps: frame centers at 0.02, 0.06, 0.10, ... Window [0.05, 0.13)
    # contains the centers of frames 1 and 2 only.
    labels = frame_labels(num_frames=5, start=0.05, end=0.13)

    assert labels.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]


def test_frame_labels_empty_window_is_all_zero() -> None:
    labels = frame_labels(num_frames=4, start=2.0, end=2.0)

    assert labels.sum() == 0


def test_moving_average_preserves_length_and_smooths() -> None:
    values = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

    smoothed = moving_average(values, window=3)

    assert len(smoothed) == len(values)
    assert smoothed[2] == pytest.approx(1.0 / 3.0)
    assert smoothed[0] == pytest.approx(0.0)


def test_longest_run_picks_longest_block() -> None:
    mask = np.array([True, False, True, True, True, False, True])

    assert longest_run(mask) == (2, 5)


def test_longest_run_none_when_empty() -> None:
    assert longest_run(np.array([False, False])) is None


def test_recover_interval_from_clean_scores() -> None:
    # Frames 10..19 hot: expect an interval close to [0.4 s, 0.8 s].
    scores = np.zeros(50)
    scores[10:20] = 1.0

    start, end = recover_interval(scores, threshold=0.5, smooth_window=1)

    assert start == pytest.approx(10 * FRAME_SECONDS)
    assert end == pytest.approx(20 * FRAME_SECONDS)


def test_recover_interval_falls_back_to_peak() -> None:
    scores = np.full(30, 0.1)
    scores[12] = 0.4  # below threshold, but the peak

    start, end = recover_interval(scores, threshold=0.9, smooth_window=5)

    assert start < 12 * FRAME_SECONDS < end
    assert end - start <= 6 * FRAME_SECONDS


def test_rank_auc_perfect_and_chance() -> None:
    labels = np.array([0, 0, 1, 1])

    assert rank_auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == pytest.approx(1.0)
    assert rank_auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == pytest.approx(0.0)
    assert rank_auc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == pytest.approx(0.5)


def test_rank_auc_degenerate_single_class() -> None:
    assert rank_auc(np.array([0.1, 0.9]), np.array([1, 1])) == 0.5


def test_grouped_split_keeps_ref_groups_together() -> None:
    refs = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]

    train_idx, val_idx = grouped_split(refs, val_fraction=0.2)

    train_refs = {refs[i] for i in train_idx}
    val_refs = {refs[i] for i in val_idx}
    assert train_refs.isdisjoint(val_refs)
    assert len(train_idx) + len(val_idx) == len(refs)
    assert val_idx  # at least one group held out


def test_interval_iou_values_matches_hand_computation() -> None:
    assert interval_iou_values(0.0, 2.0, 1.0, 3.0) == pytest.approx(1.0 / 3.0)
    assert interval_iou_values(0.0, 1.0, 2.0, 3.0) == 0.0


def test_best_constant_interval_finds_shared_window() -> None:
    truths = [(2.0, 3.0), (2.0, 3.0), (2.25, 3.25)]

    (start, end), score = best_constant_interval(truths)

    # The best constant guess should overlap the shared region around 2-3 s.
    assert start <= 2.5 <= end
    assert score > 0.5
