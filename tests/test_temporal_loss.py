import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from asa.temporal_loss import (  # noqa: E402
    IGNORE_INDEX,
    gaussian_group_targets,
    temporal_weighted_ce,
)

# A toy vocabulary: ids 0-9 are "text", 10-19 are 10 ordered anchors,
# 20-22 are 3 ordered offsets.
VOCAB = 23
ANCHORS = torch.arange(10, 20)
OFFSETS = torch.arange(20, 23)


def _random_batch(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 6, VOCAB, generator=gen)
    labels = torch.tensor(
        [
            [IGNORE_INDEX, 3, 14, 21, 5, 1],
            [IGNORE_INDEX, IGNORE_INDEX, 2, 12, 20, 4],
        ]
    )
    return logits, labels


def test_gaussian_targets_peak_at_truth_and_sum_to_one() -> None:
    targets = gaussian_group_targets(torch.tensor([5]), group_size=10, sigma=1.0)

    assert targets.shape == (1, 10)
    assert targets.sum().item() == pytest.approx(1.0)
    assert targets.argmax().item() == 5
    # Symmetric neighbors get equal partial credit.
    assert targets[0, 4].item() == pytest.approx(targets[0, 6].item())
    assert targets[0, 4].item() > targets[0, 2].item()


def test_gaussian_targets_sigma_zero_is_one_hot() -> None:
    targets = gaussian_group_targets(torch.tensor([3]), group_size=5, sigma=0.0)

    assert targets.tolist() == [[0.0, 0.0, 0.0, 1.0, 0.0]]


def test_gaussian_targets_output_device_matches_input() -> None:
    # Regression guard for the 28633523 crash: the internal arange must live
    # on the same device as label_positions, or any non-CPU run dies with
    # "two devices, cuda:0 and cpu". CPU here pins the contract; the CUDA test
    # below exercises the real failing path when a GPU is present.
    labels = torch.tensor([2, 3, 5])
    out = gaussian_group_targets(labels, group_size=60, sigma=1.0)
    assert out.device == labels.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gaussian_targets_runs_on_cuda() -> None:
    labels = torch.tensor([2, 3, 5], device="cuda")
    out = gaussian_group_targets(labels, group_size=60, sigma=1.0)
    assert out.is_cuda
    assert out.shape == (3, 60)


def test_matches_plain_ce_when_inactive() -> None:
    logits, labels = _random_batch()

    custom = temporal_weighted_ce(
        logits, labels, ANCHORS, OFFSETS, time_weight=1.0, soft_sigma=0.0
    )
    reference = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, VOCAB),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )

    assert custom.item() == pytest.approx(reference.item(), rel=1e-5)


def test_weight_scales_only_time_token_positions() -> None:
    logits, labels = _random_batch()

    base = temporal_weighted_ce(logits, labels, ANCHORS, OFFSETS, 1.0, 0.0)
    weighted = temporal_weighted_ce(logits, labels, ANCHORS, OFFSETS, 5.0, 0.0)

    # Reconstruct: weighted = base + 4 * (sum of time-token CE) / n_supervised.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != IGNORE_INDEX
    flat_logits = shift_logits[mask]
    flat_labels = shift_labels[mask]
    per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    time_mask = torch.isin(flat_labels, torch.cat([ANCHORS, OFFSETS]))
    expected = (per_token.sum() + 4.0 * per_token[time_mask].sum()) / mask.sum()

    assert weighted.item() == pytest.approx(expected.item(), rel=1e-5)
    assert weighted.item() > base.item()


def test_soft_targets_give_partial_credit_for_near_miss() -> None:
    # One supervised position, true anchor index 5 (token id 15). Two models:
    # one puts its mass on the neighbor anchor (14), one on a far anchor (19).
    labels = torch.tensor([[IGNORE_INDEX, 15]])
    near = torch.full((1, 2, VOCAB), -10.0)
    far = torch.full((1, 2, VOCAB), -10.0)
    near[0, 0, 14] = 10.0
    far[0, 0, 19] = 10.0

    near_loss = temporal_weighted_ce(near, labels, ANCHORS, OFFSETS, 1.0, 1.0)
    far_loss = temporal_weighted_ce(far, labels, ANCHORS, OFFSETS, 1.0, 1.0)
    # Under one-hot CE both misses would cost the same.
    near_hard = temporal_weighted_ce(near, labels, ANCHORS, OFFSETS, 1.0, 0.0)
    far_hard = temporal_weighted_ce(far, labels, ANCHORS, OFFSETS, 1.0, 0.0)

    assert near_loss.item() < far_loss.item()
    assert near_hard.item() == pytest.approx(far_hard.item(), rel=1e-5)


def test_ignore_index_positions_do_not_contribute() -> None:
    logits = torch.randn(1, 4, VOCAB)
    all_masked = torch.full((1, 4), IGNORE_INDEX)

    loss = temporal_weighted_ce(logits, all_masked, ANCHORS, OFFSETS, 5.0, 1.0)

    assert loss.item() == 0.0


def test_caption_positions_unaffected_by_sigma() -> None:
    # Labels contain no time tokens: sigma and weight must change nothing.
    logits, _ = _random_batch(seed=1)
    labels = torch.tensor([[IGNORE_INDEX, 3, 5, 1, 2, 4], [0, 1, 2, 3, 4, 5]])

    plain = temporal_weighted_ce(logits, labels, ANCHORS, OFFSETS, 1.0, 0.0)
    tuned = temporal_weighted_ce(logits, labels, ANCHORS, OFFSETS, 7.0, 2.0)

    assert plain.item() == pytest.approx(tuned.item(), rel=1e-6)
