"""Predicted-vs-true MOS calibration plot used in the thesis results chapter.

The figure pools the four MOS evaluation sets and compares the off-the-shelf
Qwen2-Audio baseline with the full-data SFT and DPO models. The points are
per-clip predictions; the thick line is a calibration curve, computed as the
mean predicted MOS in each half-point true-MOS bin.

Run locally:

    python scripts/analysis/eval_pred_vs_true_calibrated.py \
        --figures-dir ../bachelor-thesis/figures
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

GRID_GREY = "#d9d9d9"
ZERO_SHOT_HISTORY_COMMIT = "b597f0658004a78dd1c5603a89d53ed2c9e8b87e"

SETS = ["FOR", "LIVE", "P501", "nisqa_indomain"]
MODELS: list[dict[str, str]] = [
    {
        "key": "zeroshot",
        "label": "Zero-shot",
        "dir": "results/evaluation/zeroshot/qwen2audio_instruct_baseline",
        "point_color": "#5c6b73",
        "line_color": "#2693c8",
    },
    {
        "key": "sft",
        "label": "Full SFT",
        "dir": "results/evaluation/sft/sft_full_paper_h100_eval_greedy",
        "point_color": "#2a6f7f",
        "line_color": "#1aa6c8",
    },
    {
        "key": "dpo",
        "label": "Full DPO",
        "dir": "results/evaluation/dpo/dpo_full_sft_paired_lr1e6_eval_greedy",
        "point_color": "#d98b34",
        "line_color": "#df7f19",
    },
]


def set_paper_style() -> None:
    """Use large source fonts so text survives LaTeX down-scaling."""
    sns.set_theme(context="paper", style="whitegrid", font="serif")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 17,
            "axes.titlesize": 21,
            "axes.titleweight": "semibold",
            "axes.labelsize": 19,
            "axes.labelcolor": "#222222",
            "axes.edgecolor": "#5c6b73",
            "axes.linewidth": 1.0,
            "text.color": "#222222",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "legend.fontsize": 15,
            "legend.frameon": False,
            "grid.color": GRID_GREY,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
        }
    )


def read_result_json(path: Path) -> dict:
    """Read a result JSON, falling back to the historical zero-shot commit."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    repo_relative = path.as_posix()
    completed = subprocess.run(
        ["git", "show", f"{ZERO_SHOT_HISTORY_COMMIT}:{repo_relative}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def load_pairs(repo_root: Path, model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled true and predicted MOS arrays for one model."""
    true_values: list[float] = []
    pred_values: list[float] = []
    for eval_set in SETS:
        path = repo_root / model_dir / f"test_{eval_set}_results.json"
        payload = read_result_json(path)
        for row in payload["results"]:
            if row.get("mos") is None or row.get("predicted_mos") is None:
                continue
            true_values.append(float(row["mos"]))
            pred_values.append(float(row["predicted_mos"]))
    return np.asarray(true_values), np.asarray(pred_values)


def calibration_curve(
    true: np.ndarray, pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean prediction in each half-point true-MOS bin."""
    bins = np.arange(1.0, 5.51, 0.5)
    centers: list[float] = []
    means: list[float] = []
    for low, high in zip(bins[:-1], bins[1:]):
        if high >= bins[-1]:
            mask = (true >= low) & (true <= high)
        else:
            mask = (true >= low) & (true < high)
        if np.any(mask):
            centers.append((low + high) / 2.0)
            means.append(float(np.mean(pred[mask])))
    return np.asarray(centers), np.asarray(means)


def plot(repo_root: Path, figures_dir: Path) -> None:
    """Write the calibrated predicted-vs-true figure."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharex=True, sharey=True)
    rng = np.random.default_rng(0)

    for axis, spec in zip(axes, MODELS):
        true, pred = load_pairs(repo_root, Path(spec["dir"]))
        pearson = float(stats.pearsonr(true, pred)[0])

        axis.scatter(
            true + rng.normal(0, 0.035, len(true)),
            pred + rng.normal(0, 0.035, len(pred)),
            s=12,
            alpha=0.20,
            color=spec["point_color"],
            linewidths=0,
            zorder=3,
        )
        axis.plot(
            [1, 5],
            [1, 5],
            linestyle="--",
            linewidth=1.4,
            color="#b0b0b0",
            label="perfect",
            zorder=1,
        )
        centers, means = calibration_curve(true, pred)
        axis.plot(
            centers,
            means,
            color=spec["line_color"],
            linewidth=3.0,
            marker="o",
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="calibration",
            zorder=4,
        )
        axis.text(
            0.05,
            0.94,
            f"$r = {pearson:.2f}$",
            transform=axis.transAxes,
            va="top",
            fontsize=19,
            color="#222222",
        )
        axis.set_title(spec["label"])
        axis.set_xlabel("True MOS")
        axis.set_xlim(0.9, 5.1)
        axis.set_ylim(0.9, 5.1)
        axis.legend(loc="lower right")
        sns.despine(ax=axis)

    axes[0].set_ylabel("Predicted MOS")
    fig.tight_layout(w_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"eval_pred_vs_true_calibrated.{ext}")
    plt.close(fig)
    print(f"Wrote eval_pred_vs_true_calibrated.pdf / .png to {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Repository root holding results/evaluation/.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("../bachelor-thesis/figures"),
        help="Where to write the figure.",
    )
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()
    plot(args.data_root, args.figures_dir)


if __name__ == "__main__":
    main()
