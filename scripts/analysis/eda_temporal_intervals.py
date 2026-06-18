"""Plot the temporal degradation interval distribution.

The figure characterises the time-localized NISQA-temporal training set by the
length of the inserted degradation interval and by the same interval measured as
a share of the full clip.

Run locally:

    python scripts/analysis/eda_temporal_intervals.py \
        --figures-dir ../bachelor-thesis/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PRIMARY = "#2a6f7f"
HIGHLIGHT = "#d98b34"
HIGHLIGHT_2 = "#3d3d3d"
GRID_GREY = "#d9d9d9"


def set_paper_style() -> None:
    """Use the same thesis-friendly typography as the other EDA figures."""
    sns.set_theme(context="paper", style="whitegrid", font="serif")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.titleweight": "semibold",
            "axes.labelsize": 17,
            "axes.labelcolor": "#222222",
            "axes.edgecolor": "#5c6b73",
            "axes.linewidth": 0.9,
            "text.color": "#222222",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "legend.fontsize": 15,
            "legend.title_fontsize": 15,
            "legend.frameon": False,
            "grid.color": GRID_GREY,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
        }
    )


def load_intervals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return interval lengths and interval share of clip duration."""
    lengths: list[float] = []
    shares: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            duration = float(record["duration_seconds"])
            for segment in record["mix_deg_segments"]:
                length = float(segment["end"]) - float(segment["start"])
                lengths.append(length)
                shares.append(100.0 * length / duration)
    return np.asarray(lengths), np.asarray(shares)


def add_reference_lines(axis: plt.Axes, values: np.ndarray, unit: str) -> None:
    """Mark mean and median in a compact legend."""
    mean_value = float(np.mean(values))
    median_value = float(np.median(values))
    axis.axvline(
        mean_value,
        color=HIGHLIGHT,
        linewidth=2.0,
        linestyle="--",
        label=f"mean = {mean_value:.2f}{unit}",
    )
    axis.axvline(
        median_value,
        color=HIGHLIGHT_2,
        linewidth=1.8,
        linestyle=":",
        label=f"median = {median_value:.2f}{unit}",
    )
    axis.legend(loc="upper right")


def plot(lengths: np.ndarray, shares: np.ndarray, figures_dir: Path) -> dict:
    """Write the interval distribution figure and return caption statistics."""
    fig, (ax_len, ax_share) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    ax_len.hist(
        lengths,
        bins=np.linspace(0.55, 3.05, 28),
        color=PRIMARY,
        edgecolor="white",
        linewidth=0.6,
    )
    add_reference_lines(ax_len, lengths, " s")
    ax_len.set_title("(a) Interval length")
    ax_len.set_xlabel("Interval length (seconds)")
    ax_len.set_ylabel("Number of mixes")
    ax_len.grid(axis="x", visible=False)

    ax_share.hist(
        shares,
        bins=np.linspace(4.5, 25.5, 28),
        color=PRIMARY,
        edgecolor="white",
        linewidth=0.6,
    )
    add_reference_lines(ax_share, shares, "%")
    ax_share.set_title("(b) Share of clip")
    ax_share.set_xlabel("Interval share of clip (%)")
    ax_share.set_ylabel("Number of mixes")
    ax_share.grid(axis="x", visible=False)

    sns.despine(fig=fig)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"eda_temporal_interval.{ext}")
    plt.close(fig)

    return {
        "n": int(len(lengths)),
        "length_mean": float(np.mean(lengths)),
        "length_median": float(np.median(lengths)),
        "length_min": float(np.min(lengths)),
        "length_max": float(np.max(lengths)),
        "share_mean": float(np.mean(shares)),
        "share_median": float(np.median(shares)),
        "share_min": float(np.min(shares)),
        "share_max": float(np.max(shares)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temporal-jsonl",
        type=Path,
        default=Path(
            "data/processed/temporal/train_nisqa_temporal_global_caption_aug.json"
        ),
        help="Temporal training set JSONL with mix_deg_segments.",
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
    lengths, shares = load_intervals(args.temporal_jsonl)
    stats = plot(lengths, shares, args.figures_dir)
    stats_path = args.figures_dir / "eda_temporal_interval_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote eda_temporal_interval.pdf / .png to {args.figures_dir}")
    print(f"Wrote stats to {stats_path}")


if __name__ == "__main__":
    main()
