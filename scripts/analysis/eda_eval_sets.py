"""Descriptive analysis of the evaluation sets (in-domain + out-of-domain).

The model is trained on synthetic NISQA-SIM clips and evaluated on two kinds of
held-out data:

* in-domain  -- ``test_nisqa_indomain.json`` (500 clips), a held-out slice of the
  same synthetic NISQA-SIM distribution the model trains on.
* out-of-domain -- the three NISQA real-condition test sets ``test_FOR`` (240),
  ``test_LIVE`` (200), and ``test_P501`` (240). These are real recordings/conditions,
  not synthetic mixes.

This script characterises the eval *datasets* (their label distributions), not model
performance: it reads only the reference label fields (``mos`` and the NISQA quality
sub-dimensions). It deliberately ignores ``mos_predictions.json`` because that file's
captions are an API-key error string, not real predictions.

Two figures, reusing the shared seaborn theme so they match the training-set EDA:

1. MOS distribution across the four eval sets (the train/test distribution-shift
   story: synthetic in-domain vs real FOR / P501 / LIVE).
2. Quality sub-dimension means across the sets, on the three dimensions shared by all
   four sets (noisiness, coloration, loudness). Discontinuity is shown OOD-only with
   an explicit note, since the in-domain set does not carry it.

Run locally, no HPC:

    python scripts/analysis/eda_eval_sets.py --figures-dir ../_papers/asa-thesis/figures
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

# Reuse the exact shared visual identity from the training-set EDA so every figure
# in the chapter reads as one coherent set.
PRIMARY = "#2a6f7f"
HIGHLIGHT = "#d98b34"
HIGHLIGHT_2 = "#3d3d3d"
MUTED = "#b7c4c7"
GRID_GREY = "#d9d9d9"

CATEGORY_PALETTE: list[str] = [
    "#2a6f7f",  # deep teal
    "#d98b34",  # amber
    "#6a994e",  # sage green
    "#8e6c9b",  # plum
    "#c1574b",  # terracotta
    "#5c6b73",  # slate
]

# The four eval sets in display order, with whether each is in- or out-of-domain.
EVAL_SETS: list[dict] = [
    {
        "key": "indomain",
        "file": "test_nisqa_indomain.json",
        "label": "in-domain\n(NISQA-SIM)",
        "domain": "in-domain",
    },
    {"key": "FOR", "file": "test_FOR.json", "label": "FOR", "domain": "out-of-domain"},
    {
        "key": "LIVE",
        "file": "test_LIVE.json",
        "label": "LIVE",
        "domain": "out-of-domain",
    },
    {
        "key": "P501",
        "file": "test_P501.json",
        "label": "P501",
        "domain": "out-of-domain",
    },
]

# One colour per eval set (teal for in-domain, warm tones for the OOD sets).
SET_COLORS: dict[str, str] = {
    "indomain": "#2a6f7f",
    "FOR": "#d98b34",
    "LIVE": "#c1574b",
    "P501": "#8e6c9b",
}

# Quality sub-dimensions. Only the first three are present in every set; "dis"
# (discontinuity) is missing from the in-domain set.
SHARED_DIMS: list[tuple[str, str]] = [
    ("noi", "noisiness"),
    ("col", "coloration"),
    ("loud", "loudness"),
]
OOD_ONLY_DIM: tuple[str, str] = ("dis", "discontinuity")


def set_paper_style() -> None:
    """Apply the same cohesive seaborn theme used by the training-set EDA."""
    sns.set_theme(
        context="paper", style="whitegrid", palette=CATEGORY_PALETTE, font="serif"
    )
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


def load_records(path: Path) -> list[dict]:
    """Load a JSON array or JSONL file of eval records."""
    text = path.read_text(encoding="utf-8").strip()
    if text[:1] == "[":
        return json.loads(text)
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def collect(data_dir: Path) -> dict[str, dict]:
    """Load every eval set and extract its label arrays.

    Returns:
        Mapping from set key to ``{"mos": array, "dims": {dim: array}, "n": int}``.
    """
    out: dict[str, dict] = {}
    for spec in EVAL_SETS:
        records = load_records(data_dir / spec["file"])
        mos = np.array([float(r["mos"]) for r in records if r.get("mos") is not None])
        dims: dict[str, np.ndarray] = {}
        for dim, _ in SHARED_DIMS + [OOD_ONLY_DIM]:
            values = [float(r[dim]) for r in records if dim in r and r[dim] is not None]
            if values:
                dims[dim] = np.array(values)
        out[spec["key"]] = {"mos": mos, "dims": dims, "n": len(records)}
    return out


def fig_mos_by_set(data: dict[str, dict], figures_dir: Path) -> dict:
    """Figure 1: MOS distribution across the four eval sets (box + strip)."""
    fig, axis = plt.subplots(figsize=(7.2, 4.0))

    order_keys = [s["key"] for s in EVAL_SETS]
    mos_arrays = [data[k]["mos"] for k in order_keys]
    colors = [SET_COLORS[k] for k in order_keys]

    box = axis.boxplot(
        mos_arrays,
        vert=True,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.6},
        whiskerprops={"color": "#5c6b73", "linewidth": 1.1},
        capprops={"color": "#5c6b73", "linewidth": 1.1},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.8)

    # Light jittered points over each box for the real sample density.
    rng = np.random.default_rng(0)
    for index, (key, arr) in enumerate(zip(order_keys, mos_arrays), start=1):
        jitter = rng.uniform(-0.16, 0.16, size=len(arr))
        axis.scatter(
            np.full(len(arr), index) + jitter,
            arr,
            s=5,
            alpha=0.25,
            color="#2d2d2d",
            linewidths=0,
            zorder=3,
        )

    # Divider between the in-domain set and the OOD sets.
    axis.axvline(1.5, color="#b0b0b0", linewidth=1.0, linestyle="--", zorder=0)
    axis.text(1.0, 5.18, "in-domain", ha="center", fontsize=14, color="#2a6f7f")
    axis.text(3.0, 5.18, "out-of-domain", ha="center", fontsize=14, color="#9c5a33")

    tick_labels = [f"{s['label']}\n(n={data[s['key']]['n']})" for s in EVAL_SETS]
    axis.set_xticks(range(1, len(EVAL_SETS) + 1))
    axis.set_xticklabels(tick_labels)
    axis.set_ylabel("Mean Opinion Score (MOS)")
    axis.set_ylim(1.0, 5.35)
    axis.set_title("MOS distribution across the evaluation sets")
    axis.grid(axis="x", visible=False)
    sns.despine(ax=axis)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"eval_mos_by_set.{ext}")
    plt.close(fig)
    print("  wrote eval_mos_by_set.pdf / .png")

    return {
        key: {
            "n": int(data[key]["n"]),
            "mean": float(np.mean(data[key]["mos"])),
            "median": float(np.median(data[key]["mos"])),
            "std": float(np.std(data[key]["mos"])),
            "min": float(np.min(data[key]["mos"])),
            "max": float(np.max(data[key]["mos"])),
        }
        for key in order_keys
    }


def fig_subdims_by_set(data: dict[str, dict], figures_dir: Path) -> dict:
    """Figure 2: overall MOS plus the shared quality sub-dimension means per set.

    Overall MOS leads the chart (separated by a divider) so it reads as the headline
    metric rather than a fourth sub-dimension. Only the three sub-dimensions present
    in every set are shown; discontinuity is OOD-only and reported separately.
    """
    order_keys = [s["key"] for s in EVAL_SETS]
    # "mos" is treated as a pseudo-dimension so it shares the grouped-bar layout.
    metric_keys = ["mos"] + [d for d, _ in SHARED_DIMS]
    metric_names = ["overall\nMOS"] + [n for _, n in SHARED_DIMS]

    fig, axis = plt.subplots(figsize=(9.0, 4.6))
    n_sets = len(order_keys)
    group_width = 0.82
    bar_width = group_width / n_sets
    x_base = np.arange(len(metric_keys))

    def metric_value(key: str, metric: str) -> float:
        if metric == "mos":
            return float(np.mean(data[key]["mos"]))
        dims = data[key]["dims"]
        return float(np.mean(dims[metric])) if metric in dims else np.nan

    means: dict[str, dict[str, float]] = {}
    for set_index, key in enumerate(order_keys):
        vals = [metric_value(key, m) for m in metric_keys]
        means[key] = {m: v for m, v in zip(metric_keys, vals)}
        offsets = x_base - group_width / 2 + bar_width * (set_index + 0.5)
        bars = axis.bar(
            offsets,
            vals,
            width=bar_width * 0.9,
            color=SET_COLORS[key],
            edgecolor="white",
            linewidth=0.6,
            label=EVAL_SETS[set_index]["label"].replace("\n", " "),
            zorder=3,
        )
        # Value label on every bar (rotated to fit the narrow grouped bars).
        for rect, value in zip(bars, vals):
            axis.annotate(
                f"{value:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=11,
                color="#333333",
                zorder=4,
            )

    # Soft shaded band behind the overall-MOS group so it reads as the headline.
    axis.axvspan(-0.5, 0.5, color="#2a6f7f", alpha=0.05, zorder=0)
    axis.axvline(0.5, color="#9aa7ad", linewidth=1.0, linestyle="--", zorder=1)

    # Tighten the y-axis to the data range so the in-domain vs OOD gap is visible.
    all_vals = [v for m in means.values() for v in m.values() if not np.isnan(v)]
    y_low = max(1.0, np.floor((min(all_vals) - 0.3) * 2) / 2)
    y_high = min(5.0, np.ceil((max(all_vals) + 0.45) * 2) / 2)
    axis.set_ylim(y_low, y_high)

    axis.set_xticks(x_base)
    axis.set_xticklabels(metric_names)
    axis.set_xlim(-0.5, len(metric_keys) - 0.5)
    axis.set_ylabel("Mean rating (1-5 scale)")
    axis.set_title("MOS & Quality assessment")
    axis.legend(
        title="eval set",
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        columnspacing=1.4,
    )
    axis.grid(axis="x", visible=False)
    sns.despine(ax=axis)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"eval_subdims_by_set.{ext}")
    plt.close(fig)
    print("  wrote eval_subdims_by_set.pdf / .png")

    # Discontinuity is OOD-only: report the means but do not chart it mixed in.
    dis_key, _ = OOD_ONLY_DIM
    dis_means = {
        key: float(np.mean(data[key]["dims"][dis_key]))
        for key in order_keys
        if dis_key in data[key]["dims"]
    }
    return {
        "mos_and_shared_dim_means": means,
        "discontinuity_ood_only_means": dis_means,
        "indomain_has_discontinuity": dis_key in data["indomain"]["dims"],
    }


def main() -> None:
    """Run the eval-set analysis and write figures plus a stats JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/eval"),
        help="Directory containing the eval JSON files.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("../_papers/asa-thesis/figures"),
        help="Where to write the figures (defaults to the thesis figures dir).",
    )
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    print(f"Loading eval sets from {args.data_dir} ...")
    data = collect(args.data_dir)
    for spec in EVAL_SETS:
        print(f"  {spec['key']:>9}: n={data[spec['key']]['n']}")

    print(f"Writing figures to {args.figures_dir} ...")
    stats = {
        "mos_by_set": fig_mos_by_set(data, args.figures_dir),
        "subdims_by_set": fig_subdims_by_set(data, args.figures_dir),
    }

    stats_path = args.figures_dir / "eval_sets_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote stats to {stats_path}")
    print("\nKey numbers:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
