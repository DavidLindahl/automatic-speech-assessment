"""Merged predicted-vs-true MOS figure: scatter + binned calibration line.

This merges the two separate global-task figures (the three-panel pred-vs-true
scatter and the standalone calibration/reliability curve) into one stacked
figure. Each of the three panels (zero-shot / full SFT / full DPO) keeps its
jittered scatter and Pearson r, but the straight least-squares fit line is
replaced by the binned calibration line (mean predicted MOS per true-MOS bin),
so each panel shows directly how prediction tracks truth across the quality
range. The three panels are stacked vertically.

Reads the per-item eval JSONs for the three canonical greedy runs and pools the
four MOS test sets. Writes PDF + PNG to the thesis figures dir.

Run locally, no HPC (paths are relative to the code-repo root):

    python scripts/analysis/eval_pred_vs_true_calibrated.py \
        --figures-dir ../_papers/asa-thesis/figures
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
from scipy import stats

# Shared visual identity, identical to the sibling analysis scripts so the
# figure reads as one set with the rest of the chapter.
CATEGORY_PALETTE: list[str] = [
    "#2a6f7f", "#d98b34", "#6a994e", "#8e6c9b", "#c1574b", "#5c6b73",
]
GRID_GREY = "#d9d9d9"

# The three canonical greedy runs, in narrative order. The zero-shot panel is
# the muted slate (the "cannot do it" control), the two trained models take the
# teal/amber accent pair used elsewhere.
MODELS: list[dict] = [
    {"key": "zeroshot", "label": "Zero-shot",
     "dir": "results/evaluation/global/zeroshot/qwen2audio_instruct_baseline",
     "color": "#5c6b73"},
    {"key": "sft", "label": "Full SFT",
     "dir": "results/evaluation/global/sft/sft_full_paper_h100_eval_greedy",
     "color": "#2a6f7f"},
    {"key": "dpo", "label": "Full DPO",
     "dir": "results/evaluation/global/alld/dpo_full_sft_paired_lr1e6_eval_greedy",
     "color": "#d98b34"},
]
SETS = ["FOR", "LIVE", "P501", "nisqa_indomain"]


def set_paper_style() -> None:
    """Apply the same cohesive seaborn theme used by the eval-set EDA."""
    sns.set_theme(context="paper", style="whitegrid", palette=CATEGORY_PALETTE,
                  font="serif")
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 10.5,
        "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.labelsize": 10, "axes.labelcolor": "#222222",
        "axes.edgecolor": "#5c6b73", "axes.linewidth": 0.9,
        "text.color": "#222222", "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.color": "#444444", "ytick.color": "#444444",
        "legend.fontsize": 9, "legend.frameon": False,
        "grid.color": GRID_GREY, "grid.linewidth": 0.7, "grid.alpha": 0.7,
        "axes.axisbelow": True,
    })


def load_pairs(model_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {set: (true, pred)} arrays, dropping items with no parsed MOS."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in SETS:
        f = model_dir / f"test_{s}_results.json"
        if not f.exists():
            continue
        t, p = [], []
        for r in json.loads(f.read_text())["results"]:
            if r.get("predicted_mos") is None or r.get("mos") is None:
                continue
            t.append(float(r["mos"]))
            p.append(float(r["predicted_mos"]))
        out[s] = (np.array(t), np.array(p))
    return out


def pearson(true: np.ndarray, pred: np.ndarray) -> float:
    """Pearson r between predicted and true MOS, NaN-safe for a flat predictor."""
    if np.std(pred) > 1e-9 and np.std(true) > 1e-9:
        return float(stats.pearsonr(true, pred)[0])
    return float("nan")


def calibration_curve(true: np.ndarray, pred: np.ndarray,
                      edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean predicted MOS per true-MOS bin; returns (bin centers, means)."""
    centers = (edges[:-1] + edges[1:]) / 2
    means = np.full(len(centers), np.nan)
    idx = np.digitize(true, edges) - 1
    for b in range(len(centers)):
        sel = idx == b
        if sel.sum() >= 3:  # need a few clips for a stable mean
            means[b] = float(np.mean(pred[sel]))
    ok = ~np.isnan(means)
    return centers[ok], means[ok]


def pop(hex_color: str) -> tuple[float, float, float]:
    """Vivid version of a hex color: same hue, pushed to full saturation and a
    punchy value so the line pops against the pale dot cloud."""
    import colorsys
    r, g, b = matplotlib.colors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = min(1.0, max(s, 0.85))        # near-max saturation
    v = min(1.0, max(v, 0.80))        # bright but not white
    return colorsys.hsv_to_rgb(h, s, v)


def fig_stacked(per_model: dict[str, dict], figures_dir: Path) -> None:
    """Three side-by-side panels: scatter + binned calibration line."""
    edges = np.arange(1.0, 5.01, 0.5)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, spec in zip(axes, MODELS):
        true = per_model[spec["key"]]["true"]
        pred = per_model[spec["key"]]["pred"]
        # Jittered scatter, kept pale so the line stands out.
        jt = true + rng.normal(0, 0.035, len(true))
        jp = pred + rng.normal(0, 0.035, len(pred))
        ax.scatter(jt, jp, s=9, alpha=0.16, color=spec["color"], linewidths=0,
                   zorder=2)
        # Perfect-prediction diagonal.
        ax.plot([1, 5], [1, 5], ls="--", lw=1.0, color="#b0b0b0", zorder=1,
                label="perfect")
        # Binned calibration line: a vivid version of this panel's hue, thick,
        # with a white halo + dark marker edge so it pops off the dot cloud.
        cx, cy = calibration_curve(true, pred, edges)
        line_color = pop(spec["color"])
        ax.plot(cx, cy, "-", lw=4.0, color="white", alpha=0.9, zorder=3)
        ax.plot(cx, cy, "-o", lw=2.6, ms=5.5, color=line_color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4,
                label="calibration")
        r = per_model[spec["key"]]["pearson"]
        if not np.isnan(r):
            ax.text(0.05, 0.94, f"$r = {r:.2f}$", transform=ax.transAxes,
                    va="top", fontsize=11, color="#222222")
        ax.set_title(spec["label"])
        ax.set_xlabel("True MOS")
        ax.set_xlim(0.9, 5.1)
        ax.set_ylim(0.9, 5.1)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower right", fontsize=8)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Predicted MOS")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"eval_pred_vs_true_calibrated.{ext}")
    plt.close(fig)
    print("  wrote eval_pred_vs_true_calibrated.pdf / .png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."),
                        help="Code-repo root holding results/evaluation/.")
    parser.add_argument("--figures-dir", type=Path,
                        default=Path("../_papers/asa-thesis/figures"))
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    per_model: dict[str, dict] = {}
    for spec in MODELS:
        pairs = load_pairs(args.data_root / spec["dir"])
        all_true = np.concatenate([pairs[s][0] for s in pairs])
        all_pred = np.concatenate([pairs[s][1] for s in pairs])
        per_model[spec["key"]] = {
            "true": all_true, "pred": all_pred,
            "pearson": pearson(all_true, all_pred),
        }

    fig_stacked(per_model, args.figures_dir)


if __name__ == "__main__":
    main()
