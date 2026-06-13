"""Predicted-vs-true MOS scatter for the global task, plus correlation stats.

This backs the claim that the off-the-shelf backbone cannot rate quality while
the fine-tuned models can. It reads the per-item eval JSONs for the three
canonical greedy runs (zero-shot / full SFT / paper-faithful DPO), pools the
four MOS test sets, and produces:

1. A three-panel predicted-vs-true MOS scatter (one panel per model) in the
   shared thesis paper/serif theme, written as PDF + PNG to the thesis figures
   dir. Each panel carries its Pearson r and the least-squares fit slope, so the
   flat zero-shot fit reads directly against the near-diagonal trained fits.
2. A stats JSON with, per model and per set plus pooled: MAE, the
   constant-mean-predictor MAE floor (always answer the set mean), Pearson r,
   Spearman rho, and the fit slope. These feed the correlation columns and the
   "worse than the mean-guess floor" sentence in the results chapter.

Run locally, no HPC (paths are relative to the code-repo root):

    python scripts/analysis/eval_pred_vs_true.py \
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

# Shared visual identity, identical to scripts/analysis/eda_eval_sets.py so the
# figure reads as one set with the rest of the chapter.
CATEGORY_PALETTE: list[str] = [
    "#2a6f7f", "#d98b34", "#6a994e", "#8e6c9b", "#c1574b", "#5c6b73",
]
GRID_GREY = "#d9d9d9"

# The three canonical greedy runs, in narrative order. Colour per model: the
# zero-shot panel is the muted slate (the "cannot do it" control), the two
# trained models take the teal/amber accent pair used elsewhere.
MODELS: list[dict] = [
    {"key": "zeroshot", "label": "Zero-shot",
     "dir": "results/evaluation/zeroshot/qwen2audio_instruct_baseline",
     "color": "#5c6b73"},
    {"key": "sft", "label": "Full SFT",
     "dir": "results/evaluation/sft/sft_full_paper_h100_eval_greedy",
     "color": "#2a6f7f"},
    {"key": "dpo", "label": "Full DPO",
     "dir": "results/evaluation/dpo/dpo_full_sft_paired_lr1e6_eval_greedy",
     "color": "#d98b34"},
]
SETS = ["FOR", "LIVE", "P501", "nisqa_indomain"]
SET_LABELS = {"FOR": "FOR", "LIVE": "LIVE", "P501": "P501",
              "nisqa_indomain": "NISQA in-domain"}


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


def reg_stats(true: np.ndarray, pred: np.ndarray) -> dict:
    """MAE, constant-mean floor, Pearson, Spearman, and pred~true fit slope."""
    out = {
        "n": int(len(true)),
        "mae": float(np.mean(np.abs(pred - true))),
        "mae_const": float(np.mean(np.abs(true - np.mean(true)))),
    }
    if np.std(pred) > 1e-9 and np.std(true) > 1e-9:
        out["pearson"] = float(stats.pearsonr(true, pred)[0])
        out["spearman"] = float(stats.spearmanr(true, pred)[0])
        out["slope"] = float(np.polyfit(true, pred, 1)[0])
    else:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")
        out["slope"] = 0.0
    return out


def fig_scatter(per_model: dict[str, dict], figures_dir: Path) -> None:
    """Three-panel predicted-vs-true MOS scatter, one panel per model."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, spec in zip(axes, MODELS):
        true = per_model[spec["key"]]["true"]
        pred = per_model[spec["key"]]["pred"]
        jt = true + rng.normal(0, 0.035, len(true))
        jp = pred + rng.normal(0, 0.035, len(pred))
        ax.scatter(jt, jp, s=9, alpha=0.22, color=spec["color"], linewidths=0,
                   zorder=3)
        ax.plot([1, 5], [1, 5], ls="--", lw=1.0, color="#b0b0b0", zorder=1,
                label="perfect")
        st = per_model[spec["key"]]["pooled"]
        if not np.isnan(st["pearson"]):
            xs = np.array([1.0, 5.0])
            ax.plot(xs, st["slope"] * xs + (np.mean(pred) - st["slope"] * np.mean(true)),
                    lw=1.8, color="#2d2d2d", zorder=4,
                    label=f"fit (slope {st['slope']:.2f})")
            ax.text(0.05, 0.94, f"$r = {st['pearson']:.2f}$", transform=ax.transAxes,
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
        fig.savefig(figures_dir / f"eval_pred_vs_true_mos.{ext}")
    plt.close(fig)
    print("  wrote eval_pred_vs_true_mos.pdf / .png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."),
                        help="Code-repo root holding results/evaluation/.")
    parser.add_argument("--figures-dir", type=Path,
                        default=Path("../_papers/asa-thesis/figures"))
    parser.add_argument("--stats-out", type=Path,
                        default=Path("results/analysis/global_task/correlation_stats.json"))
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    per_model: dict[str, dict] = {}
    stats_out: dict = {"models": {}}
    for spec in MODELS:
        pairs = load_pairs(args.data_root / spec["dir"])
        all_true = np.concatenate([pairs[s][0] for s in pairs])
        all_pred = np.concatenate([pairs[s][1] for s in pairs])
        per_model[spec["key"]] = {"true": all_true, "pred": all_pred,
                                  "pooled": reg_stats(all_true, all_pred)}
        stats_out["models"][spec["key"]] = {
            "label": spec["label"],
            "pooled": per_model[spec["key"]]["pooled"],
            "per_set": {s: reg_stats(*pairs[s]) for s in pairs},
        }

    fig_scatter(per_model, args.figures_dir)
    args.stats_out.write_text(json.dumps(stats_out, indent=2))
    print("  wrote", args.stats_out)

    # Console summary for dropping into the chapter.
    print("\n  pooled (all four sets):")
    print(f"  {'model':<12}{'MAE':>7}{'floor':>8}{'r':>7}{'rho':>7}{'slope':>7}")
    for spec in MODELS:
        p = stats_out["models"][spec["key"]]["pooled"]
        print(f"  {spec['label']:<12}{p['mae']:>7.3f}{p['mae_const']:>8.3f}"
              f"{p['pearson']:>7.3f}{p['spearman']:>7.3f}{p['slope']:>7.3f}")
    print("\n  per-set Pearson r / Spearman rho (zero-shot vs SFT):")
    for s in SETS:
        z = stats_out["models"]["zeroshot"]["per_set"][s]
        f = stats_out["models"]["sft"]["per_set"][s]
        print(f"  {SET_LABELS[s]:<18} zs r={z['pearson']:+.3f} rho={z['spearman']:+.3f}"
              f"   sft r={f['pearson']:.3f} rho={f['spearman']:.3f}")


if __name__ == "__main__":
    main()
