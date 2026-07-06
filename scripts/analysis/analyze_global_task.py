"""Deep analysis of the MOS global task: zero-shot vs SFT vs DPO.

Reads the per-item eval JSONs for the three canonical models, collapses across
the four MOS test sets, and computes a battery of "is it learning the domain or
guessing?" diagnostics:

- regression accuracy (MAE, MSE) and rank/linear correlation (Pearson, Spearman)
- output diversity / numeric mode collapse (unique predicted MOS, top-value share)
- the constant-predictor floor (MAE of always-predict-the-mean) per model/set
- error vs true-MOS regime (does error blow up at the extremes?)
- calibration (binned reliability, regression-to-the-mean slope)
- caption grounding (caption diversity, length, sentiment-vs-MOS coupling)

Writes a stats JSON and a set of standalone PNG figures under
results/analysis/global_task/.

Run from the repo root with the project venv active:
    python scripts/eval/analyze_global_task.py
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "results" / "evaluation"
OUT = REPO / "results" / "analysis" / "global_task"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Canonical thesis runs (greedy decoding) ----------------------------------
MODELS = {
    "zeroshot": EVAL / "global" / "zeroshot" / "qwen2audio_instruct_baseline",
    "sft": EVAL / "global" / "sft" / "sft_full_paper_h100_eval_greedy",
    "dpo": EVAL / "global" / "alld" / "dpo_full_sft_paired_lr1e6_eval_greedy",
}
MODEL_LABELS = {
    "zeroshot": "Zero-shot",
    "sft": "Full SFT",
    "dpo": "Full DPO (paired)",
}
MODEL_COLORS = {"zeroshot": "#9aa0a6", "sft": "#1a73e8", "dpo": "#e8710a"}
SETS = ["FOR", "LIVE", "P501", "nisqa_indomain"]
SET_LABELS = {
    "FOR": "FOR",
    "LIVE": "LIVE",
    "P501": "P501",
    "nisqa_indomain": "NISQA in-domain",
}

# Tiny sentiment lexicon for caption grounding (positive minus negative hits) -
POS = [
    "clean", "clear", "continuous", "natural", "high", "good", "suitable",
    "minimal", "excellent", "pleasant", "smooth", "intelligible",
]
NEG = [
    "noise", "noisy", "distortion", "distorted", "discontinu", "muffled",
    "degrad", "poor", "low", "artifact", "clipping", "interrupt", "background",
    "tinny", "robotic", "unnatural", "harsh", "echo", "reverber",
]


@dataclass
class Item:
    set_name: str
    true_mos: float
    pred_mos: float | None
    sub: dict[str, float]
    pred_text: str
    ref_text: str


@dataclass
class ModelData:
    key: str
    items: list[Item] = field(default_factory=list)

    def arr(self, scored_only: bool = True):
        """Return (true, pred) numpy arrays. scored_only drops unparsed preds."""
        t, p = [], []
        for it in self.items:
            if scored_only and it.pred_mos is None:
                continue
            t.append(it.true_mos)
            p.append(it.pred_mos if it.pred_mos is not None else np.nan)
        return np.array(t, float), np.array(p, float)


def load() -> dict[str, ModelData]:
    data: dict[str, ModelData] = {}
    for key, d in MODELS.items():
        md = ModelData(key=key)
        for s in SETS:
            f = d / f"test_{s}_results.json"
            if not f.exists():
                continue
            for r in json.load(open(f))["results"]:
                md.items.append(
                    Item(
                        set_name=s,
                        true_mos=float(r["mos"]),
                        pred_mos=(None if r.get("predicted_mos") is None
                                  else float(r["predicted_mos"])),
                        sub={k: r.get(k) for k in ("noi", "col", "loud", "dis")},
                        pred_text=str(r.get("predicted_response", "")),
                        ref_text=str(r.get("response", "")),
                    )
                )
        data[key] = md
    return data


def sentiment(text: str) -> int:
    t = text.lower()
    pos = sum(t.count(w) for w in POS)
    neg = sum(t.count(w) for w in NEG)
    return pos - neg


def reg_stats(true: np.ndarray, pred: np.ndarray) -> dict:
    mask = ~np.isnan(pred)
    t, p = true[mask], pred[mask]
    if len(t) < 3:
        return {}
    err = p - t
    out = {
        "n": int(len(t)),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "bias": float(np.mean(err)),  # +ve = over-prediction
        "pred_std": float(np.std(p)),
        "true_std": float(np.std(t)),
        "pred_mean": float(np.mean(p)),
        "true_mean": float(np.mean(t)),
    }
    # constant-predictor floor: always predict the test-set mean true MOS
    out["mae_const"] = float(np.mean(np.abs(t - np.mean(t))))
    # correlations (need variance on both sides)
    if np.std(p) > 1e-9 and np.std(t) > 1e-9:
        out["pearson"] = float(stats.pearsonr(t, p)[0])
        out["spearman"] = float(stats.spearmanr(t, p)[0])
        # slope of pred ~ true (1.0 = perfect tracking, 0 = constant guess)
        sl, ic = np.polyfit(t, p, 1)
        out["slope"] = float(sl)
        out["intercept"] = float(ic)
    else:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")
        out["slope"] = 0.0
        out["intercept"] = float(np.mean(p))
    # variance-ratio: how much of the true spread the predictions reproduce
    out["std_ratio"] = (out["pred_std"] / out["true_std"]
                        if out["true_std"] > 1e-9 else float("nan"))
    return out


def diversity_stats(md: ModelData) -> dict:
    out = {}
    for s in SETS + ["ALL"]:
        items = (md.items if s == "ALL"
                 else [it for it in md.items if it.set_name == s])
        if not items:
            continue
        preds = [it.pred_mos for it in items if it.pred_mos is not None]
        caps = [it.pred_text.strip() for it in items if it.pred_text.strip()]
        c_mos = Counter(preds)
        c_cap = Counter(caps)
        out[s] = {
            "n": len(items),
            "n_scored": len(preds),
            "unique_mos": len(c_mos),
            "top_mos": (c_mos.most_common(1)[0] if c_mos else (None, 0)),
            "top_mos_share": (c_mos.most_common(1)[0][1] / len(preds)
                              if preds else float("nan")),
            "unique_captions": len(c_cap),
            "top_caption_share": (c_cap.most_common(1)[0][1] / len(caps)
                                  if caps else float("nan")),
            "mean_cap_len": (float(np.mean([len(x.split()) for x in caps]))
                             if caps else float("nan")),
        }
    return out


def main() -> None:
    data = load()
    stats_out: dict = {"models": {}, "sets": SETS}

    # ---- per-model, per-set + pooled regression / diversity ----
    for key, md in data.items():
        m = {"label": MODEL_LABELS[key], "per_set": {}, "diversity": diversity_stats(md)}
        for s in SETS:
            t, p = ModelData(key, [it for it in md.items if it.set_name == s]).arr()
            m["per_set"][s] = reg_stats(t, p)
        t_all, p_all = md.arr()
        m["pooled"] = reg_stats(t_all, p_all)
        # caption grounding: corr(predicted_mos, caption sentiment)
        sent, pm, tm = [], [], []
        for it in md.items:
            if it.pred_mos is None or not it.pred_text.strip():
                continue
            sent.append(sentiment(it.pred_text))
            pm.append(it.pred_mos)
            tm.append(it.true_mos)
        sent, pm, tm = np.array(sent, float), np.array(pm, float), np.array(tm, float)
        if len(sent) > 3 and np.std(sent) > 1e-9 and np.std(pm) > 1e-9:
            m["sentiment_vs_predmos_r"] = float(stats.pearsonr(sent, pm)[0])
        else:
            m["sentiment_vs_predmos_r"] = float("nan")
        if len(sent) > 3 and np.std(sent) > 1e-9 and np.std(tm) > 1e-9:
            m["sentiment_vs_truemos_r"] = float(stats.pearsonr(sent, tm)[0])
        else:
            m["sentiment_vs_truemos_r"] = float("nan")
        stats_out["models"][key] = m

    (OUT / "stats.json").write_text(json.dumps(stats_out, indent=2))
    print("wrote", OUT / "stats.json")

    # =====================================================================
    # FIGURES
    # =====================================================================
    plt.rcParams.update({
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 130, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "axes.axisbelow": True,
    })

    _fig_pooled_bars(stats_out)
    _fig_pred_distributions(data)
    _fig_calibration(data)
    _fig_scatter(data)
    _fig_error_by_regime(data)
    _fig_diversity(stats_out)
    _fig_sentiment(data)
    print("figures ->", FIG)


# --------------------------------------------------------------------------
# Individual figure builders
# --------------------------------------------------------------------------
def _bar_positions(n_groups, n_models, width=0.26):
    x = np.arange(n_groups)
    offs = (np.arange(n_models) - (n_models - 1) / 2) * width
    return x, offs, width


def _fig_pooled_bars(s: dict) -> None:
    metrics = [("mae", "MAE (lower better)", False),
               ("pearson", "Pearson r (higher better)", True),
               ("spearman", "Spearman rho (higher better)", True),
               ("std_ratio", "Pred/True std ratio (1.0 ideal)", True)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    keys = list(MODELS)
    for ax, (mk, title, _hi) in zip(axes, metrics):
        vals = [s["models"][k]["pooled"].get(mk, np.nan) for k in keys]
        bars = ax.bar([MODEL_LABELS[k] for k in keys], vals,
                      color=[MODEL_COLORS[k] for k in keys])
        ax.set_title(title, fontsize=10.5)
        ax.tick_params(axis="x", rotation=18)
        if mk == "std_ratio":
            ax.axhline(1.0, ls="--", lw=1, color="#444")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:.2f}" if not np.isnan(v) else "-",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Pooled over all four MOS test sets", y=1.04, fontsize=12)
    fig.savefig(FIG / "01_pooled_metrics.png")
    plt.close(fig)


def _fig_pred_distributions(data: dict) -> None:
    """Histogram of predicted MOS overlaid on the true-MOS distribution (pooled)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    bins = np.arange(1.0, 5.2, 0.2)
    for ax, key in zip(axes, MODELS):
        md = data[key]
        t = np.array([it.true_mos for it in md.items])
        p = np.array([it.pred_mos for it in md.items if it.pred_mos is not None])
        ax.hist(t, bins=bins, color="#bbb", alpha=0.7, label="True MOS",
                edgecolor="white", linewidth=0.4)
        ax.hist(p, bins=bins, histtype="step", color=MODEL_COLORS[key],
                linewidth=2.2, label="Predicted MOS")
        ax.set_title(MODEL_LABELS[key])
        ax.set_xlabel("MOS")
        ax.legend(fontsize=9, loc="upper left")
    axes[0].set_ylabel("count")
    fig.suptitle("Predicted vs true MOS distribution (all sets pooled)",
                 y=1.02, fontsize=12)
    fig.savefig(FIG / "02_pred_distribution.png")
    plt.close(fig)


def _fig_calibration(data: dict) -> None:
    """Binned reliability: mean predicted MOS per true-MOS bin."""
    fig, ax = plt.subplots(figsize=(6.4, 6))
    edges = np.arange(1.0, 5.01, 0.5)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot([1, 5], [1, 5], ls="--", color="#444", lw=1.2, label="perfect")
    for key in MODELS:
        md = data[key]
        t = np.array([it.true_mos for it in md.items if it.pred_mos is not None])
        p = np.array([it.pred_mos for it in md.items if it.pred_mos is not None])
        means, ses = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (t >= lo) & (t < hi)
            if m.sum() >= 5:
                means.append(np.mean(p[m]))
                ses.append(np.std(p[m]) / np.sqrt(m.sum()))
            else:
                means.append(np.nan)
                ses.append(np.nan)
        ax.errorbar(centers, means, yerr=ses, marker="o", capsize=3,
                    color=MODEL_COLORS[key], label=MODEL_LABELS[key], lw=2)
    ax.set_xlabel("True MOS (binned)")
    ax.set_ylabel("Mean predicted MOS")
    ax.set_title("Calibration / reliability curve\n(flat line = ignores the audio)")
    ax.legend()
    ax.set_xlim(1, 5)
    ax.set_ylim(1, 5)
    fig.savefig(FIG / "03_calibration.png")
    plt.close(fig)


def _fig_scatter(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, key in zip(axes, MODELS):
        md = data[key]
        t = np.array([it.true_mos for it in md.items if it.pred_mos is not None])
        p = np.array([it.pred_mos for it in md.items if it.pred_mos is not None])
        jt = t + rng.normal(0, 0.03, len(t))
        jp = p + rng.normal(0, 0.03, len(p))
        ax.scatter(jt, jp, s=8, alpha=0.28, color=MODEL_COLORS[key],
                   edgecolors="none")
        ax.plot([1, 5], [1, 5], ls="--", color="#444", lw=1)
        if np.std(p) > 1e-9:
            sl, ic = np.polyfit(t, p, 1)
            xs = np.array([1, 5])
            ax.plot(xs, sl * xs + ic, color="#111", lw=1.6,
                    label=f"fit: slope {sl:.2f}")
            r = stats.pearsonr(t, p)[0]
            ax.text(0.04, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                    va="top", fontsize=11, fontweight="bold")
            ax.legend(loc="lower right", fontsize=9)
        ax.set_title(MODEL_LABELS[key])
        ax.set_xlabel("True MOS")
        ax.set_xlim(1, 5)
        ax.set_ylim(1, 5)
    axes[0].set_ylabel("Predicted MOS")
    fig.suptitle("Predicted vs true MOS (jittered, all sets pooled)",
                 y=1.02, fontsize=12)
    fig.savefig(FIG / "04_scatter.png")
    plt.close(fig)


def _fig_error_by_regime(data: dict) -> None:
    """Mean |error| by true-MOS bin: where does each model break down?"""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    edges = np.arange(1.0, 5.01, 0.5)
    centers = (edges[:-1] + edges[1:]) / 2
    width = 0.13
    for i, key in enumerate(MODELS):
        md = data[key]
        t = np.array([it.true_mos for it in md.items if it.pred_mos is not None])
        p = np.array([it.pred_mos for it in md.items if it.pred_mos is not None])
        e = np.abs(p - t)
        maes = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (t >= lo) & (t < hi)
            maes.append(np.mean(e[m]) if m.sum() >= 3 else np.nan)
        ax.bar(centers + (i - 1) * width, maes, width=width,
               color=MODEL_COLORS[key], label=MODEL_LABELS[key])
    ax.set_xlabel("True MOS bin")
    ax.set_ylabel("Mean |MOS error|")
    ax.set_title("Where the error lives: MAE by true-quality regime")
    ax.legend()
    fig.savefig(FIG / "05_error_by_regime.png")
    plt.close(fig)


def _fig_diversity(s: dict) -> None:
    """Unique predicted MOS values + top-value share, pooled."""
    keys = list(MODELS)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    uniq = [s["models"][k]["diversity"]["ALL"]["unique_mos"] for k in keys]
    share = [s["models"][k]["diversity"]["ALL"]["top_mos_share"] * 100 for k in keys]
    b1 = a1.bar([MODEL_LABELS[k] for k in keys], uniq,
                color=[MODEL_COLORS[k] for k in keys])
    a1.set_title("Distinct predicted MOS values (pooled, 1380 clips)")
    a1.set_ylabel("# distinct MOS values")
    for b, v in zip(b1, uniq):
        a1.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center",
                va="bottom", fontsize=10)
    b2 = a2.bar([MODEL_LABELS[k] for k in keys], share,
                color=[MODEL_COLORS[k] for k in keys])
    a2.set_title("Share taken by single most-frequent MOS value")
    a2.set_ylabel("% of predictions")
    for b, v in zip(b2, share):
        a2.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}%", ha="center",
                va="bottom", fontsize=10)
    fig.suptitle("Numeric mode collapse on the rating axis", y=1.03, fontsize=12)
    fig.savefig(FIG / "06_diversity.png")
    plt.close(fig)


def _fig_sentiment(data: dict) -> None:
    """Predicted MOS vs caption sentiment: is the number tied to the words?"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True, sharey=True)
    rng = np.random.default_rng(1)
    for ax, key in zip(axes, MODELS):
        md = data[key]
        sent, pm = [], []
        for it in md.items:
            if it.pred_mos is None or not it.pred_text.strip():
                continue
            sent.append(sentiment(it.pred_text))
            pm.append(it.pred_mos)
        sent = np.array(sent, float)
        pm = np.array(pm, float)
        ax.scatter(sent + rng.normal(0, 0.08, len(sent)), pm,
                   s=10, alpha=0.25, color=MODEL_COLORS[key], edgecolors="none")
        if np.std(sent) > 1e-9 and np.std(pm) > 1e-9:
            r = stats.pearsonr(sent, pm)[0]
            sl, ic = np.polyfit(sent, pm, 1)
            xs = np.array([sent.min(), sent.max()])
            ax.plot(xs, sl * xs + ic, color="#111", lw=1.6)
            ax.text(0.04, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                    va="top", fontweight="bold")
        ax.set_title(MODEL_LABELS[key])
        ax.set_xlabel("Caption sentiment (pos - neg word count)")
    axes[0].set_ylabel("Predicted MOS")
    fig.suptitle("Does the spoken verdict match the number? "
                 "(caption sentiment vs predicted MOS)", y=1.02, fontsize=12)
    fig.savefig(FIG / "07_sentiment_vs_mos.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
