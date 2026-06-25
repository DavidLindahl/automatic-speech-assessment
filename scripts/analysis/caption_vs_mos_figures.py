"""Build the headline figure and results table for the caption-vs-MOS analysis.

Reads the per-run summary JSONs written by ``caption_vs_mos.py`` and produces:

- Fig 1: scatter of mean |rho(BERTScore, MOS error)| against mean MOS error,
  one point per model x decoder. The headline image: points march toward the
  origin as the model matures (high error + strong coupling -> low error + weak
  coupling).
- Table 1: per-test-set Spearman (BERTScore) with significance marks, as LaTeX.

Run after caption_vs_mos.py has written the 8 stripped summaries:
    python scripts/analysis/caption_vs_mos_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ANALYSIS_DIR = Path("results/analysis/caption_vs_mos")
FIG_DIR = Path("reports/figures")

# (json stem, display label, base family, decoder). Order = training maturity.
MODELS = [
    ("warmup_sft_greedy", "Warmup SFT", "warmup", "greedy"),
    ("warmup_sft_sampled", "Warmup SFT", "warmup", "sampled"),
    ("full_sft_greedy", "Full SFT", "full", "greedy"),
    ("full_sft_sampled", "Full SFT", "full", "sampled"),
    ("warmup_dpo_greedy", "Warmup-DPO", "warmup", "greedy"),
    ("warmup_dpo_sampled", "Warmup-DPO", "warmup", "sampled"),
    ("full_dpo_greedy", "Full-DPO", "full", "greedy"),
    ("full_dpo_sampled", "Full-DPO", "full", "sampled"),
]

COLORS = {
    "Warmup SFT": "#c0392b",
    "Full SFT": "#2e86c1",
    "Warmup-DPO": "#e67e22",
    "Full-DPO": "#27ae60",
}
SET_ORDER = ["FOR", "LIVE", "P501"]

# Per-point label offset (points) to avoid overlap in the dense bottom-left
# cluster of the four strong models. Keyed by json stem.
LABEL_OFFSETS = {
    "warmup_sft_greedy": (8, 6),
    "warmup_sft_sampled": (-10, 10),
    "warmup_dpo_greedy": (8, 8),
    "warmup_dpo_sampled": (8, 8),
    "full_sft_greedy": (10, -22),
    "full_sft_sampled": (12, 10),
    "full_dpo_greedy": (-30, 26),
    "full_dpo_sampled": (-58, -16),
}


def _mos_error_mean(entry: dict) -> float:
    """Pull the mean MOS error out of the stored distribution string."""
    return float(entry["dist_mos_error"].split("mean=")[1].split()[0])


def _set_name(entry: dict) -> str:
    return entry["test_set"].replace("test_", "").replace("_results", "")


def _stars(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return " (n.s.)"


def load_summary(stem: str) -> Optional[dict]:
    path = ANALYSIS_DIR / f"{stem}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_figure() -> None:
    """Fig 1: mean |rho| vs mean MOS error, 8 points."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    seen_labels: set[str] = set()

    for stem, label, _family, decoder in MODELS:
        data = load_summary(stem)
        if data is None:
            continue
        errs = [_mos_error_mean(e) for e in data["summary"]]
        rhos = [abs(e["spearman_bertscore_vs_mos_error"]) for e in data["summary"]]
        x = sum(errs) / len(errs)
        y = sum(rhos) / len(rhos)

        marker = "o" if decoder == "greedy" else "^"
        legend_label = label if label not in seen_labels else None
        seen_labels.add(label)
        ax.scatter(
            x,
            y,
            s=130,
            c=COLORS[label],
            marker=marker,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
            label=legend_label,
        )
        dx, dy = LABEL_OFFSETS.get(stem, (8, 6))
        ax.annotate(
            f"{label}\n({decoder})",
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.5,
            color="#333333",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5),
        )

    ax.set_xlim(left=0.15)
    ax.set_xlabel("Mean absolute MOS error (over FOR / LIVE / P501)")
    ax.set_ylabel(r"Mean $|\rho|$  (BERTScore-F1 vs MOS error)")
    ax.set_title(
        "Caption-MOS link vs. model MOS error", fontsize=11
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    # circle = greedy, triangle = sampled (decoder legend, manual)
    from matplotlib.lines import Line2D

    decoder_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=9, label="greedy"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=9, label="sampled"),
    ]
    leg1 = ax.legend(loc="lower right", fontsize=8, title="Model")
    ax.add_artist(leg1)
    ax.legend(handles=decoder_handles, loc="upper left", fontsize=8,
              title="Decoder")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "caption_vs_mos_coupling.pdf"
    out_png = FIG_DIR / "caption_vs_mos_coupling.png"
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_pdf} and {out_png}")


def build_table() -> None:
    """Table 1: per-test-set BERTScore Spearman, as LaTeX, greedy + sampled."""
    rows = []
    for stem, label, _family, decoder in MODELS:
        data = load_summary(stem)
        if data is None:
            continue
        by_set = {_set_name(e): e for e in data["summary"]}
        cells = []
        for s in SET_ORDER:
            e = by_set.get(s)
            if e is None:
                cells.append("--")
                continue
            rho = e["spearman_bertscore_vs_mos_error"]
            cells.append(f"{rho:+.2f}{_stars(e.get('p_bertscore'))}")
        mean_err = sum(_mos_error_mean(e) for e in data["summary"]) / len(
            data["summary"]
        )
        rows.append((label, decoder, mean_err, cells))

    lines = [
        r"\begin{tabular}{llrccc}",
        r"\toprule",
        r"Model & Decoder & Mean MOS err & "
        r"$\rho_{\text{FOR}}$ & $\rho_{\text{LIVE}}$ & $\rho_{\text{P501}}$ \\",
        r"\midrule",
    ]
    for label, decoder, mean_err, cells in rows:
        lines.append(
            f"{label} & {decoder} & {mean_err:.2f} & "
            f"{cells[0]} & {cells[1]} & {cells[2]} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "% rho = Spearman(BERTScore-F1, |MOS error|), MOS number stripped.",
        "% *** p<.001, ** p<.01, * p<.05, (n.s.) otherwise.",
    ]
    out = ANALYSIS_DIR / "table1_spearman.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print("\n".join(lines))


# Reported eval directories for the 4-model caption-quality table.
EVAL_DIRS = [
    ("Warmup SFT", "greedy", "global/sft/sft_warmup_paper_half_h100_eval_greedy"),
    ("Warmup SFT", "sampled", "global/sft/sft_warmup_paper_half_h100_eval_sampled"),
    ("Full SFT", "greedy", "global/sft/sft_full_paper_h100_eval_greedy"),
    ("Full SFT", "sampled", "global/sft/sft_full_paper_h100_eval_sampled"),
    ("Warmup-DPO", "greedy", "global/alld/dpo_paper_half_h100_lr1e6_delimiterfix_eval_greedy"),
    ("Warmup-DPO", "sampled", "global/alld/dpo_paper_half_h100_lr1e6_delimiterfix_eval_sampled"),
    ("Full-DPO", "greedy", "global/alld/dpo_full_sft_paired_lr1e6_eval_greedy"),
    ("Full-DPO", "sampled", "global/alld/dpo_full_sft_paired_lr1e6_eval_sampled"),
]
EVAL_ROOT = Path("results/evaluation")
TABLE_SETS = ["FOR", "LIVE", "P501", "nisqa_indomain"]


def build_caption_table() -> None:
    """Table 2: BLEU / ROUGE-L / BERTScore per model, averaged over test sets."""
    rows = []
    for label, decoder, d in EVAL_DIRS:
        bleu, rougel, bert = [], [], []
        for s in TABLE_SETS:
            path = EVAL_ROOT / d / f"test_{s}_results.json"
            if not path.exists():
                continue
            m = json.loads(path.read_text())["metrics"]
            if "bleu" in m:
                bleu.append(m["bleu"])
            if "rougeL_f" in m:
                rougel.append(m["rougeL_f"])
            if "bertscore_f1" in m:
                bert.append(m["bertscore_f1"])
        if not bleu:
            continue
        rows.append(
            (
                label,
                decoder,
                sum(bleu) / len(bleu),
                sum(rougel) / len(rougel) if rougel else float("nan"),
                sum(bert) / len(bert) if bert else float("nan"),
            )
        )

    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model & Decoder & BLEU $\uparrow$ & ROUGE-L $\uparrow$ "
        r"& BERTScore $\uparrow$ \\",
        r"\midrule",
    ]
    for label, decoder, bleu, rougel, bert in rows:
        lines.append(
            f"{label} & {decoder} & {bleu:.1f} & {rougel:.2f} & {bert:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "% Mean over FOR / LIVE / P501 / NISQA-indomain.",
        "% BLEU corpus [0,100]; ROUGE-L and BERTScore-F1 sample-mean [0,1].",
    ]
    out = ANALYSIS_DIR / "table2_caption_metrics.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    build_figure()
    build_table()
    build_caption_table()
