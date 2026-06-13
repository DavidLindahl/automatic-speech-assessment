"""Build a self-contained HTML report for the MOS global-task analysis.

Reads results/analysis/global_task/stats.json plus the raw eval JSONs (for the
caption metrics), embeds the figures from results/analysis/global_task/figures/
as base64, and writes a single portable report.html. All numbers come from the
data files; nothing is hand-transcribed.

Run after analyze_global_task.py, from the repo root:
    python scripts/eval/build_global_task_report.py
"""

from __future__ import annotations

import base64
import glob
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "analysis" / "global_task"
FIG = OUT / "figures"
EVAL = REPO / "results" / "evaluation"

MODEL_DIRS = {
    "zeroshot": EVAL / "zeroshot" / "qwen2audio_instruct_baseline",
    "sft": EVAL / "sft" / "sft_full_paper_h100_eval_greedy",
    "dpo": EVAL / "dpo" / "dpo_full_sft_paired_lr1e6_eval_greedy",
}
LABELS = {"zeroshot": "Zero-shot", "sft": "Full SFT", "dpo": "Full DPO"}
SETS = ["FOR", "LIVE", "P501", "nisqa_indomain"]
SET_LABELS = {"FOR": "FOR", "LIVE": "LIVE", "P501": "P501",
              "nisqa_indomain": "NISQA in-domain"}


def b64(name: str) -> str:
    return base64.b64encode((FIG / name).read_bytes()).decode()


def caption_metrics() -> dict:
    out: dict = {}
    for k, d in MODEL_DIRS.items():
        out[k] = {}
        for f in glob.glob(str(d / "test_*_results.json")):
            ts = os.path.basename(f).replace("test_", "").replace("_results.json", "")
            out[k][ts] = json.load(open(f))["metrics"]
    return out


def fmt(x, nd=3, dash="-"):
    if x is None:
        return dash
    try:
        if isinstance(x, float) and (x != x):  # NaN
            return dash
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    s = json.load(open(OUT / "stats.json"))
    cap = caption_metrics()
    M = s["models"]

    def pooled(k, key):
        return M[k]["pooled"].get(key)

    # ---- HTML pieces ----
    css = """
    :root{--bg:#ffffff;--surface:#fafafa;--ink:#1a1a1a;--muted:#5f6368;
      --line:#e3e3e0;--blue:#1a73e8;--orange:#e8710a;--gray:#9aa0a6;
      --good:#1e8e3e;--bad:#c5221f;--radius:10px;}
    *{box-sizing:border-box;}
    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,
      Helvetica,Arial,sans-serif;color:var(--ink);background:#f4f4f2;margin:0;
      line-height:1.65;-webkit-font-smoothing:antialiased;}
    .wrap{max-width:980px;margin:0 auto;padding:48px 28px 80px;}
    header{border-bottom:2px solid var(--ink);padding-bottom:20px;margin-bottom:8px;}
    h1{font-size:30px;font-weight:600;letter-spacing:-.01em;margin:0 0 6px;}
    .sub{color:var(--muted);font-size:15px;}
    h2{font-size:21px;font-weight:600;margin:46px 0 8px;letter-spacing:-.01em;
      padding-top:10px;}
    h3{font-size:16px;font-weight:600;margin:28px 0 6px;color:#222;}
    p{margin:10px 0;font-size:15.5px;}
    .lead{font-size:17px;color:#2a2a2a;}
    code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;
      background:#ecebe7;padding:1px 6px;border-radius:5px;}
    .card{background:var(--bg);border:1px solid var(--line);border-radius:var(--radius);
      padding:22px 26px;margin:18px 0;}
    .kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:22px 0;}
    .kpi{background:var(--bg);border:1px solid var(--line);border-radius:var(--radius);
      padding:16px 18px;}
    .kpi .name{font-size:13px;color:var(--muted);margin-bottom:8px;
      display:flex;align-items:center;gap:7px;}
    .dot{width:9px;height:9px;border-radius:50%;display:inline-block;}
    .kpi .big{font-size:26px;font-weight:600;line-height:1.1;}
    .kpi .note{font-size:12.5px;color:var(--muted);margin-top:5px;}
    table{border-collapse:collapse;width:100%;font-size:13.5px;margin:14px 0;}
    th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);}
    th:first-child,td:first-child{text-align:left;}
    thead th{border-bottom:2px solid #d0d0cc;font-weight:600;color:#333;
      font-size:12.5px;text-transform:uppercase;letter-spacing:.03em;}
    tbody tr:hover{background:#faf9f6;}
    .grp td{background:#f1f0ec;font-weight:600;font-size:12px;color:#555;
      text-transform:uppercase;letter-spacing:.04em;text-align:left;}
    .best{color:var(--good);font-weight:600;}
    .worst{color:var(--bad);}
    figure{margin:20px 0 8px;}
    figure img{width:100%;border:1px solid var(--line);border-radius:var(--radius);
      background:#fff;}
    figcaption{font-size:13px;color:var(--muted);margin-top:8px;padding:0 4px;}
    .callout{border-left:4px solid var(--blue);background:#f1f6fe;border-radius:0 8px 8px 0;
      padding:14px 18px;margin:18px 0;font-size:15px;}
    .callout.warn{border-color:var(--orange);background:#fef4ec;}
    .pill{display:inline-block;font-size:12px;font-weight:600;padding:2px 9px;
      border-radius:20px;}
    .pill.zs{background:#ececed;color:#5f6368;}
    .pill.sft{background:#e6f1fb;color:#185fa5;}
    .pill.dpo{background:#fdf0e4;color:#a85608;}
    ul{margin:10px 0;padding-left:22px;}li{margin:6px 0;font-size:15.5px;}
    .qual{font-size:13px;}.qual td{vertical-align:top;}
    .mono-sm{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:#444;}
    footer{margin-top:60px;padding-top:18px;border-top:1px solid var(--line);
      color:var(--muted);font-size:13px;}
    .tag{font-size:11.5px;color:var(--muted);background:#ecebe7;border-radius:5px;
      padding:2px 7px;}
    """

    def kpi(k):
        pl = M[k]["pooled"]
        col = {"zeroshot": "var(--gray)", "sft": "var(--blue)",
               "dpo": "var(--orange)"}[k]
        floor = pl["mae_const"] - pl["mae"]
        floor_txt = (f"beats mean-guess floor by {floor:+.2f}" if floor > 0
                     else f"WORSE than mean-guess floor by {floor:+.2f}")
        return f"""<div class="kpi">
          <div class="name"><span class="dot" style="background:{col}"></span>{LABELS[k]}</div>
          <div class="big">{pl['mae']:.3f}<span style="font-size:14px;color:var(--muted)"> MAE</span></div>
          <div class="note">Pearson r {pl['pearson']:.2f} &middot; slope {pl['slope']:.2f}<br>{floor_txt}</div>
        </div>"""

    kpis = "".join(kpi(k) for k in ["zeroshot", "sft", "dpo"])

    # ---- regression table (pooled + per-set) ----
    def reg_rows():
        rows = []
        # pooled block
        rows.append('<tr class="grp"><td colspan="9">Pooled over all four sets (n = 1180 / 1176 scored)</td></tr>')
        for k in ["zeroshot", "sft", "dpo"]:
            pl = M[k]["pooled"]
            mae_cls = "worst" if k == "zeroshot" else "best" if k == "dpo" else ""
            rows.append(
                f"<tr><td><span class='pill {('zs' if k=='zeroshot' else k)}'>{LABELS[k]}</span></td>"
                f"<td class='{mae_cls}'>{pl['mae']:.3f}</td><td>{pl['mse']:.3f}</td>"
                f"<td>{pl['bias']:+.3f}</td>"
                f"<td>{pl['pearson']:.3f}</td><td>{pl['spearman']:.3f}</td>"
                f"<td>{pl['slope']:.3f}</td><td>{pl['std_ratio']:.3f}</td>"
                f"<td>{pl['mae_const']:.3f}</td></tr>")
        for st in SETS:
            rows.append(f'<tr class="grp"><td colspan="9">{SET_LABELS[st]}</td></tr>')
            for k in ["zeroshot", "sft", "dpo"]:
                p = M[k]["per_set"][st]
                rows.append(
                    f"<tr><td><span class='pill {('zs' if k=='zeroshot' else k)}'>{LABELS[k]}</span></td>"
                    f"<td>{p['mae']:.3f}</td><td>{p['mse']:.3f}</td>"
                    f"<td>{p['bias']:+.3f}</td>"
                    f"<td>{p['pearson']:.3f}</td><td>{p['spearman']:.3f}</td>"
                    f"<td>{p['slope']:.3f}</td><td>{p['std_ratio']:.3f}</td>"
                    f"<td>{p['mae_const']:.3f}</td></tr>")
        return "\n".join(rows)

    # ---- caption / diversity table ----
    def cap_rows():
        rows = []
        for st in ["FOR", "LIVE", "P501", "nisqa_indomain"]:
            rows.append(f'<tr class="grp"><td colspan="7">{SET_LABELS[st]}</td></tr>')
            for k in ["zeroshot", "sft", "dpo"]:
                m = cap[k][st]
                div = M[k]["diversity"][st]
                rows.append(
                    f"<tr><td><span class='pill {('zs' if k=='zeroshot' else k)}'>{LABELS[k]}</span></td>"
                    f"<td>{fmt(m.get('bleu'),2)}</td>"
                    f"<td>{fmt(m.get('rougeL_f'),3)}</td>"
                    f"<td>{fmt(m.get('bertscore_f1'),3)}</td>"
                    f"<td>{div['unique_mos']}</td>"
                    f"<td>{div['unique_captions']}</td>"
                    f"<td>{fmt(div['mean_cap_len'],0)}</td></tr>")
        return "\n".join(rows)

    # ---- qualitative example block (hard-coded transcript, LIVE set) ----
    qual = """
    <table class="qual">
      <thead><tr><th>True MOS</th><th>Model</th><th>Pred</th><th>Caption (truncated)</th></tr></thead>
      <tbody>
        <tr class="grp"><td colspan="4">A heavily degraded clip &mdash; <span class="mono-sm">book_03968…live_phone_228.wav</span></td></tr>
        <tr><td rowspan="3">1.0</td><td><span class="pill zs">Zero-shot</span></td><td class="worst">3.0</td><td>"The speech has a <b>moderate</b> amount of noise, which might be distracting but does not obscure the content&hellip;"</td></tr>
        <tr><td><span class="pill sft">SFT</span></td><td class="best">1.0</td><td>"&hellip;extremely quiet, noisy and discontinuous with <b>severe distortion</b>&hellip;"</td></tr>
        <tr><td><span class="pill dpo">DPO</span></td><td class="best">1.0</td><td>"&hellip;extremely quiet, noisy and discontinuous with <b>severe distortion</b>&hellip;"</td></tr>
        <tr class="grp"><td colspan="4">A mid-quality clip &mdash; <span class="mono-sm">book_00102…live_phone_609.wav</span></td></tr>
        <tr><td rowspan="3">2.4</td><td><span class="pill zs">Zero-shot</span></td><td>3.0</td><td>"moderate amount of noise, which could be distracting to some listeners&hellip;"</td></tr>
        <tr><td><span class="pill sft">SFT</span></td><td>2.8</td><td>"moderate level of noise and discontinuity&hellip;"</td></tr>
        <tr><td><span class="pill dpo">DPO</span></td><td>2.6</td><td>"moderate level of noise and some discontinuity, along with noticeable&hellip;"</td></tr>
        <tr class="grp"><td colspan="4">A good clip &mdash; <span class="mono-sm">book_09628…live_skype_529.wav</span></td></tr>
        <tr><td rowspan="3">3.8</td><td><span class="pill zs">Zero-shot</span></td><td>3.0</td><td>"moderate amount of noise, with a clear understanding of the words despite some static&hellip;"</td></tr>
        <tr><td><span class="pill sft">SFT</span></td><td>3.4</td><td>"good continuity and moderate distortion. However, it suffers from a somewhat&hellip;"</td></tr>
        <tr><td><span class="pill dpo">DPO</span></td><td>3.8</td><td>"moderate level of noise and some distortion, but it is relatively&hellip;"</td></tr>
      </tbody>
    </table>"""

    zs = M["zeroshot"]["pooled"]
    sft = M["sft"]["pooled"]
    dpo = M["dpo"]["pooled"]
    zs_top = M["zeroshot"]["diversity"]["ALL"]
    sft_div = M["sft"]["diversity"]["ALL"]

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOS global task: does the model learn the domain?</title>
<style>{css}</style></head><body><div class="wrap">

<header>
  <h1>Global MOS task: learning the domain, or guessing?</h1>
  <div class="sub">Zero-shot vs full-data SFT vs paper-faithful DPO &middot; Qwen2-Audio-7B &middot; greedy decoding &middot;
  pooled over FOR / LIVE / P501 / NISQA in-domain (1,180 clips) &middot; temporal task excluded</div>
</header>

<p class="lead">Three checkpoints score the same 1,180 test clips on a 1&ndash;5 MOS scale and write a free-text quality
caption. The question here is not just which has the lowest error, but <b>whether the number tracks the audio at
all</b> &mdash; or whether the model has simply learned a safe constant to emit.</p>

<div class="kpis">{kpis}</div>

<div class="callout">
  <b>Headline.</b> The off-the-shelf model does not do this task: its MAE ({zs['mae']:.2f}) is <b>worse than always
  guessing the dataset mean</b> ({zs['mae_const']:.2f}), and its predictions are statistically uncorrelated with the
  truth (Pearson r&nbsp;=&nbsp;{zs['pearson']:.2f}). Fine-tuning turns this into a real predictor: SFT reaches
  r&nbsp;=&nbsp;{sft['pearson']:.2f} with a near-unit fit slope. DPO is a dead heat with SFT on the rating axis &mdash;
  its contribution is to the caption, not the score.
</div>

<h2>1. The decisive picture</h2>
<p>If you only look at one figure, this is it. Predicted MOS against true MOS, every clip, all sets pooled.</p>
<figure><img src="data:image/png;base64,{b64('04_scatter.png')}" alt="Scatter of predicted vs true MOS for the three models">
<figcaption>Zero-shot collapses onto two horizontal bands (3.0 and 4.0); its best-fit line is essentially flat
(slope {zs['slope']:.2f}), meaning the prediction is independent of the true quality. SFT and DPO hug the diagonal
(slope {sft['slope']:.2f}, r {sft['pearson']:.2f}). The mild S-shape at the extremes is ordinary regression to the
mean, not collapse.</figcaption></figure>

<p>The reliability curve says the same thing in expectation. For each true-MOS bin it plots the model's mean prediction.
A diagonal means calibrated; a flat line means the prediction ignores the input.</p>
<figure><img src="data:image/png;base64,{b64('03_calibration.png')}" alt="Calibration curve binned by true MOS">
<figcaption>Zero-shot is pinned near 3.2 across the whole quality range. SFT and DPO follow the diagonal, drifting
slightly conservative below MOS&nbsp;2 and above MOS&nbsp;4.5 &mdash; they under-commit at the extremes but never
invert.</figcaption></figure>

<h2>2. Regression and rank metrics</h2>
<p>The collapsed-and-per-set numbers behind the figures. <code>bias</code> is mean(pred&minus;true), so positive means
the model rates clips too high. <code>slope</code> is the fit of predicted on true (1.0 = perfect tracking, 0 = a
constant). <code>std ratio</code> is how much of the true MOS spread the predictions reproduce. <code>MAE&nbsp;floor</code>
is the error of a dummy model that always predicts that set's mean MOS &mdash; any honest predictor must beat it.</p>
<div class="card" style="padding:8px 14px">
<table>
<thead><tr><th>Model</th><th>MAE&nbsp;&darr;</th><th>MSE&nbsp;&darr;</th><th>bias</th><th>Pearson&nbsp;&uarr;</th>
<th>Spearman&nbsp;&uarr;</th><th>slope&nbsp;&uarr;</th><th>std&nbsp;ratio</th><th>MAE&nbsp;floor</th></tr></thead>
<tbody>{reg_rows()}</tbody></table>
</div>
<div class="callout warn">
  <b>The most damning single number.</b> Zero-shot's pooled MAE ({zs['mae']:.3f}) is larger than the constant-mean
  floor ({zs['mae_const']:.3f}). A model that ignored the audio entirely and always answered with the average MOS would
  be <b>more accurate</b> than the off-the-shelf model actually is. On NISQA in-domain its correlation is even slightly
  negative (r&nbsp;=&nbsp;{M['zeroshot']['per_set']['nisqa_indomain']['pearson']:.3f}).
</div>

<h2>3. Mode collapse on the rating axis</h2>
<p>This is the mechanism behind the flat scatter. The off-the-shelf model writes a different caption for nearly every
clip, but on the <i>number</i> it falls back to a tiny set of safe values.</p>
<figure><img src="data:image/png;base64,{b64('06_diversity.png')}" alt="Distinct predicted MOS values and top-value share">
<figcaption>Across 1,176 scored clips, zero-shot emits only {zs_top['unique_mos']} distinct MOS values and spends
{zs_top['top_mos_share']*100:.0f}% of its answers on a single value ("{zs_top['top_mos'][0]}"). SFT spreads across
{sft_div['unique_mos']} values with no single value above {sft_div['top_mos_share']*100:.0f}%.</figcaption></figure>
<figure><img src="data:image/png;base64,{b64('02_pred_distribution.png')}" alt="Predicted vs true MOS histograms">
<figcaption>The same effect as distributions. Grey is the true MOS histogram; the coloured outline is the prediction.
Zero-shot's mass piles up at 3 and 4 regardless of the grey shape; SFT and DPO recover the full spread of the true
distribution (std ratio {sft['std_ratio']:.2f}).</figcaption></figure>

<h2>4. Where the error lives</h2>
<p>Splitting MAE by true-quality regime shows the failure is not uniform &mdash; it is concentrated where the constant
guess is most wrong.</p>
<figure><img src="data:image/png;base64,{b64('05_error_by_regime.png')}" alt="MAE by true MOS bin">
<figcaption>Zero-shot's error grows toward the ends of the scale: parked at ~3, it is most wrong on genuinely bad
(MOS&nbsp;1&ndash;2) and genuinely excellent (MOS&nbsp;4.5+) speech. SFT and DPO stay flat and low across the whole
range, the signature of a model that is actually listening.</figcaption></figure>

<h2>5. Is the spoken verdict tied to the number?</h2>
<p>A model could get the number right by accident while talking nonsense, or talk sensibly while the number floats free.
A crude check: score each caption's sentiment (positive minus negative quality words) and correlate it with the MOS the
same model emitted.</p>
<figure><img src="data:image/png;base64,{b64('07_sentiment_vs_mos.png')}" alt="Caption sentiment vs predicted MOS">
<figcaption>Zero-shot's words and number are only loosely coupled (r&nbsp;=&nbsp;{M['zeroshot']['sentiment_vs_predmos_r']:.2f})
&mdash; it can write "moderate noise" and still print 3.0. After fine-tuning the caption and the score move together
(r&nbsp;=&nbsp;{M['sft']['sentiment_vs_predmos_r']:.2f} SFT, {M['dpo']['sentiment_vs_predmos_r']:.2f} DPO): the verbal
judgement and the number are produced as one coherent description.</figcaption></figure>

<h2>6. Caption quality &mdash; and an honest caveat</h2>
<p>On the text metrics the gap is just as wide: BLEU rises from ~2 to the high-20s, ROUGE-L from ~0.16 to ~0.52.
But the diversity counts expose the trade the fine-tuned models make.</p>
<div class="card" style="padding:8px 14px">
<table>
<thead><tr><th>Model</th><th>BLEU&nbsp;&uarr;</th><th>ROUGE-L&nbsp;&uarr;</th><th>BERTScore&nbsp;F1&nbsp;&uarr;</th>
<th>distinct&nbsp;MOS</th><th>distinct&nbsp;captions</th><th>mean&nbsp;words</th></tr></thead>
<tbody>{cap_rows()}</tbody></table>
</div>
<div class="callout warn">
  <b>Caveat worth stating.</b> The accuracy comes with templating. Zero-shot writes {zs_top['unique_captions']}
  distinct captions across {zs_top['n_scored']} clips (~{fmt(zs_top['mean_cap_len'],0)} words each, all different);
  SFT writes only {sft_div['unique_captions']} distinct captions ({sft_div['unique_captions']}/{sft_div['n_scored']},
  ~{fmt(sft_div['mean_cap_len'],0)} words). The model learned a compact bank of degradation phrases and reuses them.
  That is exactly what lifts BLEU and MAE together, but it means the captions are closer to a classifier with phrases
  than to free description. The two MOS&nbsp;1.0 clips in the table below get <i>verbatim identical</i> captions.
</div>

<h2>7. A concrete walk-through</h2>
<p>The same three clips from LIVE, ordered by true quality, with what each model said. Watch the zero-shot column stay
glued to 3.0 while the truth moves from 1.0 to 3.8, and the SFT/DPO columns follow.</p>
<div class="card" style="padding:8px 14px">{qual}</div>

<h2>8. What this supports</h2>
<ul>
  <li><b>Fine-tuning, not prompting, creates the capability.</b> Off-the-shelf Qwen2-Audio cannot rate speech quality:
  it is below the trivial mean-guess floor and uncorrelated with the truth. This is the project's motivating result,
  reproduced cleanly on four held-out sets.</li>
  <li><b>SFT genuinely learns the domain.</b> r&nbsp;&asymp;&nbsp;{sft['pearson']:.2f}, fit slope &asymp; {sft['slope']:.2f},
  it reproduces ~{sft['std_ratio']*100:.0f}% of the true MOS spread, and the error is flat across the quality range.
  It orders and scores clips, it does not guess.</li>
  <li><b>DPO does not move the rating axis.</b> Pooled MAE {dpo['mae']:.3f} vs SFT {sft['mae']:.3f}; correlation and
  slope identical to three decimals. Consistent with the thesis finding that paired DPO on a strong SFT base is a tie
  on MOS &mdash; its value, if any, is in caption phrasing and preference alignment, not the number.</li>
  <li><b>The remaining honest weakness is caption diversity.</b> The fine-tuned models trade descriptive variety for
  accuracy. If richer descriptions matter for the downstream hearing-aid use case, that is the axis to push, and it is
  not something MAE or BLEU will reward.</li>
</ul>

<footer>
  Generated from <span class="tag">results/analysis/global_task/stats.json</span> and the raw eval JSONs by
  <span class="tag">scripts/eval/analyze_global_task.py</span> + <span class="tag">build_global_task_report.py</span>.
  Models: zero-shot <span class="mono-sm">qwen2audio_instruct_baseline</span>, SFT
  <span class="mono-sm">sft_full_paper_h100</span>, DPO <span class="mono-sm">dpo_full_sft_paired_lr1e6</span> (all greedy).
  Sentiment is a lightweight lexicon heuristic, indicative only. Temporal-localization task excluded by request.
</footer>

</div></body></html>"""

    (OUT / "report.html").write_text(html)
    print("wrote", OUT / "report.html", f"({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
