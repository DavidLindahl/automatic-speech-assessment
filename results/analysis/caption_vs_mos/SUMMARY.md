# Caption quality vs MOS accuracy

Per-sample Spearman correlation between caption similarity (to the reference
caption) and absolute MOS error, across the four reported models, each with
greedy and sampled decoding, on FOR / LIVE / P501.

Method (`scripts/analysis/caption_vs_mos.py`):

- Caption similarity = BERTScore-F1 and ROUGE-L, computed per sample. Sentence
  BLEU is excluded (degenerate on n=1).
- The numeric MOS rating is stripped from both reference and prediction before
  scoring, so the correlation is not an artifact of copying the score.
- `mos_error` is the stored per-sample value (|true - predicted MOS|).
- Spearman (rank), per test set; never pooled across sets.
- Direction: better caption -> lower error -> expect negative rho.

Source files are the canonical reported evals pulled from DTU (MAE verified
against `runs/INDEX.md`):

| Model | Eval directory |
|---|---|
| Warmup SFT | `sft/sft_warmup_paper_half_h100_eval_{greedy,sampled}` |
| Full SFT | `sft/sft_full_paper_h100_eval_{greedy,sampled}` |
| Warmup-DPO | `dpo/dpo_paper_half_h100_lr1e6_delimiterfix_eval_{greedy,sampled}` |
| Full-DPO | `dpo/dpo_full_sft_paired_lr1e6_eval_{greedy,sampled}` |

## Headline (mean over the 3 test sets)

| Model | mean MOS err (greedy) | mean \|rho_BERT\| (greedy) | mean MOS err (sampled) | mean \|rho_BERT\| (sampled) |
|---|---|---|---|---|
| Warmup SFT | 1.47 | 0.572 | 1.58 | 0.507 |
| Full SFT | 0.33 | 0.157 | 0.39 | 0.213 |
| Warmup-DPO | 0.79 | 0.512 | 0.79 | 0.328 |
| Full-DPO | 0.31 | 0.184 | 0.38 | 0.167 |

The correlation magnitude tracks how weak the model is: the more MOS error a
model has, the more strongly its caption quality predicts that error. Strong
models (Full SFT, Full-DPO) have small errors AND a weak caption-MOS coupling;
weak models (Warmup SFT, Warmup-DPO) have large errors AND a strong coupling.

Sign is negative and consistent across all 3 test sets for every model/decoder
(24/24). Within a model, the coupling is strongest on the hardest set (P501).

ROUGE-L vs BERTScore per-sample redundancy: rho 0.71-0.91 across the board.
The two metrics are correlated but not interchangeable; the gap supports
reporting both a lexical and a semantic metric (appendix).
