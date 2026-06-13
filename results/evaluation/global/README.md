# Global MOS-caption evaluations

Results for the global descriptive-quality task: the model reads a clip and emits a single
MOS-style caption (a sentence describing overall quality, ending in a numeric MOS). Metrics are
MOS `mae` / `mse` against the reference score and caption `bleu` (0-100 scale), on four sets:
FOR, LIVE, P501, and `nisqa_indomain`.

## Layout

Grouped by training method (mirrored by [`../temporal/`](../temporal/README.md)):

```
global/
  alld/       DPO / ALLD-aligned models   (dpo_*)
  sft/        supervised fine-tuned        (sft_*)
  zeroshot/   untrained Qwen2-Audio-Instruct baseline
```

Dir naming is `<model>_eval_<decode>`: `greedy` = temperature 0, `sampled`/`_sampled` = temperature > 0.
Greedy is the canonical decode for thesis numbers.

## Experiments

Each row links the result dir to the **training** job that produced the checkpoint and the
**eval** job that produced the numbers. JOBIDs reference the DTU/LSF runs logged in the vault
`runs/` ledger (`studies/speech-quality-assesment/runs/`). FOR-set greedy MAE/BLEU shown as the
at-a-glance headline; full numbers are in each dir's `test_<set>_results.json`.

### `alld/` — DPO / ALLD-aligned

| Result dir | Train job | Eval job | FOR mae / bleu | Train / eval JOBID |
|---|---|---|---|---|
| `dpo_full_sft_paired_lr1e6_eval_greedy` ⭐ | [`dpo_full_sft_paired_lr1e6.sh`](../../../jobs/global/alld/dpo_full_sft_paired_lr1e6.sh) | [`evaluate_dpo_full_sft_paired_lr1e6_greedy.sh`](../../../jobs/global/eval/evaluate_dpo_full_sft_paired_lr1e6_greedy.sh) | **0.263 / 27.8** | 28557823 / 28563601 |
| `dpo_full_sft_paired_lr1e6_eval_sampled` | (same) | [`..._sampled.sh`](../../../jobs/global/eval/evaluate_dpo_full_sft_paired_lr1e6_sampled.sh) | 0.322 / 25.7 | 28557823 / 28563602 |
| `dpo_cycle2_eval_greedy` | [`dpo_cycle2.sh`](../../../jobs/global/alld/dpo_cycle2.sh) | [`evaluate_dpo_cycle2_greedy.sh`](../../../jobs/global/eval/evaluate_dpo_cycle2_greedy.sh) | 0.290 / 26.3 | 28596884 / 28598682 |
| `dpo_cycle2_eval_sampled` | (same) | [`..._sampled.sh`](../../../jobs/global/eval/evaluate_dpo_cycle2_sampled.sh) | 0.339 / 24.5 | 28596884 / 28598683 |
| `dpo_nonorm_eval_greedy` | [`dpo_full_sft_paired_nonorm.sh`](../../../jobs/global/alld/dpo_full_sft_paired_nonorm.sh) | [`evaluate_dpo_nonorm_greedy.sh`](../../../jobs/global/eval/evaluate_dpo_nonorm_greedy.sh) | 0.400 / 23.0 | 28599881 / 28602353 |
| `dpo_nonorm_eval_sampled` | (same) | [`..._sampled.sh`](../../../jobs/global/eval/evaluate_dpo_nonorm_sampled.sh) | 0.397 / 22.6 | 28599881 / 28602354 |
| `dpo_full_sft_lr1e6_eval_greedy` | [`dpo_full_sft_lr1e6.sh`](../../../jobs/global/alld/dpo_full_sft_lr1e6.sh) | [`evaluate_dpo_full_sft_lr1e6_greedy.sh`](../../../jobs/global/eval/evaluate_dpo_full_sft_lr1e6_greedy.sh) | 0.453 / 23.9 | 28517231 / 28554408 |
| `dpo_full_sft_lr1e6_eval_sampled` | (same) | [`..._sampled.sh`](../../../jobs/global/eval/evaluate_dpo_full_sft_lr1e6_sampled.sh) | 0.463 / 23.2 | 28517231 / 28554409 |
| `dpo_paper_half_h100_lr1e6_eval_greedy` | [`dpo_paper_half_h100_lr1e6.sh`](../../../jobs/global/alld/dpo_paper_half_h100_lr1e6.sh) | [`evaluate_dpo_lr1e6_delimiterfix.sh`](../../../jobs/global/eval/evaluate_dpo_lr1e6_delimiterfix.sh) | 0.760 / 21.2 | weak - early half-data run |
| `dpo_paper_half_h100_lr1e6_sampled` | (same) | [`evaluate_dpo_lr1e6_sampled.sh`](../../../jobs/global/eval/evaluate_dpo_lr1e6_sampled.sh) | 0.823 / 20.2 | weak |

### `sft/` — supervised fine-tuned

| Result dir | Train job | Eval job | FOR mae / bleu | Train / eval JOBID |
|---|---|---|---|---|
| `sft_full_paper_h100_eval_greedy` ⭐ | [`sft_full_paper_h100.sh`](../../../jobs/global/sft/sft_full_paper_h100.sh) | [`evaluate_sft_full_paper_h100_greedy.sh`](../../../jobs/global/eval/evaluate_sft_full_paper_h100_greedy.sh) | **0.273 / 26.4** | 28504316 / 28515705 |
| `sft_full_paper_h100_eval_sampled` | (same) | [`..._sampled.sh`](../../../jobs/global/eval/evaluate_sft_full_paper_h100_sampled.sh) | 0.348 / 23.1 | 28504316 / 28515706 |
| `sft_clean9500_eval_greedy` | [`sft_full_clean9500_h100.sh`](../../../jobs/global/sft/sft_full_clean9500_h100.sh) | [`evaluate_sft_clean9500_greedy.sh`](../../../jobs/global/eval/evaluate_sft_clean9500_greedy.sh) | 0.309 / 26.2 | 28599857 / 28602351 |
| `sft_clean9500_eval_sampled` | (same) | [`..._sampled.sh`](../../../jobs/global/eval/evaluate_sft_clean9500_sampled.sh) | 0.367 / 23.7 | 28599857 / 28602352 |
| `sft_warmup_paper_half_h100_eval_greedy` | [`sft_warmup_paper_half_h100.sh`](../../../jobs/global/sft/sft_warmup_paper_half_h100.sh) | [`evaluate_sft_warmup_paper_half_h100_greedy.sh`](../../../jobs/global/eval/evaluate_sft_warmup_paper_half_h100_greedy.sh) | 1.382 / 12.1 | weak - warmup-only, half data |
| `sft_warmup_paper_half_h100_eval_sampled` | (same) | [`evaluate_sft_warmup_paper_half_h100.sh`](../../../jobs/global/eval/evaluate_sft_warmup_paper_half_h100.sh) | 1.629 / 11.5 | weak |

### `zeroshot/` — untrained baseline

| Result dir | Model | Eval JOBID | FOR mae / bleu |
|---|---|---|---|
| `qwen2audio_instruct_baseline` | `Qwen/Qwen2-Audio-7B-Instruct` (no fine-tuning) | 28611505 | 1.030 / 2.2 |

## The thesis comparison

The three canonical rows consumed by [`scripts/eval/analyze_global_task.py`](../../../scripts/eval/analyze_global_task.py)
and [`scripts/eval/build_global_task_report.py`](../../../scripts/eval/build_global_task_report.py)
(greedy decode):

| Stage | Dir | FOR mae | FOR bleu |
|---|---|---|---|
| Zero-shot | `zeroshot/qwen2audio_instruct_baseline` | 1.030 | 2.2 |
| + SFT | `sft/sft_full_paper_h100_eval_greedy` | 0.273 | 26.4 |
| + DPO (paired) | `alld/dpo_full_sft_paired_lr1e6_eval_greedy` | **0.263** | **27.8** |

The story: zero-shot Qwen2-Audio barely captions quality (BLEU ~2, MAE ~1.0). SFT on the
NISQA-LLaMA captions collapses MAE to ~0.27 and lifts BLEU to ~26. DPO with paired preferences
edges SFT further (MAE 0.263, BLEU 27.8). The unpaired/no-norm/half-data DPO variants are
ablations that under-perform the paired run.

## `_trash/`

Five broken-BLEU eval dirs removed 2026-06-13 (`sft_full_eval`, `sft_warmup_eval_a40`,
`sft_warmup_plus1epoch_eval_a40`, `sft_warmup_plus2epoch_l40s_eval_a40`,
`sft_warmup_plus3epoch_..._eval_h100`). They computed BLEU on a 0-1 scale (a bug); the
`_v2`/`paper_h100` re-runs fixed it to 0-100. MAE in the trashed dirs is still valid. Gitignored,
recoverable until deleted.
