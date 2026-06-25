# Temporal-localization evaluations

Results for the time-localized degradation task (consolidated onto `main`).
Each model reads a synthetic NISQA mix (mostly clean, one degraded window) and predicts
the degraded interval `[start, end]` plus a global MOS-style caption. The headline metric is
mean temporal IoU (`mean_tiou`) between the predicted and construction-time ground-truth
interval, on three NISQA test sets: FOR, LIVE, P501.

## Layout

Grouped by training method (mirrors [`../global/`](../global/README.md)):

```
temporal/
  sft/        global-caption + timestamp SFT models (gc-*)
  zeroshot/   untrained Qwen2-Audio-Instruct baseline
  (alld/)     reserved for a DPO/ALLD temporal model - none trained yet
```

Dir naming is `<family>__<model>__<decode>`:
- **family** = output-format experiment (`gc-plain`, `gc-anchoroffset`, `gc-timelast`, `gc-timelast-softloss`)
- **model** = checkpoint backbone (`timeaudio-h100`, `h100`, `qwen2-instruct`)
- **decode** = `greedy` (temp 0) or `t07` (temp 0.7)

## Experiments

The four `gc-*` runs are an ablation over **how the timestamp is encoded and ordered** in the
target string - same base (Qwen2-Audio-7B), same NISQA mix data, only the output format and time
mechanism change. JOBIDs reference the DTU/LSF runs in the vault `runs/` ledger
(`studies/speech-quality-assesment/runs/`).

### `sft/` — global-caption + timestamp SFT

| Result dir | Output format | Time mechanism | FOR / LIVE / P501 t-IoU | Train / eval JOBID |
|---|---|---|---|---|
| `gc-timelast__timeaudio-h100__greedy` ⭐ | caption then `<\|s\|>…<\|e\|>` **last** | anchor/offset tokens + abs-time embedding | **0.883 / 0.896 / 0.871** | 28633522 / 28645927 |
| `gc-timelast-softloss__h100__greedy` | as above + soft-loss on time tokens | anchor/offset + abs-time, soft-loss | 0.802 / 0.818 / 0.801 | 28644592 / 28647106 |
| `gc-anchoroffset__timeaudio-h100__greedy` | caption with anchor/offset time tokens | anchor/offset tokens + abs-time embedding | 0.145 / 0.083 / 0.100 | 28615749 / 28618994 |
| `gc-plain__h100__greedy` | caption with plain `<\|s\|>…<\|e\|>` | none (plain decimal timestamps) | 0.095 / 0.022 / 0.007 | 28615748 / 28618993 |

Train entrypoints: `gc-plain` via [`jobs/temporal/alld/dpo_temporal_gc_plain.sh`](../../../jobs/temporal/alld/dpo_temporal_gc_plain.sh).
`gc-timelast`, `gc-timeaudio`, and `gc-timelast-softloss` were trained through the temporal
SFT path ([`jobs/temporal/sft/sft_temporal.sh`](../../../jobs/temporal/sft/sft_temporal.sh))
and are identified by JOBID above.

### `zeroshot/` — untrained baseline

| Result dir | Model | Eval JOBID | FOR / LIVE / P501 t-IoU |
|---|---|---|---|
| `zeroshot__qwen2-instruct__t07` | `Qwen/Qwen2-Audio-7B-Instruct`, chatml prompt | 28618191 | 0.185 / 0.201 / 0.111 |

## The headline finding

`gc-timelast` is the breakthrough: **emitting the timestamp last in the caption** (after the
descriptive text, rather than embedded mid-sentence) broke an earlier training collapse and
lifted t-IoU from ~0.1 to ~0.88. The progression makes the ablation clear:

| Format | FOR t-IoU | What changed |
|---|---|---|
| plain `<\|s\|>` timestamps | 0.095 | baseline, no time mechanism - barely localizes |
| anchor/offset time tokens | 0.145 | adds TimeAudio tokens + abs-time embedding - small lift |
| **time-last ordering** | **0.883** | same tokens, but timestamp emitted last - collapse broken |
| time-last + soft-loss | 0.802 | soft-loss on time tokens - slightly below hard-loss |

Zero-shot Qwen2-Instruct (0.185) sits between the two weak SFT formats, confirming the plain and
anchoroffset SFT runs were under-fitting the localization signal, not the timestamp ordering
being the decisive factor.

## `_trash/`

Dirs removed during the 2026-06-13 cleanup: superseded partials (`timeaudio_ckpt400_FOR`),
byte-identical duplicates (`timeaudio_full_FOR`), an empty stub (`sft_temporal`), an
input-only stub (`timeaudio_zero_shot`), smoke tests (`zeroshot_instruct_smoke`, n=10), and
earlier de-scoped ad-hoc evals (`timeaudio_full`, `sft_temporal_test_temp0`,
`sft_temporal_localized_test_temp0`, `sft_temporal_max_mos3*`). Gitignored, recoverable until
deleted. Two were tracked in git; recover with
`git checkout HEAD -- results/evaluation/temporal/<dir>`.
