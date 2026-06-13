# `results/evaluation/` — eval JSON outputs, grouped by task then method

Two tasks, each grouped by training method. Inside each method folder is one directory per
evaluated checkpoint; inside that, one `test_<split>_results.json` per eval set plus any
decoding-mode metadata.

| Task | Holds | README |
|---|---|---|
| `global/` | global MOS-caption task (MAE/MSE + BLEU) | [`global/README.md`](global/README.md) |
| `temporal/` | time-localization task (temporal IoU) | [`temporal/README.md`](temporal/README.md) |

Each task splits into method subdirs:

```
global/                       temporal/
  alld/      DPO checkpoints     sft/       gc-* SFT checkpoints
  sft/       SFT checkpoints     zeroshot/  zero-shot baseline
  zeroshot/  zero-shot baseline  (alld/)    reserved - none yet
```

See the per-task README for the full experiment-to-result mapping (each dir linked to its
training job, eval job, and DTU JOBID).

## Where the outputs come from

Eval job scripts under `jobs/global/eval/` and `jobs/temporal/eval/` write here. The shared
template `jobs/_lib/templates/evaluate_mos.sh` takes `MODEL_CATEGORY` and routes output into the
matching subdir as `results/evaluation/<task>/<category>/${MODEL_NAME}_eval_${DECODE_MODE}/`.
Older job scripts that don't use the template hardcode the full path including the subdir.

## Tests evaluated on

The default eval bundle (template + most ad-hoc eval scripts) is the four-set NISQA bundle:

- `test_FOR.json`, `test_LIVE.json`, `test_P501.json` — out-of-domain
- `test_nisqa_indomain.json` — in-domain

All four live in `data/processed/eval/`. (Temporal evals use the FOR/LIVE/P501 subset.)

## `_trash/`

Each task has a gitignored `_trash/` holding dirs removed in the 2026-06-13 cleanup
(broken-BLEU re-runs, smoke tests, dupes, de-scoped evals). Recoverable until deleted; see the
per-task README for the specifics.
