# `results/evaluation/` — eval JSON outputs, grouped by model type

Each subdir holds one folder per evaluated model checkpoint. Inside each
checkpoint folder, one `test_<split>_results.json` per dataset under
`data/processed/eval/`, plus any decoding-mode metadata file.

| Subdir | Holds |
|---|---|
| `dpo/` | DPO checkpoints (paper-half, lr1e6, delimiterfix, step200, plus2epoch). |
| `sft/` | SFT checkpoints (warmup, warmup-paper-half, full, plus1/2/3epoch variants). |
| `temporal/` | SFT checkpoints trained on NISQA-SIM temporal mixes. Current scope. |

## Where the outputs come from

Eval job scripts under `jobs/evaluate/` write here. The shared template
`jobs/_lib/templates/evaluate_mos.sh` requires `MODEL_CATEGORY` (one of
`dpo`, `sft`, `temporal`) and routes the output into the matching subdir
as `results/evaluation/$MODEL_CATEGORY/${MODEL_NAME}_eval_${DECODE_MODE}/`.

Older job scripts that don't use the template hardcode the full path
including the category subdir.

## Tests evaluated on

The default eval bundle (set by the template and most ad-hoc eval
scripts) is the four-test-set NISQA bundle:

- `test_FOR.json`, `test_LIVE.json`, `test_P501.json` — out-of-domain
- `test_nisqa_indomain.json` — in-domain

All four live in `data/processed/eval/`.
