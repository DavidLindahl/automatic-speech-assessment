# `data/processed/` — datasets, grouped by use

Each subdir holds JSON/JSONL files for one stage of the pipeline. Job
scripts reference these paths directly; if you move a file, update the
matching `--json-path` or `--dataset-path` argument in `jobs/`.

| Subdir | Contents |
|---|---|
| `sft/` | SFT training inputs. `train_nisqa_llama_10k.json` is the current default, used by `jobs/global/sft/sft_warmup_paper_half_h100.sh` and the paper-faithful full-SFT job. |
| `dpo/` | DPO chosen/rejected pairs built by `scripts/data/generate_dpo_data.py`. `train_dpo_paper_half_h100_clean.json` is the current default. |
| `eval/` | Held-out test splits: `test_FOR.json`, `test_LIVE.json`, `test_P501.json`, `test_nisqa_indomain.json`. Plus the legacy `mos_predictions.json` output. Consumed by everything in `jobs/global/eval/` and `jobs/temporal/eval/`. |
| `temporal/` | Current temporal-localization datasets. `train_nisqa_temporal.json` is the base SFT input; the global-caption variants (`train_nisqa_temporal_global_caption*.json`, plain + anchoroffset + aug) feed the gc-* runs. Matching test sets are `test_FOR_temporal.json`, `test_LIVE_temporal.json`, `test_P501_temporal.json` (plus their `_global_caption[_anchoroffset]` variants). Responses use compact metadata labels like `The overall speech is clear, but the quality is interrupted by localized degradation occurring between <|start|> and <|end|>.` The smoke variant `train_temporal_smoke.jsonl` is gitignored and built on demand. |
| `intermediate/` | Build artifacts. `mos_dataset.json` is the pre-caption MOS dataset. Not a direct training input. (The pre-2026-04-13 AB-direction files were removed in the 2026-06-13 cleanup.) |

Generated mix output dirs (e.g. `nisqa_sim_mix_lowmos_active_3000/`,
`temporal/nisqa_sim_mix_lowmos_active_max_mos3/`) are built by
`scripts/data/generate_nisqa_sim_lowmos_active.py` and mostly gitignored.
