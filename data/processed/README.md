# `data/processed/` — datasets, grouped by use

Each subdir holds JSON/JSONL files for one stage of the pipeline. Job
scripts reference these paths directly; if you move a file, update the
matching `--json-path` or `--dataset-path` argument in `jobs/`.

| Subdir | Contents |
|---|---|
| `sft/` | SFT training inputs. `train_nisqa_llama_10k.json` is the current default, used by `jobs/global/sft/sft_warmup_paper_half_h100.sh` and the paper-faithful full-SFT job. |
| `dpo/` | DPO chosen/rejected pairs built by `scripts/data/generate_dpo_data.py`. `train_dpo_paper_half_h100_clean.json` is the current default; `train_dpo_10k.json` is the full-10k variant. |
| `eval/` | Held-out test splits: `test_FOR.json`, `test_LIVE.json`, `test_P501.json`, `test_nisqa_indomain.json`. Consumed by everything in `jobs/global/eval/` and `jobs/temporal/eval/`. |
| `temporal/` | Temporal-localization datasets. The **headline gc-timelast model** (t-IoU 0.883) trains on `train_nisqa_temporal_gc_timelast_aug_anchoroffset.json` — caption-first / timestamp-last, anchor/offset time tokens. The earlier `train_nisqa_temporal_global_caption_aug.json` (+ `_anchoroffset`) is the non-timelast augmented variant; `train_nisqa_temporal.json` and the non-aug `_global_caption` pair are upstream build artifacts. Active test sets are `test_{FOR,LIVE,P501}_temporal_global_caption.json` (+ `_anchoroffset`, + `_timelast_anchoroffset` for the format-matched eval). Responses look like `The overall speech is clear, but the quality is interrupted by localized degradation occurring between <\|start\|> and <\|end\|>.` The smoke variant `train_temporal_smoke.jsonl` is gitignored and built on demand. |

Generated mix output dirs (e.g. `nisqa_sim_mix_lowmos_active_3000/`,
`temporal/nisqa_sim_mix_lowmos_active_max_mos3/`) are built by
`scripts/data/generate_nisqa_sim_lowmos_active.py` and mostly gitignored.
