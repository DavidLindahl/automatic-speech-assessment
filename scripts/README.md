# `scripts/` — runnable utilities

Library code lives in `../src/asa/`. Everything you can run, including
LSF entrypoints, lives here.

| Subdir | Purpose | Called by |
|---|---|---|
| `train/` | SFT and DPO trainers + the multi-step submission shell. | `jobs/*/sft/`, `jobs/*/alld/` |
| `eval/` | `evaluate.py` (MOS-style eval) and `evaluate_temporal.py` (temporal IoU + interval metrics). | `jobs/*/eval/` |
| `data/` | Dataset builders and smoke-set prep. `generate_nisqa_sim_lowmos_active.py` synthesises REF/DEG mixes, `build_nisqa_temporal_json.py` produces the SFT JSONL, `generate_dpo_data.py` builds chosen/rejected pairs, `prepare_temporal_smoke.py` builds the small smoke set. | `jobs/*/data/`, ad-hoc |
| `diagnostics/` | Probes for when DPO collapses: empty-output diagnosis and pre/post sanity checks. Some (`sanity_check_dpo.py`, `dpo_sanity_check.py`) are wired into live jobs and kept; one-off ones are deleted when their bug closes. | ad-hoc + a few eval jobs |
| `analysis/` | Thesis figures and post-eval aggregators: `eval_pred_vs_true.py` (zero-shot vs SFT vs DPO calibration), `caption_vs_mos.py` (caption-quality vs MOS-error coupling), `extract_datasize_sweep.py` + `plot_datasize_sweep.py` (the data-size figures), `probe_temporal_frames.py` (frozen-feature linear probe), `eda_eval_sets.py` / `eda_nisqa_mos_10k.py` (dataset EDA), `audit_response_diversity.py`. | ad-hoc |

## When to add a new file

- Hot-path library function imported by many places → `src/asa/`.
- A new training paradigm or eval CLI → `scripts/train/` or `scripts/eval/`. Add a matching `jobs/` shell wrapper.
- A new data builder → `scripts/data/`.
- A one-shot probe for a current bug → `scripts/diagnostics/`. Don't worry about over-polishing; these scripts get deleted when the bug closes (or they get kept around if they're broadly useful, like the dpo-empty-output prober).
- An aggregator that consumes eval outputs → `scripts/analysis/`.
