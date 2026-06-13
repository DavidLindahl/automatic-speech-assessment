# `scripts/` — runnable utilities

Library code lives in `../src/asa/`. Everything you can run, including
LSF entrypoints, lives here.

| Subdir | Purpose | Called by |
|---|---|---|
| `train/` | SFT and DPO trainers + the multi-step submission shell. | `jobs/*/sft/`, `jobs/*/alld/` |
| `eval/` | `evaluate.py` (MOS-style eval) and `evaluate_temporal.py` (temporal IoU + interval metrics). | `jobs/*/eval/` |
| `data/` | Dataset builders and smoke-set prep. `generate_nisqa_sim_lowmos_active.py` synthesises REF/DEG mixes, `build_nisqa_temporal_json.py` produces the SFT JSONL, `generate_dpo_data.py` builds chosen/rejected pairs, `prepare_temporal_smoke.py` builds the small smoke set. | `jobs/*/data/`, ad-hoc |
| `diagnostics/` | One-shot probes for when DPO collapses again: empty-output diagnosis and pre/post sanity checks. | ad-hoc + a few eval jobs |
| `analysis/` | Thesis figures and post-eval aggregators: zero-shot pred-vs-true, caption-vs-MOS coupling, eval-set EDA, response-diversity audit. | ad-hoc |

## When to add a new file

- Hot-path library function imported by many places → `src/asa/`.
- A new training paradigm or eval CLI → `scripts/train/` or `scripts/eval/`. Add a matching `jobs/` shell wrapper.
- A new data builder → `scripts/data/`.
- A one-shot probe for a current bug → `scripts/diagnostics/`. Don't worry about over-polishing; these scripts get deleted when the bug closes (or they get kept around if they're broadly useful, like the dpo-empty-output prober).
- An aggregator that consumes eval outputs → `scripts/analysis/`.
