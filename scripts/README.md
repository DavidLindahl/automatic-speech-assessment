# `scripts/` — runnable utilities

Library code lives in `../src/asa/`. Everything you can run, including
LSF entrypoints, lives here.

| Subdir | Purpose | Called by |
|---|---|---|
| `train/` | SFT and DPO trainers + the multi-step submission shell. | `jobs/*/sft/`, `jobs/*/alld/` |
| `eval/` | `evaluate.py` (MOS-style eval) and `evaluate_temporal.py` (temporal IoU + interval metrics). | `jobs/*/eval/` |
| `data/` | Dataset builders and smoke-set prep. `generate_nisqa_sim_lowmos_active.py` synthesises REF/DEG mixes, `build_nisqa_temporal_json.py` produces the SFT JSONL, `generate_dpo_data.py` builds chosen/rejected pairs, `prepare_temporal_smoke.py` builds the small smoke set. | `jobs/*/data/`, ad-hoc |
| `analysis/replication/` | Aggregators in the documented reproduction path: `extract_datasize_sweep.py` + `plot_datasize_sweep.py` (data-size figures), `probe_temporal_frames.py` (frozen-feature linear probe, also exercised by `tests/`). | `jobs/temporal/eval/frame_probe_temporal.sh`, README repro steps |
| `analysis/thesis_figures/` | One-off figure generators that write straight into the thesis `figures/` dir: `eval_pred_vs_true_calibrated.py` (the cited calibration scatter). Not on any job path. | ad-hoc |

## When to add a new file

- Hot-path library function imported by many places → `src/asa/`.
- A new training paradigm or eval CLI → `scripts/train/` or `scripts/eval/`. Add a matching `jobs/` shell wrapper.
- A new data builder → `scripts/data/`.
- An aggregator on the reproduction path (regenerates a cited number or figure from committed eval output) → `scripts/analysis/replication/`.
- A one-off generator for a specific thesis figure → `scripts/analysis/thesis_figures/`.
