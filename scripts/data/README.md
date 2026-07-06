# `scripts/data/` — Dataset Preparation and Mix Synthesizers

This directory contains utility scripts to preprocess datasets, synthesize degraded audio mixes, and compile SFT/DPO training splits.

## Scripts

- **`generate_nisqa_sim_lowmos_active.py`**: Main synthesizer that overlays reference audio with localized packet loss, clipping, or noise to produce target degraded mixtures.
- **`build_nisqa_temporal_json.py`**: Parses the audio mixtures and formats SFT JSONL training records featuring temporal interval tags.
- **`generate_dpo_data.py`**: Standard utility to pair preferred (chosen) and dispreferred (rejected) responses into DPO-compatible datasets.
- **`prepare_temporal_smoke.py`**: Builds small CPU-friendly test sets used in the local pytest suite.
- **`build_dpo_cycle_splices.py`**: Prepares preference pairs designed for assessing cycles of degradations.
- **`generate_dpo_temporal_factor.py`**: Generates specialized temporal alignment sets.
- **`preprocess_clean_refs.py`**: Normalizes reference text formats across datasets.
- **`join_dis_into_dpo.py`**: Helper to merge discriminator outputs into DPO files.
