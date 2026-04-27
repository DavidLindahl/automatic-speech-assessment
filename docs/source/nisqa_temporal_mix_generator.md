# NISQA Temporal Mix Generator

This page documents the script that generates low-MOS NISQA-SIM temporal mixes with one active-region degradation
segment per clip.

## Script

`src/asa/generate_nisqa_sim_lowmos_active.py`

## What It Produces

- Mixed `.wav` files in the selected output directory.
- A `manifest.csv` with:
  - `mix_deg_segments`
  - `mix_timeline`
  - `switch_points`
  - `segment_active_fraction`
  - source metadata columns (`filename_ref`, `filename_deg`, `mos`, etc.)

## Basic Usage

Generate the full dataset (3000 files):

```bash
uv run python src/asa/generate_nisqa_sim_lowmos_active.py \
  --total-mix-files 3000 \
  --output-dir data/processed/nisqa_sim_mix_lowmos_active_3000 \
  --overwrite
```

## Common Options

- `--mos-max-threshold`: Maximum MOS for source selection. Default is `3.0`.
- `--require-active-degradation-types / --no-require-active-degradation-types`: Filter source rows by active tags.
- `--output-active-fraction-min`: Minimum active-speech fraction required in selected segment.
- `--max-row-attempts`: Placement retries per source row.
- `--allow-source-reuse / --no-allow-source-reuse`: Allow source-row reuse across passes.
- `--max-source-passes`: Number of shuffled passes over eligible rows.
- `--seed`: Random seed for reproducibility.

See all options:

```bash
uv run python src/asa/generate_nisqa_sim_lowmos_active.py --help
```

## Inspector Website

Build a static HTML inspector for browsing waveform overlays and audio playback:

```bash
uv run python src/asa/build_temporal_inspector_site.py \
  --manifest-path data/processed/nisqa_sim_mix_lowmos_active_3000/manifest.csv
```

This generates:

- `data/processed/nisqa_sim_mix_lowmos_active_3000/inspector/index.html`
- `data/processed/nisqa_sim_mix_lowmos_active_3000/inspector/records.json`

Serve the site locally:

```bash
cd data/processed/nisqa_sim_mix_lowmos_active_3000/inspector
uv run python -m http.server 8000
```
