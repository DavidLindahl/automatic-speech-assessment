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
uv run python notebooks/build_temporal_inspector_site.py \
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

## Build Temporal Training JSONL

Reuse existing NISQA captions and inject temporal targets without new Gemini calls:

```bash
uv run python src/asa/build_nisqa_temporal_json.py \
  --manifest-path data/processed/nisqa_sim_mix_lowmos_active_3000/manifest.csv \
  --caption-jsonl data/processed/train_nisqa_llama_10k.json \
  --mixes-dir data/processed/nisqa_sim_mix_lowmos_active_3000 \
  --output-jsonl data/processed/train_nisqa_temporal_mix_3000.json
```

The output keeps familiar keys like `audios`, `response`, `query`, and `mos`, and adds temporal fields from the
manifest (`mix_deg_segments`, `source_degradation_types`, `mix_filename`). Each response is timestamp-supervised in
this form:

`... interrupted by <degradation phrase> occurring between <|start|> and <|end|>.`

When fine-tuning, pass `--use-query-prompt` so the model sees the timestamp-localization instruction in each record's
`query` field.

## Evaluate Temporal Localization

Run temporal inference and compute localization metrics (t-IoU, hit rates, start/end timestamp errors):

```bash
uv run python src/asa/evaluate_temporal.py \
  --model-path models/sft_temporal_max_mos3 \
  --dataset-path data/processed/train_nisqa_temporal_mix_max_mos3.json \
  --data-root data \
  --output-dir results/evaluation/sft_temporal_max_mos3 \
  --batch-size 4 \
  --greedy \
  --use-query-prompt
```

For a quick smoke check before a full run:

```bash
uv run python src/asa/evaluate_temporal.py \
  --model-path models/sft_temporal_max_mos3 \
  --dataset-path data/processed/train_nisqa_temporal_mix_max_mos3.json \
  --data-root data \
  --max-samples 64 \
  --output-dir results/evaluation/sft_temporal_max_mos3_smoke
```
