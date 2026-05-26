# `scripts/_legacy/` — archival code

This directory holds code that's no longer on the active path but is preserved
for archival reproducibility. **Nothing here is imported by `src/asa/`.**

## What's here

### `caption_generator.py`

Generates descriptive MOS / A/B preference captions for NISQA samples using
the Google Gemini API. Output feeds an older training flow that ingested
captions instead of synthesizing supervision from NISQA-SIM mixes.

Requires `GEMINI_API_KEY` in the environment.

### `legacy_data_cli.py`

Two typer commands extracted from the original `src/asa/data.py`:

- `download` — pulls NISQA from Google Cloud Storage (uses `google.cloud.storage`).
- `generate-captions` — drives `asa.sampler` + `caption_generator` to produce
  `train_nisqa_llama_10k.json` (and the now-defunct A/B JSON).

Run via:

```sh
python scripts/_legacy/legacy_data_cli.py download
python scripts/_legacy/legacy_data_cli.py generate-captions data/raw data/processed
```

## Why it's archived, not deleted

The 2026-04-13 pivot moved supervision to synthetic NISQA-SIM mixes with
construction-time ground truth (see `src/asa/generate_nisqa_sim_lowmos_active.py`
and `src/asa/build_nisqa_temporal_json.py`). Caption-generation is no longer
how training data is produced. The code stays here so a past run can be
reproduced if needed.
