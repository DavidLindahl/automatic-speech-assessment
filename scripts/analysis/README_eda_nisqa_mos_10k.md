# EDA: global MOS SFT training set (`train_nisqa_llama_10k`)

Descriptive analysis of the dataset every downstream model is built from: the
Qwen2-Audio MOS captioning SFT set. Runs locally (no HPC, stdlib + matplotlib only).

## What it does

`eda_nisqa_mos_10k.py` loads `data/processed/sft/train_nisqa_llama_10k.json`
(10,000 JSONL records: `audios`, `query`, `response`, `mos`, `noi`, `col`, `loud`,
`clean_path`), joins each record 1:1 to `NISQA_TRAIN_SIM_file.csv` by degraded
filename to recover the degradation tags and source corpus, reads clip durations
directly from the WAV RIFF headers (the clips are PCM WAVs, but the reader needs no
`soundfile`/`scipy`), and writes five publication-styled figures plus a stats JSON.

## Figures (written as PDF + PNG)

| File | What it shows | Decision it supports |
|---|---|---|
| `eda_mos_distribution` | MOS distribution, full 1.0-5.0 range | The SFT set is not MOS-filtered; the model sees the whole quality scale |
| `eda_degradation_taxonomy` | 13 raw NISQA tags vs 6-category collapse | Honesty of the 13->6 collapse; the tail and residual imbalance caveat |
| `eda_degradations_per_clip` | Simultaneous tags per clip (0-9) | 80.4% of clips are multi-degradation; captions summarise a mixture |
| `eda_clip_duration` | Clip-length histogram (seconds) | Absolute time scale of one training example |
| `eda_source_composition` | Clips per source corpus + MOS spread per corpus | Corpus composition; corpus identity only weakly tracks quality |

`eda_nisqa_mos_10k_stats.json` holds every number behind the figures.

## Run

```bash
# from the repo root; defaults write into the thesis figures dir
python scripts/analysis/eda_nisqa_mos_10k.py \
  --figures-dir ../_papers/asa-thesis/figures
```

Useful flags: `--duration-sample N` (cap clips probed for duration; header reads
only, so all 10k is fast and the default), `--sft-jsonl`, `--nisqa-csv`,
`--data-root`, `--figures-dir`.

## Key numbers (current data)

- MOS: n=10,000, range 1.0-5.0, mean 2.98, median 3.0, std 1.10; 51.7% <= 3.0,
  48.3% > 3.0.
- Raw tags: codec1=7,841, plcMode1=7,841, bgn=3,077, filter=1,133, codec2=876,
  plcMode2=876, arb_filter=810, clipping=761, wbgn=428, timeclipping=155, p50mnru=125,
  codec3=101, plcMode3=101.
- 6 categories: codec artifacts 8,818, packet-loss concealment 8,818, background noise
  3,630, band-limiting filter 1,943, clipping distortion 761, time clipping 155.
- Per clip: 5.1% zero-tag, 14.5% single-tag, 80.4% multi-tag, mean 2.41 tags/clip
  (max 9).
- Duration: range 4.5-12.0s, median 9.0s, 5-95% range 6.0-12.0s (quantised to whole
  seconds).
- Source: AusTalk 4,500 (mean MOS 2.77), DNS 4,000 (3.17), UKIRE 1,000 (3.14),
  TSP 500 (3.07).

## Notes

- The figures are consumed by `chapters/04_method.tex` in the thesis repo
  (`CarlSvejstrup/bachelor-thesis`), in the new "The MOS supervision set" subsection.
- This set is distinct from the temporal-mix dataset. The MOS<=3.0 filter and the
  active-window / interval analyses belong to that separate pipeline
  (`scripts/data/generate_nisqa_sim_lowmos_active.py`), not to this set.
