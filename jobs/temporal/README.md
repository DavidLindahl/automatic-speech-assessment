# `jobs/temporal/` — Temporal Localization Jobs

This directory contains LSF job scripts for the temporal localization task (TimeAudio). The scripts are grouped by pipeline stage.

## Directory Structure

- **[`data/`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/jobs/temporal/data)**: Prepares temporal dataset splits, annotations, and low-MOS active simulation mixes.
- **[`sft/`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/jobs/temporal/sft)**: Supervised Fine-Tuning jobs for training temporal localization models (e.g., `sft_gc_timelast_softloss_h100.sh`).
- **[`alld/`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/jobs/temporal/alld)**: DPO alignment jobs optimized for localization (e.g., `dpo_temporal_armA_full.sh`).
- **[`eval/`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/jobs/temporal/eval)**: Evaluation jobs using temporal metrics (t-IoU, BLEU) and frame-level probing jobs.
