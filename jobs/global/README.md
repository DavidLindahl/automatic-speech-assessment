# `jobs/global/` — Global MOS-Captioning Jobs

This directory contains LSF job scripts for the global Mean Opinion Score (MOS) captioning task. The scripts are grouped by pipeline stage.

## Directory Structure

- **[`data/`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/jobs/global/data)**: Prepares training mixes and structures SFT/DPO JSON datasets (e.g., `generate_dpo_paper_half_h100.sh`).
- **[`sft/`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/jobs/global/sft)**: Warmup and full-scale Supervised Fine-Tuning jobs (e.g., `sft_warmup_paper_half_h100.sh`).
- **[`alld/`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/jobs/global/alld)**: Direct Preference Optimization (DPO/ALLD) training jobs (e.g., `dpo_paper_half_h100_lr1e6.sh`).
- **[`eval/`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/jobs/global/eval)**: Baseline and checkpoint evaluation jobs in greedy or sampled decode modes (e.g., `evaluate_dpo_paper_half_h100_greedy.sh`).
