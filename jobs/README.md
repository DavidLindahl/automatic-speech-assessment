# `jobs/` — LSF Job Submission Scripts

This directory contains the Load Sharing Facility (LSF) job scripts used to run data generation, training, and evaluation pipelines on the DTU HPC cluster.

## Directory Structure

- **[`_lib/`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/jobs/_lib)**: Shared LSF infrastructure, including the environment configuration preamble (`preamble.sh`), memory budget linter (`lint-budget.sh`), and reusable job templates. See [`jobs/_lib/README.md`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/jobs/_lib/README.md) for details.
- **[`global/`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/jobs/global)**: Job scripts for the global MOS-captioning task.
- **[`temporal/`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/jobs/temporal)**: Job scripts for the temporal localization task.

## Task Directory Subdivisions

Both `global/` and `temporal/` task directories are subdivided by their roles:
- `data/`: Dataset preprocessing and mix generation jobs.
- `sft/`: Supervised Fine-Tuning (SFT) training jobs.
- `alld/`: Aligning LLMs with Direct Preference Optimization (DPO/ALLD) training jobs.
- `eval/`: Model inference and evaluation scripts.
