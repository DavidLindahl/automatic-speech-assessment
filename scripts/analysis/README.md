# `scripts/analysis/` — Post-Evaluation Analysis

This directory contains analysis scripts for generating thesis figures and running replication probes.

## Subdirectories

- **[`replication/`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/scripts/analysis/replication)**: Scripts on the documented reproduction path:
  - `extract_datasize_sweep.py` & `plot_datasize_sweep.py`: Aggregates metrics and plots curves/bars for data-size sweep experiments.
  - `probe_temporal_frames.py`: Performs a frozen-feature linear probe on temporal frames.
- **[`thesis_figures/`](file:///Users/davidlindahl/Documents/DTU/Bachelor/automatic-speech-assessment/scripts/analysis/thesis_figures)**: Scripts for generating specific one-off figures that write directly to the thesis `figures/` directory:
  - `eval_pred_vs_true_calibrated.py`: Generates the calibration scatter plot comparing predicted vs. ground-truth values.
