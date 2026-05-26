# `src/asa/` — importable library

This package holds only library code. Anything runnable as a CLI or via
`torchrun` lives under `../scripts/`. The boundary is intentional:
`import asa` should never have side effects beyond defining symbols.

## Modules

| Module | Purpose |
|---|---|
| `audio.py` | `load_audio`, `TARGET_SR` (16 kHz). Mono resampling. |
| `prompts.py` | `PROMPT_TEMPLATE` for SFT inputs; `build_expert_prompt_MOS` for the ALLD reference stream. |
| `datasets.py` | `SFTDataset`, `DPODataset`. Read JSONL records, resolve audio paths, return PyTorch dicts. |
| `collators.py` | `Qwen2AudioCollator` (SFT), `ALLDDPOCollator` (DPO dual-stream with label masking). |
| `inference.py` | `load_model(model_id)`, `run_inference(...)`. Used by evaluators and by `scripts/data/generate_dpo_data.py`. |
| `processed_data.py` | `load_processed_records`, `write_processed_records`, `resolve_audio_path`. |
| `generate_temporal_data.py` | `overlay_noise`, `apply_packet_loss`, `apply_clipping`. Library helpers; the runnable mix builder is `scripts/data/generate_nisqa_sim_lowmos_active.py`. |
| `distill_temporal_targets.py` | `generate_targets`. Used by `scripts/data/prepare_temporal_smoke.py`. |
| `sampler.py` | Dataset-sampling utilities for preprocessing. |
| `data.py` | Compatibility shim re-exporting from `audio.py`, `prompts.py`, `datasets.py`, `collators.py`. Kept so existing `from asa.data import SFTDataset` callers keep working. Remove after callers migrate. |

## Where the things you'd actually run live

- Training: `scripts/train/supervised-finetune.py`, `scripts/train/dpo-finetune.py`
- Evaluation: `scripts/eval/evaluate.py`, `scripts/eval/evaluate_temporal.py`
- Data generation: `scripts/data/`
- Debugging / collapse probes: `scripts/diagnostics/`
- Post-eval analysis: `scripts/analysis/`
