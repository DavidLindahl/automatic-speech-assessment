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
| `inference.py` | `load_model(model_id)`, `run_inference(...)`. Used by evaluators and by `scripts/data/generate_dpo_data.py`. Auto-detects TimeAudio checkpoints (via `config.use_abs_time_embedding`) and loads the subclass. |
| `temporal_tokens.py` | Anchor/offset `<aN><fK>` time tokens for temporal localization (TimeAudio mechanism 1). `encode_time`, `decode_all_times`, `all_time_tokens` at 0.1 s resolution. |
| `modeling_timeaudio.py` | `Qwen2AudioTimeForConditionalGeneration` (Qwen2-Audio + optional learnable absolute-time frame embedding, TimeAudio mechanism 2) and `install_time_tokens` (register + numeral-seed the time tokens). Time embedding is zero-init and gated by `config.use_abs_time_embedding` so on/off is a clean ablation. |
| `processed_data.py` | `load_processed_records`, `write_processed_records`, `resolve_audio_path`. |
| `data.py` | Compatibility shim re-exporting from `audio.py`, `prompts.py`, `datasets.py`, `collators.py`. Kept so existing `from asa.data import SFTDataset` callers keep working. Remove after callers migrate. |

## Where the things you'd actually run live

- Training: `scripts/train/supervised-finetune.py`, `scripts/train/dpo-finetune.py`
- Evaluation: `scripts/eval/evaluate.py`, `scripts/eval/evaluate_temporal.py`
- Data generation: `scripts/data/`
- Post-eval analysis: `scripts/analysis/` (`replication/` for reproduction-path aggregators, `thesis_figures/` for one-off figure generators)
