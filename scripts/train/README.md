# `scripts/train/` — Model Training Scripts

This directory contains the main entrypoint scripts for fine-tuning and aligning the audio language models.

## Scripts

- **`supervised-finetune.py`**: Handles Supervised Fine-Tuning (SFT) of Qwen2-Audio. Features configurable learning rates, DeepSpeed integration, label masking, and custom temporal loss weights (including Gaussian-smoothed anchor/offset targets).
- **`dpo-finetune.py`**: Performs Direct Preference Optimization (DPO) aligning model output text with target preferences (ALLD preference alignment).
