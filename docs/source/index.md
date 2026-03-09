## Documentation

Current DTU HPC flow:

1. Run `uv run python src/asa/preflight.py check --mode pipeline` before submitting jobs.
2. Train SFT from `data/processed/train_nisqa_llama_10k.json`.
3. Generate DPO pairs into canonical JSONL at `data/processed/train_dpo_10k.json`.
4. Train ALLD/DPO from the SFT warmup checkpoint in `models/sft_warmup`.
5. Evaluate against the processed test splits, including `test_LIVE.json` on HPC.
