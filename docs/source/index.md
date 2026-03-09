## Documentation

Current DTU HPC flow:

1. Train SFT from `data/processed/train_nisqa_llama_10k.json`.
2. Generate DPO pairs into canonical JSONL at `data/processed/train_dpo_10k.json`.
3. Train ALLD/DPO from the SFT warmup checkpoint in `models/sft_warmup`.
4. Evaluate against the processed test splits, including `test_LIVE.json` on HPC.
