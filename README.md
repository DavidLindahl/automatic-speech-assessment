# Automatic Speech Quality Assessment

Bachelor project on **audio LLMs for descriptive speech quality assessment with temporal localization**. We fine-tune **Qwen2-Audio** with the **ALLD** alignment method on synthetic NISQA-SIM mixes, producing a model that outputs a base MOS-style caption plus time-localized degradation annotations of the form `(start, end, degradation_category)`.

Authors: Carl Schmidt-Svejstrup, David Lindahl. DTU, 2026.

## Active entrypoints

```
src/asa/
  audio.py                       # 16 kHz mono WAV loader, audio constants
  prompts.py                     # PROMPT_TEMPLATE + MOS expert-prompt builder
  datasets.py                    # SFTDataset, DPODataset (PyTorch Datasets)
  collators.py                   # Qwen2AudioCollator, ALLDDPOCollator
  data.py                        # compatibility shim re-exporting the above
  supervised-finetune.py         # SFT entrypoint (Qwen2-Audio backbone)
  dpo-finetune.py                # ALLD-DPO entrypoint with custom trainer
  evaluate.py                    # MOS-style eval CLI
  evaluate_temporal.py           # temporal-localization eval CLI
  inference.py                   # public load_model() + run_inference() API
  generate_dpo_data.py           # DPO-pair generator (calls SFT model)
  generate_nisqa_sim_lowmos_active.py  # NISQA-SIM REF/DEG mix synthesis
  build_nisqa_temporal_json.py   # SFT JSONL builder with temporal targets
  generate_temporal_data.py      # noise overlay / packet loss / clipping
  processed_data.py              # dataset I/O + audio path resolution
  sampler.py
```

`src/asa/data.py` is a 47-line shim re-exporting from the focused modules
(`audio.py`, `prompts.py`, `datasets.py`, `collators.py`). Existing
`from asa.data import SFTDataset` imports keep working unchanged.

## Running on the DTU HPC

LSF job scripts live under `jobs/`. Sourceable infrastructure under
`jobs/_lib/`:

- `jobs/_lib/preamble.sh` — strict mode, CUDA module, venv activation,
  HF cache off `/work3`, HF auth preflight, startup banner. Source from
  each job *after* the `#BSUB` header.
- `jobs/_lib/lint-budget.sh` — enforces the LSF `rusage[mem]`-per-core
  rule programmatically. Run before submission to catch over-budget
  scripts that would otherwise trigger the angry-DTU-IT failure mode.
- `jobs/_lib/templates/evaluate_mos.sh` — shared MOS-eval body for
  greedy/sampled drivers.

Reference migrations:
- `jobs/sft/sft_warmup_paper_half_h100.sh` (preamble pattern)
- `jobs/evaluate/evaluate_dpo_paper_half_h100_{greedy,sampled}.sh` (template pattern)

See `jobs/_lib/README.md` for the conventions.

## Experiment tracking

Run state lives in the parent vault, not in this repo:

- `~/Library/Mobile Documents/com~apple~CloudDocs/svejstrup-os/studies/speech-quality-assesment/runs/`
  — one Markdown ledger entry per submitted job, plus an `INDEX.md`
  catalog and a `TEMPLATE.md`.
- The public experiment-log website mirrors that ledger.

Do not invent a parallel tracking surface in this repo.

## Setup

```sh
uv sync --locked
```

Python 3.12, GPU only (CUDA 11.8 wheel pinned in `pyproject.toml` for
Linux). Heavy deps include `torch==2.6.0`, `transformers`, `deepspeed`,
`librosa`, `soundfile`, `wandb`. Lockfile is canonical — there's no
`requirements.txt`.

## Tests

```sh
uv run pytest tests/
```

CPU-safe subset (no GPU, no HF model download):

```sh
uv run pytest tests/test_processed_data.py tests/test_collator.py \
              tests/test_dataset.py tests/test_jobs.py -q
```

CI runs the CPU-safe subset on push and pull request. GPU-pulling tests
are skipped in CI; run them on HPC.

## Archive

`scripts/_legacy/` holds the older NISQA caption-generation flow
(`caption_generator.py` + `legacy_data_cli.py`). Preserved for archival
reproducibility of pre-2026-04-13 runs but no longer on the active path.
