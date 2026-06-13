# Automatic Speech Quality Assessment

Bachelor project on **audio LLMs for descriptive speech quality assessment with temporal localization**. We fine-tune **Qwen2-Audio** with the **ALLD** alignment method on synthetic NISQA-SIM mixes, producing a model that outputs a base MOS-style caption plus time-localized degradation annotations of the form `(start, end, degradation_category)`.

Authors: Carl Schmidt-Svejstrup, David Lindahl. DTU, 2026.

## Repository layout

```
src/asa/                       # importable library code (no entrypoints)
  audio.py                       # 16 kHz mono WAV loader, audio constants
  prompts.py                     # PROMPT_TEMPLATE + MOS expert-prompt builder
  datasets.py                    # SFTDataset, DPODataset
  collators.py                   # Qwen2AudioCollator, ALLDDPOCollator
  inference.py                   # public load_model() + run_inference()
  processed_data.py              # dataset I/O + audio path resolution
  generate_temporal_data.py      # library: noise overlay / packet loss / clipping
  distill_temporal_targets.py    # library: smoke-set target distillation
  data.py                        # compatibility shim re-exporting the above

scripts/                       # runnable entrypoints, grouped by purpose
  train/                         # SFT and DPO trainers (called by jobs/*/sft/, jobs/*/alld/)
    supervised-finetune.py
    dpo-finetune.py
    submit_dpo_paper_half_pipeline.sh
  eval/                          # eval CLIs (called by jobs/*/eval/)
    evaluate.py
    evaluate_temporal.py
  data/                          # dataset builders + smoke prep
    generate_nisqa_sim_lowmos_active.py
    build_nisqa_temporal_json.py
    generate_dpo_data.py
    prepare_temporal_smoke.py
  diagnostics/                   # probes for when DPO collapses again
    diagnose_dpo_empty_output.py
    dpo_sanity_check.py
    sanity_check_dpo.py
  analysis/                      # post-eval aggregators + thesis figures
    eval_pred_vs_true.py
    caption_vs_mos.py
    audit_response_diversity.py
  _legacy/                       # pre-temporal caption generator; archival only

data/processed/                # training and eval data, grouped by use
  sft/                           # SFT training inputs
  dpo/                           # DPO chosen/rejected pairs
  eval/                          # held-out test splits (test_FOR, _LIVE, _P501, _nisqa_indomain)
  temporal/                      # temporal-localization mixes and metadata (current scope)
  intermediate/                  # build artifacts, AB legacy; not direct inputs

jobs/                          # LSF job submission, grouped by task then role
  global/                        # global MOS-caption task
    alld/                          # DPO training jobs
    sft/                           # SFT training jobs
    eval/                          # eval jobs
    data/                          # generate_*/build_* data-prep jobs
  temporal/                      # time-localization task (same alld/sft/eval/data split)
    alld/  sft/  eval/  data/
  upload/                        # HF Hub checkpoint uploaders
  tests/                         # pipeline smoke tests
  _lib/                          # shared preamble + memory-budget linter + eval template
  _archive/                      # historical scripts; not on the live path

tests/                         # pytest, CPU-safe subset runs in CI
```

`src/asa/data.py` is a compatibility shim re-exporting from `audio.py`,
`prompts.py`, `datasets.py`, `collators.py`. Existing
`from asa.data import SFTDataset` imports keep working unchanged.

Per-directory READMEs in `src/asa/`, `scripts/`, and `data/processed/`
restate this map close to the files they describe.

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
- `jobs/global/sft/sft_warmup_paper_half_h100.sh` (preamble pattern)
- `jobs/global/eval/evaluate_dpo_paper_half_h100_{greedy,sampled}.sh` (template pattern)

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

The pre-2026-04-13 AB direction and its data pipeline (the old NISQA
caption-generation flow and standalone CLI) were removed in the 2026-06-13
cleanup. They remain recoverable from git history if a pre-cutover run ever
needs reproducing.
