# `tests/` — pytest suite

Run everything:

```sh
uv run pytest tests/
```

The suite splits into two tiers. The **CPU-safe subset** needs no GPU and no
Hugging Face model download, and is what CI runs on every push / PR:

```sh
uv run pytest tests/test_processed_data.py tests/test_collator.py \
              tests/test_dataset.py tests/test_jobs.py -q
```

GPU-pulling tests (anything that loads Qwen2-Audio weights or needs CUDA) are
skipped automatically off-HPC and should be run on a DTU GPU node.

## What the tests cover

| Area | Tests |
|---|---|
| Data I/O + audio | `test_processed_data`, `test_audio_loading` |
| Datasets + collators | `test_dataset`, `test_collator`, `test_dpo_meta_prompt` |
| Data builders | `test_build_nisqa_temporal_json`, `test_build_dpo_cycle_splices`, `test_generate_dpo_temporal_factor` |
| Temporal mechanism | `test_temporal_tokens`, `test_temporal_loss`, `test_modeling_timeaudio`, `test_frame_probe` |
| Eval CLIs | `test_evaluate_cli`, `test_evaluate_temporal`, `test_evaluate_gemini_temporal` |
| Training smoke (GPU) | `test_sft_forwardpass`, `test_sft_full`, `test_alld_pipeline`, `test_deepspeed`, `test_inference` |
| Job-script validity | `test_jobs` (every `jobs/**/*.sh` parses and obeys the LSF mem rule) |

`test_jobs.py` is why job scripts must end in `.sh` — it only globs `*.sh`, so a
job without the extension is never syntax-checked.
