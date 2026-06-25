# `notebooks/` — interactive demos and exploratory scratch

Notebooks are not on the reproducibility path (the thesis results come from
`jobs/` + `scripts/`); they are for inspecting the model and the data by hand.

| Notebook | What it is |
|---|---|
| `inference_demo.ipynb` | **Start here.** Loads a trained checkpoint and runs it on a clip, so you can see the joint caption + interval output end to end. |
| `gemini31_pro_zero_shot_timestamp_pilot.ipynb` | The Gemini-3.1-Pro zero-shot temporal pilot (the baseline the thesis compares against). |
| `nisqa_sim_mix_lowmos_active_segment_generator.ipynb` | Scratch for the temporal mix generator; the production version is `scripts/data/generate_nisqa_sim_lowmos_active.py`. |
| `temporal_generation_samples.ipynb` | Eyeballs generated mixes (waveform + ground-truth interval). |
| `audio_player.ipynb` | Small helper for playing clips inline. |
| `build_temporal_inspector_site.py` | Not a notebook — builds the standalone HTML inspector for grouped temporal mixes. Referenced by the temporal data jobs. |

To run the notebooks, install the project first (`uv sync --locked`) and select
the project venv as the kernel.
