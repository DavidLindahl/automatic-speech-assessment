# `notebooks/` — local exploratory scratch (not tracked)

The notebooks in this directory are exploratory scratch for inspecting the
model and data by hand. They are **gitignored** (only this README is tracked),
because they are not part of the reproducibility path: the thesis results come
from `jobs/` + `scripts/`, not from notebooks.

If you have a local checkout, the notebooks you may find here include an
inference demo (load a checkpoint, run it on a clip, see the joint caption +
interval output), a Gemini zero-shot pilot, and mix-generation/inspection
scratch. None are required to reproduce the thesis.

The one runnable helper that used to live here,
`build_temporal_inspector_site.py` (builds the standalone HTML inspector for
grouped temporal mixes), has moved to
[`scripts/data/build_temporal_inspector_site.py`](../scripts/data/build_temporal_inspector_site.py),
since it is called by the temporal data jobs and belongs on the runnable path.

To run any local notebook, install the project first (`uv sync --locked`) and
select the project venv as the kernel.
