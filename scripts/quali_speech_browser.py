"""Local browser for the QualiSpeech dataset.

The app serves a small HTML interface that lets you browse the extracted audio
files, play a selected sample, and inspect its metadata from the CSV splits.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
import soundfile as sf
import uvicorn


@dataclass(frozen=True)
class SampleRecord:
    """Metadata for one QualiSpeech sample."""

    split: str
    audio_path: Path
    metadata: dict[str, Any]
    duration_seconds: float


class QualiSpeechIndex:
    """Index the local QualiSpeech dataset snapshot."""

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self._records = self._load_records()

    def _load_records(self) -> dict[str, SampleRecord]:
        records: dict[str, SampleRecord] = {}
        for split in ("train", "val", "test"):
            csv_path = self.dataset_dir / f"{split}.csv"
            audio_root = self.dataset_dir / "wav" / split
            df = pd.read_csv(csv_path)
            for row in df.to_dict(orient="records"):
                audio_name = str(row["id"])
                audio_path = audio_root / audio_name
                metadata = dict(row)
                metadata["split"] = split
                metadata["audio_exists"] = audio_path.exists()
                duration_seconds = self._read_duration(audio_path) if audio_path.exists() else 0.0
                metadata["duration_seconds"] = duration_seconds
                records[f"{split}:{audio_name}"] = SampleRecord(
                    split=split,
                    audio_path=audio_path,
                    metadata=metadata,
                    duration_seconds=duration_seconds,
                )
        return records

    @staticmethod
    def _read_duration(audio_path: Path) -> float:
        """Read the WAV duration without loading the full audio into memory."""

        info = sf.info(str(audio_path))
        return info.frames / info.samplerate if info.samplerate else 0.0

    @property
    def splits(self) -> list[str]:
        return ["train", "val", "test"]

    def list_samples(
        self,
        split: str,
        query: str = "",
        limit: int = 500,
        filters: dict[str, float | str | None] | None = None,
    ) -> list[dict[str, Any]]:
        if split not in self.splits:
            raise HTTPException(status_code=404, detail="Unknown split")

        filters = filters or {}
        query_lower = query.lower().strip()
        results: list[dict[str, Any]] = []
        for record in self._records.values():
            if record.split != split:
                continue

            if not self._matches_filters(record, filters):
                continue

            if query_lower and query_lower not in record.audio_path.name.lower():
                text_blob = " ".join(str(value) for value in record.metadata.values()).lower()
                if query_lower not in text_blob:
                    continue
            results.append(
                {
                    "id": record.audio_path.name,
                    "audio_exists": record.metadata["audio_exists"],
                    "overall": record.metadata.get("Overall quality"),
                    "speed": record.metadata.get("Speed"),
                    "naturalness": record.metadata.get("Naturalness"),
                    "listening_effort": record.metadata.get("Listening effort"),
                    "continuity": record.metadata.get("Continuity"),
                    "noise": record.metadata.get("Background noise"),
                    "distortion": record.metadata.get("Distortion"),
                    "feeling_of_voice": record.metadata.get("Feeling of voice"),
                    "duration_seconds": round(record.duration_seconds, 2),
                }
            )
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _matches_filters(record: SampleRecord, filters: dict[str, float | str | None]) -> bool:
        """Return whether a record matches the active filter set."""

        duration_min = filters.get("duration_min")
        duration_max = filters.get("duration_max")
        if duration_min is not None and record.duration_seconds < float(duration_min):
            return False
        if duration_max is not None and record.duration_seconds > float(duration_max):
            return False

        for key, value in filters.items():
            if value is None or key in {"duration_min", "duration_max"}:
                continue
            if key.endswith("_min") or key.endswith("_max"):
                feature_name = key.rsplit("_", 1)[0]
                feature_value = record.metadata.get(feature_name)
                if feature_value is None:
                    continue
                numeric_value = float(feature_value)
                if key.endswith("_min") and numeric_value < float(value):
                    return False
                if key.endswith("_max") and numeric_value > float(value):
                    return False
        return True

    def get_sample(self, split: str, audio_id: str) -> SampleRecord:
        key = f"{split}:{audio_id}"
        if key not in self._records:
            raise HTTPException(status_code=404, detail="Sample not found")
        return self._records[key]


def build_app(index: QualiSpeechIndex) -> FastAPI:
    """Create the FastAPI app."""

    app = FastAPI(title="QualiSpeech Browser")

    @app.get("/", response_class=HTMLResponse)
    def index_page() -> str:
        return HTML_TEMPLATE

    @app.get("/api/splits")
    def get_splits() -> dict[str, list[str]]:
        return {"splits": index.splits}

    @app.get("/api/samples")
    def get_samples(
        split: str,
        q: str = "",
        limit: int = 500,
        duration_min: float | None = None,
        duration_max: float | None = None,
        speed_min: float | None = None,
        speed_max: float | None = None,
        naturalness_min: float | None = None,
        naturalness_max: float | None = None,
        background_noise_min: float | None = None,
        background_noise_max: float | None = None,
        distortion_min: float | None = None,
        distortion_max: float | None = None,
        listening_effort_min: float | None = None,
        listening_effort_max: float | None = None,
        continuity_min: float | None = None,
        continuity_max: float | None = None,
        overall_quality_min: float | None = None,
        overall_quality_max: float | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        filters: dict[str, float | str | None] = {
            "duration_min": duration_min,
            "duration_max": duration_max,
            "Speed_min": speed_min,
            "Speed_max": speed_max,
            "Naturalness_min": naturalness_min,
            "Naturalness_max": naturalness_max,
            "Background noise_min": background_noise_min,
            "Background noise_max": background_noise_max,
            "Distortion_min": distortion_min,
            "Distortion_max": distortion_max,
            "Listening effort_min": listening_effort_min,
            "Listening effort_max": listening_effort_max,
            "Continuity_min": continuity_min,
            "Continuity_max": continuity_max,
            "Overall quality_min": overall_quality_min,
            "Overall quality_max": overall_quality_max,
        }
        return {"samples": index.list_samples(split=split, query=q, limit=limit, filters=filters)}

    @app.get("/api/sample/{split}/{audio_id}")
    def get_sample(split: str, audio_id: str) -> dict[str, Any]:
        record = index.get_sample(split, audio_id)
        return {
            "split": split,
            "audio_id": audio_id,
            "audio_exists": record.audio_path.exists(),
            "metadata": record.metadata,
        }

    @app.get("/audio/{split}/{audio_id}")
    def get_audio(split: str, audio_id: str) -> Response:
        record = index.get_sample(split, audio_id)
        if not record.audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        return Response(
            content=record.audio_path.read_bytes(),
            media_type="audio/wav",
            headers={"Content-Disposition": f'inline; filename="{audio_id}"'},
        )

    return app


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>QualiSpeech Browser</title>
    <style>
      :root {
        color-scheme: dark;
        --bg: #0f1115;
        --panel: #171a21;
        --panel-2: #1f2430;
        --border: #2d3442;
        --text: #e7eaf0;
        --muted: #9aa4b2;
        --accent: #7dd3fc;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: radial-gradient(circle at top, #182235 0%, var(--bg) 55%);
        color: var(--text);
      }
      header {
        padding: 20px 24px 12px;
        border-bottom: 1px solid var(--border);
        background: rgba(15, 17, 21, 0.9);
        position: sticky;
        top: 0;
        backdrop-filter: blur(10px);
      }
      h1 { margin: 0 0 6px; font-size: 20px; }
      .sub { color: var(--muted); font-size: 13px; }
      .wrap {
        display: grid;
        grid-template-columns: 360px 1fr;
        min-height: calc(100vh - 74px);
      }
      .left, .right { padding: 18px; }
      .card {
        background: rgba(23, 26, 33, 0.92);
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
      }
      .controls {
        display: flex;
        gap: 10px;
        padding: 14px;
        border-bottom: 1px solid var(--border);
        flex-wrap: wrap;
      }
      .filters {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        padding: 14px;
        border-bottom: 1px solid var(--border);
      }
      .filters input {
        min-width: 0;
      }
      select, input {
        width: 100%;
        background: var(--panel-2);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 12px;
      }
      .list {
        max-height: calc(100vh - 220px);
        overflow: auto;
      }
      .item {
        padding: 12px 14px;
        border-bottom: 1px solid rgba(45, 52, 66, 0.7);
        cursor: pointer;
      }
      .item:hover, .item.active { background: rgba(125, 211, 252, 0.08); }
      .item .id { font-size: 13px; }
      .item .meta { color: var(--muted); font-size: 12px; margin-top: 4px; }
      .detail { display: grid; gap: 14px; }
      .player {
        padding: 16px;
      }
      audio { width: 100%; margin-top: 10px; }
      .grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }
      .kv {
        padding: 12px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(31, 36, 48, 0.65);
      }
      .k { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
      .v { font-size: 14px; line-height: 1.45; white-space: pre-wrap; }
      .full {
        grid-column: 1 / -1;
      }
      .hint { color: var(--muted); font-size: 13px; }
      @media (max-width: 980px) {
        .wrap { grid-template-columns: 1fr; }
        .list { max-height: 42vh; }
      }
    </style>
  </head>
  <body>
    <header>
      <h1>QualiSpeech Browser</h1>
      <div class="sub">Browse local WAV files, play audio, and inspect metadata without streaming from Hugging Face.</div>
    </header>
    <div class="wrap">
      <section class="left">
        <div class="card">
          <div class="controls">
            <select id="split"></select>
          </div>
          <div class="controls" style="padding-top: 0;">
            <input id="query" type="search" placeholder="Search by file id or metadata text" />
          </div>
          <div class="filters">
            <input id="duration-min" type="number" min="0" step="0.1" placeholder="Duration min (s)" />
            <input id="duration-max" type="number" min="0" step="0.1" placeholder="Duration max (s)" />
            <input id="overall-min" type="number" min="1" max="5" step="1" placeholder="Overall quality min" />
            <input id="overall-max" type="number" min="1" max="5" step="1" placeholder="Overall quality max" />
            <input id="speed-min" type="number" min="1" max="5" step="1" placeholder="Speed min" />
            <input id="speed-max" type="number" min="1" max="5" step="1" placeholder="Speed max" />
            <input id="naturalness-min" type="number" min="1" max="5" step="1" placeholder="Naturalness min" />
            <input id="naturalness-max" type="number" min="1" max="5" step="1" placeholder="Naturalness max" />
          </div>
          <div id="list" class="list"></div>
        </div>
      </section>
      <section class="right">
        <div class="card detail">
          <div class="player">
            <div id="selected-title" style="font-size: 18px; font-weight: 600;">Select a sample</div>
            <div id="selected-subtitle" class="hint">Audio and metadata will appear here.</div>
            <audio id="audio" controls></audio>
          </div>
          <div id="metadata" class="grid" style="padding: 0 16px 16px;"></div>
        </div>
      </section>
    </div>
    <script>
      const splitEl = document.getElementById("split");
      const queryEl = document.getElementById("query");
      const durationMinEl = document.getElementById("duration-min");
      const durationMaxEl = document.getElementById("duration-max");
      const overallMinEl = document.getElementById("overall-min");
      const overallMaxEl = document.getElementById("overall-max");
      const speedMinEl = document.getElementById("speed-min");
      const speedMaxEl = document.getElementById("speed-max");
      const naturalnessMinEl = document.getElementById("naturalness-min");
      const naturalnessMaxEl = document.getElementById("naturalness-max");
      const listEl = document.getElementById("list");
      const titleEl = document.getElementById("selected-title");
      const subtitleEl = document.getElementById("selected-subtitle");
      const audioEl = document.getElementById("audio");
      const metadataEl = document.getElementById("metadata");
      let activeKey = null;

      function renderMetadata(metadata) {
        const entries = [
          ["Speed", metadata["Speed"]],
          ["Naturalness", metadata["Naturalness"]],
          ["Background noise", metadata["Background noise"]],
          ["Distortion", metadata["Distortion"]],
          ["Listening effort", metadata["Listening effort"]],
          ["Continuity", metadata["Continuity"]],
          ["Overall quality", metadata["Overall quality"]],
          ["Feeling of voice", metadata["Feeling of voice"]],
          ["Noise Description", metadata["Noise Description"]],
          ["Distortion description", metadata["Distortion description"]],
          ["Unnatural pause", metadata["Unnatural pause"]],
          ["Natural language description", metadata["Natural language description"]],
        ];
        metadataEl.innerHTML = entries.map(([key, value]) => `
          <div class="kv ${key === "Natural language description" ? "full" : ""}">
            <div class="k">${key}</div>
            <div class="v">${value ?? ""}</div>
          </div>
        `).join("");
      }

      async function loadSelection(split, sampleId) {
        const res = await fetch(`/api/sample/${encodeURIComponent(split)}/${encodeURIComponent(sampleId)}`);
        const payload = await res.json();
        const metadata = payload.metadata;
        activeKey = `${split}:${sampleId}`;
        titleEl.textContent = sampleId;
        subtitleEl.textContent = `${split} split · audio ${payload.audio_exists ? "available" : "missing"}`;
        audioEl.src = `/audio/${encodeURIComponent(split)}/${encodeURIComponent(sampleId)}`;
        renderMetadata(metadata);
        [...document.querySelectorAll(".item")].forEach((el) => {
          el.classList.toggle("active", el.dataset.key === activeKey);
        });
      }

      async function loadSamples() {
        const split = splitEl.value;
        const q = queryEl.value;
        const params = new URLSearchParams({
          split,
          q,
          limit: "800",
        });
        [
          ["duration_min", durationMinEl.value],
          ["duration_max", durationMaxEl.value],
          ["overall_quality_min", overallMinEl.value],
          ["overall_quality_max", overallMaxEl.value],
          ["speed_min", speedMinEl.value],
          ["speed_max", speedMaxEl.value],
          ["naturalness_min", naturalnessMinEl.value],
          ["naturalness_max", naturalnessMaxEl.value],
        ].forEach(([key, value]) => {
          if (value !== "") {
            params.set(key, value);
          }
        });
        const res = await fetch(`/api/samples?${params.toString()}`);
        const payload = await res.json();
        listEl.innerHTML = payload.samples.map((sample) => `
          <div class="item" data-key="${split}:${sample.id}" data-id="${sample.id}">
            <div class="id">${sample.id}</div>
            <div class="meta">Overall ${sample.overall} · Speed ${sample.speed} · Naturalness ${sample.naturalness} · ${sample.duration_seconds}s</div>
          </div>
        `).join("");
        [...document.querySelectorAll(".item")].forEach((el) => {
          el.addEventListener("click", () => loadSelection(split, el.dataset.id));
        });
        if (!activeKey && payload.samples.length > 0) {
          loadSelection(split, payload.samples[0].id);
        }
      }

      async function init() {
        const res = await fetch("/api/splits");
        const payload = await res.json();
        splitEl.innerHTML = payload.splits.map((split) => `<option value="${split}">${split}</option>`).join("");
        splitEl.value = payload.splits[0];
        splitEl.addEventListener("change", () => {
          activeKey = null;
          loadSamples();
        });
        [
          queryEl,
          durationMinEl,
          durationMaxEl,
          overallMinEl,
          overallMaxEl,
          speedMinEl,
          speedMaxEl,
          naturalnessMinEl,
          naturalnessMaxEl,
        ].forEach((el) => el.addEventListener("input", () => loadSamples()));
        await loadSamples();
      }

      init();
    </script>
  </body>
</html>
"""


def main() -> None:
    """Run the local QualiSpeech browser."""

    parser = ArgumentParser(description="Browse local QualiSpeech audio and metadata.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/raw/QualiSpeech"),
        help="Local QualiSpeech directory created by the download script.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    args = parser.parse_args()

    index = QualiSpeechIndex(args.dataset_dir)
    app = build_app(index)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
