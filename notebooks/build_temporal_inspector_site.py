"""Build a static HTML inspector for temporal mix datasets.

The generated site includes:
- Audio playback
- Waveform visualization
- Overlay of degradation segments
- Prev/Next/Random navigation across a sampled working set
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import shutil

import pandas as pd
import typer


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Temporal Dataset Inspector</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #52606d;
      --accent: #2f80ed;
      --overlay: rgba(255, 179, 71, 0.32);
      --border: #d9e2ec;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .wrap {
      max-width: 1100px;
      margin: 0 auto;
      display: grid;
      gap: 16px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.2;
    }
    .sub {
      color: var(--muted);
      font-size: 14px;
      margin: 0;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .controls button, .controls input {
      border-radius: 8px;
      border: 1px solid var(--border);
      font-size: 14px;
      padding: 8px 10px;
      background: #fff;
      color: var(--ink);
    }
    .controls button {
      cursor: pointer;
    }
    .controls button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    .meta-grid {
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      margin-top: 12px;
      font-size: 14px;
    }
    .meta-grid div {
      padding: 8px 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fcfdff;
    }
    .label {
      color: var(--muted);
      margin-right: 6px;
    }
    .wave-wrap {
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fff;
      padding: 8px;
    }
    #wave {
      width: 100%;
      height: 280px;
      display: block;
      border-radius: 8px;
      background: #fff;
    }
    .row {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 10px;
      font-size: 14px;
      color: var(--muted);
    }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .loading {
      color: var(--muted);
      font-size: 14px;
      margin-top: 8px;
    }
    audio {
      width: 100%;
      margin-top: 12px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Temporal Dataset Inspector</h1>
      <p class="sub">
        Browse waveform + degradation overlays and listen to clips.
      </p>
    </section>

    <section class="card">
      <div class="controls">
        <span>Total records: <strong id="total-count">0</strong></span>
        <label>Working set size:
          <input id="sample-size" type="number" min="1" step="1" value="200" style="width: 90px;">
        </label>
        <button id="resample" class="primary">Resample Set</button>
        <button id="prev">Prev</button>
        <button id="next">Next</button>
        <button id="random">Random</button>
        <button id="toggle-distortion">Reveal Distortion</button>
        <label>Jump to record index:
          <input id="jump-index" type="number" min="0" step="1" style="width: 100px;">
        </label>
        <button id="jump">Go</button>
      </div>
      <div class="row">
        <span>Position in working set: <strong id="set-pos">-</strong></span>
        <span>Current record index: <strong id="rec-index">-</strong></span>
      </div>
      <div id="status" class="loading">Loading records...</div>
    </section>

    <section class="card">
      <div class="meta-grid">
        <div><span class="label">filename_deg:</span><span id="filename-deg" class="mono">-</span></div>
        <div><span class="label">filename_ref:</span><span id="filename-ref" class="mono">-</span></div>
        <div><span class="label">MOS:</span><span id="mos">-</span></div>
        <div><span class="label">duration (s):</span><span id="duration">-</span></div>
        <div><span class="label">active fraction:</span><span id="active-frac">-</span></div>
        <div><span class="label">degradation types:</span><span id="deg-types" class="mono">-</span></div>
      </div>

      <div class="wave-wrap">
        <canvas id="wave"></canvas>
      </div>
      <audio id="player" controls preload="metadata"></audio>
      <div class="row">
        <span class="label">mix file:</span><span id="mix-file" class="mono">-</span>
      </div>
      <div class="row">
        <span class="label">mix_deg_segments:</span><span id="segments" class="mono">-</span>
      </div>
      <div class="row">
        <span class="label">degradation types:</span><span id="deg-types-inline" class="mono">-</span>
      </div>
      <div class="row">
        <span class="label">playhead:</span><span id="playhead-info" class="mono">-</span>
      </div>
    </section>
  </div>

  <script>
    const canvas = document.getElementById("wave");
    const statusEl = document.getElementById("status");
    const audioEl = document.getElementById("player");
    const sampleSizeEl = document.getElementById("sample-size");
    const jumpIndexEl = document.getElementById("jump-index");
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();

    let records = [];
    let workingSet = [];
    let currentPos = 0;
    let waveCache = null;
    let playheadRaf = null;
    let showDistortion = false;

    function shuffleIndices(n) {
      const arr = Array.from({ length: n }, (_, i) => i);
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
      return arr;
    }

    function clamp(value, minValue, maxValue) {
      return Math.min(maxValue, Math.max(minValue, value));
    }

    function updateRevealUI() {
      const button = document.getElementById("toggle-distortion");
      button.textContent = showDistortion ? "Hide Distortion" : "Reveal Distortion";
      button.classList.toggle("primary", showDistortion);
    }

    function updateMeta(rec) {
      document.getElementById("rec-index").textContent = String(rec.index);
      document.getElementById("set-pos").textContent = `${currentPos + 1}/${workingSet.length}`;
      document.getElementById("filename-deg").textContent = rec.filename_deg ?? "-";
      document.getElementById("filename-ref").textContent = rec.filename_ref ?? "-";
      document.getElementById("mos").textContent = rec.mos ?? "-";
      document.getElementById("duration").textContent = rec.duration_seconds ?? "-";
      document.getElementById("active-frac").textContent = rec.segment_active_fraction ?? "-";
      const rawTypes = rec.source_degradation_types ?? [];
      const typeText = Array.isArray(rawTypes) ? (rawTypes.join(", ") || "none") : String(rawTypes);
      document.getElementById("deg-types").textContent = typeText;
      document.getElementById("deg-types-inline").textContent = typeText;
      document.getElementById("mix-file").textContent = rec.audio_relpath ?? "-";
      document.getElementById("segments").textContent = showDistortion
        ? JSON.stringify(rec.mix_deg_segments ?? [])
        : "(hidden: click 'Reveal Distortion')";
    }

    function buildWaveCache(audioData, durationSeconds, segments) {
      const dpr = window.devicePixelRatio || 1;
      const width = Math.floor(canvas.clientWidth * dpr);
      const height = Math.floor(canvas.clientHeight * dpr);
      const step = Math.max(1, Math.floor(audioData.length / width));
      const minVals = new Float32Array(width);
      const maxVals = new Float32Array(width);

      for (let x = 0; x < width; x++) {
        const i0 = x * step;
        const i1 = Math.min(audioData.length, i0 + step);
        let min = 1.0;
        let max = -1.0;

        for (let i = i0; i < i1; i++) {
          const v = audioData[i];
          if (v < min) min = v;
          if (v > max) max = v;
        }
        minVals[x] = min;
        maxVals[x] = max;
      }
      return {
        audioData,
        durationSeconds,
        segments: segments || [],
        dpr,
        width,
        height,
        minVals,
        maxVals,
      };
    }

    function getPlayheadSource(timeSeconds, segments) {
      const t = Number(timeSeconds || 0);
      for (const seg of segments || []) {
        const start = Number(seg.start || 0);
        const end = Number(seg.end || 0);
        if (Number.isFinite(start) && Number.isFinite(end) && t >= start && t <= end) {
          return "DEG";
        }
      }
      return "REF";
    }

    function drawFromCache(playheadSeconds = null) {
      if (!waveCache) {
        return;
      }
      const { dpr, width, height, durationSeconds, segments, minVals, maxVals } = waveCache;
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);

      if (showDistortion) {
        ctx.fillStyle = "rgba(255, 179, 71, 0.32)";
        for (const seg of segments || []) {
          const start = Number(seg.start || 0);
          const end = Number(seg.end || 0);
          if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
            continue;
          }
          const x0 = Math.floor((start / durationSeconds) * width);
          const x1 = Math.ceil((end / durationSeconds) * width);
          ctx.fillRect(x0, 0, Math.max(1, x1 - x0), height);
        }
      }

      ctx.strokeStyle = "#111111";
      ctx.lineWidth = Math.max(1, dpr);
      ctx.beginPath();
      for (let x = 0; x < width; x++) {
        const y0 = (1 - maxVals[x]) * 0.5 * height;
        const y1 = (1 - minVals[x]) * 0.5 * height;
        ctx.moveTo(x, y0);
        ctx.lineTo(x, y1);
      }
      ctx.stroke();

      const clampedTime = clamp(Number(playheadSeconds || 0), 0, Math.max(0.001, durationSeconds));
      const markerX = Math.floor((clampedTime / Math.max(0.001, durationSeconds)) * width);

      ctx.strokeStyle = "#2f80ed";
      ctx.lineWidth = Math.max(2, 2 * dpr);
      ctx.beginPath();
      ctx.moveTo(markerX, 0);
      ctx.lineTo(markerX, height);
      ctx.stroke();

      const source = showDistortion ? getPlayheadSource(clampedTime, segments) : "hidden";
      document.getElementById("playhead-info").textContent = `${clampedTime.toFixed(2)}s (${source})`;
    }

    function stopPlayheadLoop() {
      if (playheadRaf !== null) {
        cancelAnimationFrame(playheadRaf);
        playheadRaf = null;
      }
    }

    function drawPlayheadAtCurrentTime() {
      drawFromCache(audioEl.currentTime || 0);
    }

    function startPlayheadLoop() {
      stopPlayheadLoop();
      const tick = () => {
        drawPlayheadAtCurrentTime();
        if (!audioEl.paused && !audioEl.ended) {
          playheadRaf = requestAnimationFrame(tick);
        } else {
          playheadRaf = null;
        }
      };
      playheadRaf = requestAnimationFrame(tick);
    }

    async function renderCurrent() {
      if (!workingSet.length) {
        statusEl.textContent = "No records available.";
        return;
      }
      const rec = workingSet[currentPos];
      updateMeta(rec);
      stopPlayheadLoop();
      waveCache = null;
      document.getElementById("playhead-info").textContent = "-";

      const audioUrl = encodeURI(rec.audio_relpath);
      audioEl.src = audioUrl;

      statusEl.textContent = "Loading audio and drawing waveform...";
      const response = await fetch(audioUrl);
      const audioBytes = await response.arrayBuffer();
      const decoded = await audioContext.decodeAudioData(audioBytes.slice(0));
      const audioData = decoded.getChannelData(0);

      waveCache = buildWaveCache(
        audioData,
        Number(rec.duration_seconds || decoded.duration || 1.0),
        rec.mix_deg_segments || [],
      );
      drawPlayheadAtCurrentTime();
      statusEl.textContent = "Ready.";
    }

    function sampleWorkingSet() {
      if (!records.length) {
        return;
      }
      const requested = Number.parseInt(sampleSizeEl.value || "200", 10);
      const n = clamp(Number.isFinite(requested) ? requested : 200, 1, records.length);
      sampleSizeEl.value = String(n);

      const sampled = shuffleIndices(records.length).slice(0, n);
      workingSet = sampled.map((idx) => records[idx]);
      currentPos = 0;
      renderCurrent();
    }

    function toggleDistortion() {
      showDistortion = !showDistortion;
      updateRevealUI();
      if (!workingSet.length) {
        return;
      }
      updateMeta(workingSet[currentPos]);
      drawPlayheadAtCurrentTime();
    }

    function move(delta) {
      if (!workingSet.length) {
        return;
      }
      currentPos = (currentPos + delta + workingSet.length) % workingSet.length;
      renderCurrent();
    }

    function randomPick() {
      if (!workingSet.length) {
        return;
      }
      currentPos = Math.floor(Math.random() * workingSet.length);
      renderCurrent();
    }

    function jumpToRecordIndex() {
      if (!workingSet.length) {
        return;
      }
      const target = Number.parseInt(jumpIndexEl.value || "", 10);
      if (!Number.isFinite(target)) {
        return;
      }
      const foundPos = workingSet.findIndex((rec) => Number(rec.index) === target);
      if (foundPos >= 0) {
        currentPos = foundPos;
        renderCurrent();
        return;
      }

      const globalPos = records.findIndex((rec) => Number(rec.index) === target);
      if (globalPos >= 0) {
        workingSet = [records[globalPos]];
        sampleSizeEl.value = "1";
        currentPos = 0;
        renderCurrent();
      }
    }

    async function init() {
      try {
        const response = await fetch("records.json");
        records = await response.json();
        document.getElementById("total-count").textContent = String(records.length);
        sampleSizeEl.value = String(Math.min(200, records.length || 1));
        sampleWorkingSet();
      } catch (error) {
        statusEl.textContent = `Failed to load records.json: ${String(error)}`;
      }
    }

    document.getElementById("resample").addEventListener("click", sampleWorkingSet);
    document.getElementById("prev").addEventListener("click", () => move(-1));
    document.getElementById("next").addEventListener("click", () => move(1));
    document.getElementById("random").addEventListener("click", randomPick);
    document.getElementById("toggle-distortion").addEventListener("click", toggleDistortion);
    document.getElementById("jump").addEventListener("click", jumpToRecordIndex);
    audioEl.addEventListener("play", startPlayheadLoop);
    audioEl.addEventListener("pause", () => {
      stopPlayheadLoop();
      drawPlayheadAtCurrentTime();
    });
    audioEl.addEventListener("timeupdate", () => {
      if (audioEl.paused) {
        drawPlayheadAtCurrentTime();
      }
    });
    audioEl.addEventListener("seeking", drawPlayheadAtCurrentTime);
    audioEl.addEventListener("seeked", drawPlayheadAtCurrentTime);
    audioEl.addEventListener("ended", () => {
      stopPlayheadLoop();
      drawPlayheadAtCurrentTime();
    });
    window.addEventListener("resize", () => {
      if (!waveCache) {
        return;
      }
      waveCache = buildWaveCache(
        waveCache.audioData,
        waveCache.durationSeconds,
        waveCache.segments,
      );
      drawPlayheadAtCurrentTime();
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft") move(-1);
      if (event.key === "ArrowRight") move(1);
      if (event.key === "r") randomPick();
      if (event.key === "v") toggleDistortion();
    });

    updateRevealUI();
    init();
  </script>
</body>
</html>
"""


def parse_json_field(value: object, fallback: object) -> object:
    """Parse a JSON-serialized manifest field.

    Args:
        value: Input dataframe cell.
        fallback: Fallback value returned on parse failure.

    Returns:
        Parsed object or fallback.
    """

    if pd.isna(value):
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def infer_mix_filename(row: pd.Series, index_width: int) -> str:
    """Infer output mix filename for manifests without explicit mix filename.

    Args:
        row: One manifest row.
        index_width: Zero-padding width used in generated names.

    Returns:
        Mix filename.
    """

    if "mix_filename" in row.index and pd.notna(row["mix_filename"]):
        text = str(row["mix_filename"]).strip()
        if text:
            return text
    stem = Path(str(row["filename_deg"])).stem
    index = int(row["index"])
    return f"{index:0{index_width}d}_mix_{stem}.wav"


def build_records(
    manifest_df: pd.DataFrame,
    mixes_dir: Path,
    audio_dir: Path,
) -> list[dict[str, object]]:
    """Build JSON records consumed by the static inspector UI.

    Args:
        manifest_df: Manifest dataframe.
        mixes_dir: Directory where mixed audio files live.
        audio_dir: Directory where audio assets are materialized for the inspector.

    Returns:
        List of serializable records.
    """

    if "index" not in manifest_df.columns:
        raise ValueError("Manifest is missing required 'index' column.")
    max_index = int(manifest_df["index"].max())
    index_width = max(3, len(str(max_index)))
    records: list[dict[str, object]] = []

    for _, row in manifest_df.sort_values("index").iterrows():
        mix_filename = infer_mix_filename(row=row, index_width=index_width)
        mix_path = mixes_dir / mix_filename
        if not mix_path.exists():
            continue
        asset_path = audio_dir / mix_filename
        if asset_path.exists() or asset_path.is_symlink():
            asset_path.unlink()
        rel_target = os.path.relpath(mix_path, audio_dir)
        try:
            asset_path.symlink_to(rel_target)
        except OSError:
            shutil.copy2(mix_path, asset_path)

        records.append(
            {
                "index": int(row["index"]),
                "filename_ref": str(row.get("filename_ref", "")),
                "filename_deg": str(row.get("filename_deg", "")),
                "mos": float(row["mos"]) if pd.notna(row.get("mos", None)) else None,
                "duration_seconds": float(row["duration_seconds"])
                if pd.notna(row.get("duration_seconds", None))
                else None,
                "mix_deg_segments": parse_json_field(
                    row.get("mix_deg_segments", "[]"), []
                ),
                "source_degradation_types": parse_json_field(
                    row.get("source_degradation_types", "[]"), []
                ),
                "segment_active_fraction": float(row["segment_active_fraction"])
                if pd.notna(row.get("segment_active_fraction", None))
                else None,
                "audio_relpath": f"audio/{mix_filename}",
            }
        )
    return records


def main(
    manifest_path: Path = Path(
        "data/processed/nisqa_sim_mix_lowmos_active_3000/manifest.csv"
    ),
    mixes_dir: Path | None = None,
    site_dir: Path | None = None,
    max_records: int | None = None,
    seed: int = 42,
) -> None:
    """Build static temporal inspector site from a manifest.

    Args:
        manifest_path: Path to manifest CSV.
        mixes_dir: Directory containing mixed WAV files. Defaults to manifest parent.
        site_dir: Output directory for generated site. Defaults to manifest parent / "inspector".
        max_records: Optional cap on number of records exported.
        seed: Random seed used for optional record sampling.
    """

    if not manifest_path.exists():
        raise ValueError(f"Manifest not found: {manifest_path}")

    mixes_directory = mixes_dir or manifest_path.parent
    output_site_dir = site_dir or (manifest_path.parent / "inspector")
    output_site_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_site_dir / "audio"
    shutil.rmtree(audio_dir, ignore_errors=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    if max_records is not None and max_records > 0 and len(manifest_df) > max_records:
        manifest_df = manifest_df.sample(n=max_records, random_state=seed).copy()
        manifest_df = manifest_df.sort_values("index").reset_index(drop=True)

    records = build_records(
        manifest_df=manifest_df,
        mixes_dir=mixes_directory,
        audio_dir=audio_dir,
    )

    records_path = output_site_dir / "records.json"
    index_path = output_site_dir / "index.html"
    records_path.write_text(json.dumps(records, indent=2))
    index_path.write_text(HTML_TEMPLATE)

    print(f"Site written to: {output_site_dir}")
    print(f"Records exported: {len(records)}")
    print("Serve locally with:")
    print(f"  cd {output_site_dir}")
    print("  uv run python -m http.server 8000")
    print("Then open:")
    print("  http://localhost:8000")


if __name__ == "__main__":
    typer.run(main)
