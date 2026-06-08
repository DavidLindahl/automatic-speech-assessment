"""Build a grouped temporal-augmentation inspector, one card per source REF.

Purpose: verify the placement-augmentation worked. Each original NISQA-SIM
reference (``filename_ref``) gets one card showing ALL of its augmented mix
windows on a single duration-normalized timeline, so you can confirm at a glance
that the reuses are distinct and non-overlapping.

The view focuses on the ``--top-n`` (default 30) most-reused REFs, one card each:
- The ORIGINAL clean reference waveform is rendered (precomputed peak envelope,
  embedded as JSON), with each augmented window drawn as a color-coded overlay
  box on it. This shows the windows land on real speech and sit in different
  places per reuse.
- **Each window is clickable**: clicking a colored box (or its "play degraded"
  button) plays THAT degraded mix, so you hear the actual training clip, not just
  the clean reference. A separate player offers the clean reference for contrast.

Input is the manifest CSV from generate_nisqa_sim_lowmos_active.py. Run locally
off a pulled manifest; needs the clean reference WAVs (NISQA_TRAIN_SIM/ref/) via
``--refs-dir`` and the degraded mix WAVs via ``--mixes-dir``. Only the top-N
REFs' files are read, so the site stays small and self-contained.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import typer


def waveform_peaks(wav_path: Path, buckets: int = 480) -> list[float] | None:
    """Return a downsampled abs-amplitude envelope in [0,1] for a WAV file.

    The signal is split into ``buckets`` equal segments and each segment is
    reduced to its peak absolute amplitude, then the whole envelope is normalized
    to its own max. This is enough to draw a recognizable waveform behind the
    degradation overlays without shipping the raw audio samples.

    Args:
        wav_path: Path to a mono/stereo WAV file.
        buckets: Number of envelope points to produce.

    Returns:
        List of ``buckets`` floats in [0,1], or ``None`` if the file is unreadable.
    """
    try:
        data, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception:
        return None
    if data.ndim > 1:
        data = data.mean(axis=1)
    n = len(data)
    if n == 0:
        return None
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    peaks = np.empty(buckets, dtype=np.float32)
    for i in range(buckets):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        peaks[i] = np.max(np.abs(data[a:b]))
    top = float(peaks.max())
    if top <= 0:
        return [0.0] * buckets
    return [round(float(v) / top, 4) for v in peaks]

# Distinct, color-blind-friendly palette cycled per window within a REF card.
WINDOW_COLORS = [
    "#2f80ed",
    "#eb5757",
    "#27ae60",
    "#9b51e0",
    "#f2994a",
    "#00b8d9",
    "#bb6bd9",
]

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Temporal Augmentation Inspector (grouped by reference)</title>
<style>
  :root {
    --bg:#f4f6f8; --panel:#fff; --ink:#1f2933; --muted:#52606d;
    --border:#d9e2ec; --bar:#e4e9f0;
  }
  *{box-sizing:border-box}
  body{margin:0;padding:24px;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  h1{font-size:20px;margin:0 0 4px}
  .sub{color:var(--muted);font-size:13px;margin:0 0 16px}
  .controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:18px}
  input[type=search]{padding:8px 12px;border:1px solid var(--border);border-radius:8px;
    font-size:14px;min-width:280px}
  .stat{background:var(--panel);border:1px solid var(--border);border-radius:8px;
    padding:8px 12px;font-size:13px}
  .stat b{font-size:15px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:12px;
    padding:14px 16px;margin-bottom:12px}
  .card h2{font-size:13px;margin:0 0 2px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    word-break:break-all}
  .meta{color:var(--muted);font-size:12px;margin-bottom:10px}
  .meta .pill{display:inline-block;background:var(--bar);border-radius:10px;
    padding:1px 8px;margin-right:6px}
  .timeline{position:relative;height:34px;background:var(--bar);border-radius:6px;
    overflow:hidden;margin-bottom:8px}
  .wave{position:relative;height:90px;background:#0f1722;border-radius:6px;
    overflow:hidden;margin-bottom:8px}
  .wave svg{position:absolute;inset:0;width:100%;height:100%}
  .wave .win{mix-blend-mode:normal;opacity:.42;cursor:pointer;transition:opacity .12s}
  .wave .win:hover{opacity:.62}
  .wave .win.playing{opacity:.72;box-shadow:inset 0 0 0 2px #fff}
  .win{position:absolute;top:0;height:100%;opacity:.78;border-radius:3px;
    display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;
    font-weight:600;text-shadow:0 1px 1px rgba(0,0,0,.3)}
  .wave .winlabel{position:absolute;top:2px;font-size:10px;font-weight:700;color:#fff;
    text-shadow:0 1px 2px rgba(0,0,0,.6);pointer-events:none}
  .hint{color:var(--muted);font-size:11px;margin:0 0 6px}
  .row{cursor:pointer}
  .row:hover{background:#f0f4fa;border-radius:4px}
  .row.playing{background:#eef4ff;border-radius:4px}
  .playbtn{margin-left:auto;font-size:11px;border:1px solid var(--border);background:#fff;
    border-radius:6px;padding:2px 8px;cursor:pointer}
  .nowplaying{font-size:11px;color:var(--muted);margin-top:4px;min-height:14px}
  .axis{display:flex;justify-content:space-between;color:var(--muted);font-size:10px;margin-bottom:8px}
  .rows{font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .row{display:flex;align-items:center;gap:8px;padding:2px 0}
  .dot{width:10px;height:10px;border-radius:50%;flex:0 0 auto}
  .row audio{height:28px;margin-left:auto}
  .warn{color:#eb5757;font-weight:700}
  .hidden{display:none}
</style>
</head>
<body>
<h1>Temporal Augmentation Inspector</h1>
<p class="sub">One card per original reference. Each colored box is one augmented mix; the box is the degraded window placed in that reuse. Distinct, non-touching boxes mean the augmentation produced non-overlapping placements.</p>
<div class="controls">
  <input id="q" type="search" placeholder="Filter by reference filename...">
  <div class="stat">References: <b id="nref">0</b></div>
  <div class="stat">Mix files: <b id="nmix">0</b></div>
  <div class="stat">Multiplier: <b id="mult">0</b>x</div>
  <div class="stat">Overlap violations: <b id="nviol">0</b></div>
  <label class="stat"><input type="checkbox" id="onlyMulti"> only reused (&gt;1 window)</label>
</div>
<div id="list"></div>
<script>
const COLORS = __COLORS__;
const DATA = __DATA__;
function overlaps(a,b){return a[0] < b[1] && b[0] < a[1];}
function card(g,ci){
  const dur = g.duration || Math.max(...g.windows.map(w=>w.end), 1);
  let viol = 0;
  for(let i=0;i<g.windows.length;i++)
    for(let j=i+1;j<g.windows.length;j++)
      if(overlaps([g.windows[i].start,g.windows[i].end],[g.windows[j].start,g.windows[j].end])) viol++;
  const boxes = g.windows.map((w,i)=>{
    const left=(w.start/dur*100).toFixed(2), width=Math.max(0.6,(w.end-w.start)/dur*100).toFixed(2);
    const c=COLORS[i%COLORS.length];
    return `<div class="win" style="left:${left}%;width:${width}%;background:${c}" title="${w.start.toFixed(2)}-${w.end.toFixed(2)}s">${i+1}</div>`;
  }).join("");
  const rows = g.windows.map((w,i)=>{
    const c=COLORS[i%COLORS.length];
    const canPlay = w.mixAudio ? "" : ' style="cursor:default;opacity:.6"';
    const btn = w.mixAudio ? `<button class="playbtn" data-card="${ci}" data-win="${i}">play degraded</button>` : `<span class="playbtn" style="opacity:.5">no audio</span>`;
    return `<div class="row" data-card="${ci}" data-win="${i}"${canPlay}><span class="dot" style="background:${c}"></span>window ${i+1}: [${w.start.toFixed(2)}s, ${w.end.toFixed(2)}s]  (len ${(w.end-w.start).toFixed(2)}s, MOS ${w.mos==null?"-":w.mos})${btn}</div>`;
  }).join("");
  // Waveform layer: original clean REF envelope + clickable overlay windows.
  let waveBlock = "";
  if(g.peaks && g.peaks.length){
    const N=g.peaks.length, H=90, MID=H/2;
    let up="", dn="";
    for(let i=0;i<N;i++){
      const x=(i/(N-1)*100).toFixed(3);
      const a=g.peaks[i]*(MID-2);
      up+=`${x},${(MID-a).toFixed(2)} `;
      dn=`${x},${(MID+a).toFixed(2)} `+dn;
    }
    const poly=up+dn;
    const overlays=g.windows.map((w,i)=>{
      const left=(w.start/dur*100).toFixed(2), width=Math.max(0.6,(w.end-w.start)/dur*100).toFixed(2);
      const c=COLORS[i%COLORS.length];
      const t=w.mixAudio?`title="click to play degraded window ${i+1}"`:'title="no audio"';
      return `<div class="win" data-card="${ci}" data-win="${i}" ${t} style="left:${left}%;width:${width}%;background:${c}"></div>`+
             `<div class="winlabel" style="left:calc(${left}% + 3px)">${i+1}</div>`;
    }).join("");
    const refPlayer=g.refAudio?`<div class="hint">Clean reference (full clip): <audio controls preload="none" src="${g.refAudio}" style="height:28px;vertical-align:middle"></audio></div>`:"";
    waveBlock=`<p class="hint">Click any colored box (or "play degraded") to hear that degraded mix. The box marks where the degradation was spliced in.</p>
    <div class="wave">
      <svg viewBox="0 0 100 ${H}" preserveAspectRatio="none"><polygon points="${poly}" fill="#5ea0ff" opacity="0.9"/></svg>
      ${overlays}
    </div>
    <div class="axis"><span>clean reference waveform, colored boxes = degraded windows (click to hear the degraded version)</span></div>
    ${refPlayer}
    <div class="nowplaying" id="np-${ci}"></div>
    <audio id="au-${ci}" preload="none"></audio>`;
  }
  return `<div class="card" data-ref="${g.ref.toLowerCase()}" data-n="${g.windows.length}">
    <h2>${g.ref}</h2>
    <div class="meta"><span class="pill">${g.windows.length} window${g.windows.length>1?"s":""}</span>
      <span class="pill">duration ${dur.toFixed(2)}s</span>
      ${viol>0?`<span class="warn">OVERLAP x${viol}</span>`:""}</div>
    ${waveBlock}
    <div class="timeline">${boxes}</div>
    <div class="axis"><span>0s</span><span>${dur.toFixed(1)}s</span></div>
    <div class="rows">${rows}</div>
  </div>`;
}
function render(){
  const q=document.getElementById("q").value.trim().toLowerCase();
  const onlyMulti=document.getElementById("onlyMulti").checked;
  const list=document.getElementById("list");
  let html="", shown=0;
  DATA.forEach((g,ci)=>{
    if(q && !g.ref.toLowerCase().includes(q)) return;
    if(onlyMulti && g.windows.length<2) return;
    html+=card(g,ci); shown++;
  });
  list.innerHTML=html;
}
function playWindow(ci,wi){
  const g=DATA[ci]; if(!g) return;
  const w=g.windows[wi]; if(!w||!w.mixAudio) return;
  const au=document.getElementById("au-"+ci);
  const np=document.getElementById("np-"+ci);
  if(au){
    au.src=w.mixAudio; au.play().catch(()=>{});
    if(np) np.textContent="Playing degraded window "+(wi+1)+": "+w.mixName;
  }
  // visual active state within this card
  const cardEl=au?au.closest(".card"):null;
  if(cardEl){
    cardEl.querySelectorAll(".win.playing,.row.playing").forEach(e=>e.classList.remove("playing"));
    cardEl.querySelectorAll(`[data-card="${ci}"][data-win="${wi}"]`).forEach(e=>{
      if(e.classList.contains("win")||e.classList.contains("row")) e.classList.add("playing");
    });
  }
}
(function(){
  const nmix=DATA.reduce((s,g)=>s+g.windows.length,0);
  let viol=0;
  for(const g of DATA)
    for(let i=0;i<g.windows.length;i++)
      for(let j=i+1;j<g.windows.length;j++)
        if(overlaps([g.windows[i].start,g.windows[i].end],[g.windows[j].start,g.windows[j].end])) viol++;
  document.getElementById("nref").textContent=DATA.length;
  document.getElementById("nmix").textContent=nmix;
  document.getElementById("mult").textContent=(nmix/Math.max(1,DATA.length)).toFixed(2);
  const v=document.getElementById("nviol"); v.textContent=viol; if(viol>0) v.classList.add("warn");
  document.getElementById("q").addEventListener("input",render);
  document.getElementById("onlyMulti").addEventListener("change",render);
  // Event delegation: any element carrying data-card + data-win plays that mix.
  document.getElementById("list").addEventListener("click",ev=>{
    const t=ev.target.closest("[data-card][data-win]");
    if(!t) return;
    playWindow(parseInt(t.dataset.card,10), parseInt(t.dataset.win,10));
  });
  render();
})();
</script>
</body>
</html>
"""


def main(
    manifest_path: Path = typer.Option(..., help="Path to the augmented manifest CSV."),
    site_dir: Path | None = typer.Option(
        None, help="Output dir. Defaults to manifest parent / grouped_inspector."
    ),
    refs_dir: Path | None = typer.Option(
        None, help="Dir with CLEAN reference WAVs (NISQA_TRAIN_SIM/ref), for the waveform."
    ),
    mixes_dir: Path | None = typer.Option(
        None, help="Dir with degraded mix WAVs, for click-to-play on each window."
    ),
    top_n: int = typer.Option(
        30, help="Render only the N highest-reuse REFs (one card each)."
    ),
    waveform_buckets: int = typer.Option(
        480, help="Envelope resolution (points) for each rendered waveform."
    ),
) -> None:
    """Build the grouped-by-reference augmentation inspector for the top-N REFs.

    Each card shows the clean reference waveform with every augmented window as a
    color-coded overlay; clicking a window plays that degraded mix.
    """
    if not manifest_path.exists():
        raise ValueError(f"Manifest not found: {manifest_path}")

    out_dir = site_dir or (manifest_path.parent / "grouped_inspector")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    # REFs sorted by reuse count, descending; keep only the top-N most-augmented.
    order = df["filename_ref"].value_counts().index.tolist()
    if top_n > 0:
        order = order[:top_n]

    audio_dir = out_dir / "audio"
    shutil.rmtree(audio_dir, ignore_errors=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    groups: list[dict] = []
    n_mix_audio = 0
    for ref in order:
        g = df[df["filename_ref"] == ref].sort_values("index")
        windows = []
        for _, row in g.iterrows():
            seg = json.loads(row["mix_deg_segments"])[0]
            mix_name = str(row["mix_filename"])
            mix_rel = None
            if mixes_dir is not None:
                src = Path(mixes_dir) / mix_name
                if src.exists():
                    dst = audio_dir / mix_name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    mix_rel = f"audio/{mix_name}"
                    n_mix_audio += 1
            windows.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "mos": (None if pd.isna(row.get("mos")) else round(float(row["mos"]), 2)),
                    "mixAudio": mix_rel,
                    "mixName": mix_name,
                }
            )

        peaks = None
        ref_audio_rel = None
        if refs_dir is not None:
            src = Path(refs_dir) / str(ref)
            if src.exists():
                peaks = waveform_peaks(src, buckets=waveform_buckets)
                dst = audio_dir / str(ref)
                if not dst.exists():
                    shutil.copy2(src, dst)
                ref_audio_rel = f"audio/{ref}"

        groups.append(
            {
                "ref": str(ref),
                "duration": (
                    None
                    if pd.isna(g.iloc[0].get("duration_seconds"))
                    else float(g.iloc[0]["duration_seconds"])
                ),
                "windows": windows,
                "peaks": peaks,
                "refAudio": ref_audio_rel,
            }
        )

    html = HTML_TEMPLATE.replace("__COLORS__", json.dumps(WINDOW_COLORS)).replace(
        "__DATA__", json.dumps(groups)
    )
    (out_dir / "index.html").write_text(html)

    n_mix = sum(len(g["windows"]) for g in groups)
    print(f"Grouped inspector written to: {out_dir / 'index.html'}")
    print(f"References: {len(groups)}  Mix files: {n_mix}  Multiplier: {n_mix/max(1,len(groups)):.2f}x")
    print(f"Clickable degraded-mix players: {n_mix_audio}")
    print("Open the index.html directly in a browser (no server needed).")


if __name__ == "__main__":
    typer.run(main)
