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
- **Playhead + seek**: a white vertical playhead tracks the audio position on the
  waveform; clicking anywhere on the waveform track seeks to that point in the
  clip (scrub back and forth).

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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0c1016; --bg2:#11161f; --panel:#141b26; --panel2:#19212e;
    --ink:#e7edf5; --muted:#8a99ad; --faint:#5b6a7d;
    --line:#26303f; --line2:#1c2531;
    --accent:#56e0c8; --accent2:#7c9cff;
    --wave:#56e0c8; --wavefill:rgba(86,224,200,.16); --warn:#ff6b6b;
    --shadow:0 1px 0 rgba(255,255,255,.03),0 12px 32px -12px rgba(0,0,0,.6);
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{margin:0;padding:34px 30px 60px;color:var(--ink);
    font-family:"Sora",-apple-system,BlinkMacSystemFont,sans-serif;
    background:
      radial-gradient(1200px 600px at 12% -8%,rgba(86,224,200,.10),transparent 60%),
      radial-gradient(1000px 520px at 100% 0%,rgba(124,156,255,.09),transparent 55%),
      var(--bg);
    background-attachment:fixed}
  .head{max-width:1060px;margin:0 auto 22px}
  .eyebrow{font-family:"Space Mono",monospace;font-size:11px;letter-spacing:.32em;
    text-transform:uppercase;color:var(--accent);margin:0 0 8px}
  h1{font-size:30px;line-height:1.05;margin:0 0 8px;font-weight:700;letter-spacing:-.01em}
  h1 em{font-style:normal;color:var(--muted);font-weight:500}
  .sub{color:var(--muted);font-size:13.5px;line-height:1.5;margin:0 0 18px;max-width:680px}
  .controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
  input[type=search]{padding:10px 14px;border:1px solid var(--line);border-radius:10px;
    font-size:13.5px;min-width:300px;background:var(--panel);color:var(--ink);
    font-family:inherit;outline:none;transition:border-color .15s,box-shadow .15s}
  input[type=search]::placeholder{color:var(--faint)}
  input[type=search]:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(86,224,200,.14)}
  .stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;
    padding:8px 13px;font-size:12px;color:var(--muted);display:flex;gap:7px;align-items:baseline}
  .stat b{font-size:15px;color:var(--ink);font-family:"Space Mono",monospace}
  .stat.ok b{color:var(--accent)}
  .stat.bad b{color:var(--warn)}
  .stat label,label.stat{cursor:pointer}
  label.stat input{accent-color:var(--accent)}
  .list{max-width:1060px;margin:18px auto 0;display:flex;flex-direction:column;gap:14px}
  .card{background:linear-gradient(180deg,var(--panel),var(--panel2));
    border:1px solid var(--line);border-radius:16px;padding:18px 20px 16px;
    box-shadow:var(--shadow);position:relative;overflow:hidden}
  .card::before{content:"";position:absolute;inset:0 auto 0 0;width:3px;
    background:linear-gradient(var(--accent),var(--accent2));opacity:.0;transition:opacity .2s}
  .card.active::before{opacity:.9}
  .card h2{font-size:12.5px;margin:0 0 3px;font-family:"Space Mono",monospace;
    color:var(--ink);word-break:break-all;letter-spacing:-.01em}
  .meta{color:var(--muted);font-size:11.5px;margin-bottom:12px;display:flex;gap:7px;align-items:center;flex-wrap:wrap}
  .meta .pill{display:inline-flex;align-items:center;gap:5px;background:var(--bg2);
    border:1px solid var(--line);border-radius:999px;padding:3px 10px;
    font-family:"Space Mono",monospace;font-size:10.5px;color:var(--muted)}
  .meta .pill b{color:var(--ink);font-weight:700}
  .hint{color:var(--faint);font-size:11px;margin:0 0 8px;display:flex;align-items:center;gap:6px}
  .hint kbd{font-family:"Space Mono",monospace;background:var(--bg2);border:1px solid var(--line);
    border-radius:5px;padding:1px 6px;font-size:10px;color:var(--muted)}

  .wave{position:relative;height:104px;border-radius:12px;overflow:hidden;margin-bottom:0;
    cursor:crosshair;background:
      linear-gradient(180deg,#0a1018,#0d141d);
    border:1px solid var(--line2);box-shadow:inset 0 0 40px rgba(0,0,0,.45)}
  .wave svg{position:absolute;inset:0;width:100%;height:100%;display:block}
  .wave .grid{position:absolute;inset:0;pointer-events:none;opacity:.5;
    background:repeating-linear-gradient(90deg,transparent 0 9.9%,rgba(255,255,255,.035) 10% 10.05%)}
  .win{position:absolute;top:0;height:100%;cursor:pointer;
    border-left:1px solid rgba(255,255,255,.18);border-right:1px solid rgba(255,255,255,.18);
    transition:filter .15s,opacity .15s;mix-blend-mode:screen;opacity:.34}
  .win:hover{opacity:.5;filter:brightness(1.15)}
  .win.playing{opacity:.6;filter:brightness(1.3)}
  .winlabel{position:absolute;top:6px;font-family:"Space Mono",monospace;font-size:10px;
    font-weight:700;color:#fff;text-shadow:0 1px 3px rgba(0,0,0,.8);pointer-events:none;
    padding:1px 5px;border-radius:5px;background:rgba(8,12,18,.5)}
  .playhead{position:absolute;top:0;bottom:0;width:2px;background:#fff;left:0;
    box-shadow:0 0 10px rgba(255,255,255,.9);pointer-events:none;z-index:5;opacity:0}
  .playhead.on{opacity:1}
  .playhead::after{content:"";position:absolute;top:-2px;left:-4px;width:10px;height:10px;
    border-radius:50%;background:#fff;box-shadow:0 0 8px rgba(255,255,255,.9)}
  .axis{display:flex;justify-content:space-between;color:var(--faint);
    font-family:"Space Mono",monospace;font-size:10px;margin:7px 1px 12px}

  .rows{display:flex;flex-direction:column;gap:4px}
  .row{display:flex;align-items:center;gap:11px;padding:8px 12px;border-radius:10px;
    cursor:pointer;font-size:12.5px;border:1px solid transparent;transition:background .14s,border-color .14s}
  .row:hover{background:var(--bg2);border-color:var(--line2)}
  .row.playing{background:rgba(86,224,200,.08);border-color:rgba(86,224,200,.35)}
  .swatch{width:12px;height:12px;border-radius:4px;flex:0 0 auto;box-shadow:0 0 0 1px rgba(255,255,255,.12)}
  .row .lab{font-family:"Space Mono",monospace;color:var(--muted);font-size:11.5px}
  .row .lab b{color:var(--ink);font-weight:700}
  .row .tt{font-family:"Space Mono",monospace;color:var(--ink);font-size:11.5px}
  .row .mos{color:var(--faint);font-family:"Space Mono",monospace;font-size:11px}
  .playbtn{margin-left:auto;display:inline-flex;align-items:center;gap:6px;font-size:11px;
    font-family:"Sora";font-weight:600;border:1px solid var(--line);background:var(--bg2);
    color:var(--muted);border-radius:8px;padding:5px 11px;cursor:pointer;transition:all .14s}
  .playbtn:hover{border-color:var(--accent);color:var(--ink)}
  .row.playing .playbtn{background:var(--accent);color:#06120f;border-color:var(--accent)}
  .ico{width:11px;height:11px;display:inline-block;vertical-align:-1px}
  .ico-slot{display:inline-flex}
  .warn{color:var(--warn);font-weight:700}
  .empty{max-width:1060px;margin:40px auto;text-align:center;color:var(--faint);font-size:14px}
</style>
</head>
<body>
<div class="head">
  <p class="eyebrow">NISQA-SIM &middot; placement augmentation</p>
  <h1>Temporal Augmentation Inspector <em>/ grouped by reference</em></h1>
  <p class="sub">One card per original reference. Each colored band on the waveform is one augmented mix, marking where the degradation was spliced into the clean speech. Distinct, non-touching bands confirm the reuses are non-overlapping placements.</p>
  <div class="controls">
    <input id="q" type="search" placeholder="Filter by reference filename...">
    <div class="stat">References <b id="nref">0</b></div>
    <div class="stat">Mix files <b id="nmix">0</b></div>
    <div class="stat ok">Multiplier <b id="mult">0</b></div>
    <div class="stat" id="violstat">Overlaps <b id="nviol">0</b></div>
    <label class="stat"><input type="checkbox" id="onlyMulti"> reused only</label>
  </div>
</div>
<div id="list" class="list"></div>
<script>
const COLORS = __COLORS__;
const DATA = __DATA__;
const ICO_PLAY=`<svg class="ico" viewBox="0 0 12 12" fill="currentColor"><path d="M2 1.5v9l8-4.5z"/></svg>`;
const ICO_PAUSE=`<svg class="ico" viewBox="0 0 12 12" fill="currentColor"><rect x="2" y="1.5" width="3" height="9" rx="1"/><rect x="7" y="1.5" width="3" height="9" rx="1"/></svg>`;
function overlaps(a,b){return a[0] < b[1] && b[0] < a[1];}
// Smooth filled waveform path (Catmull-Rom -> cubic bezier) for a peak array.
function wavePath(peaks,H){
  const N=peaks.length, MID=H/2, A=MID-3;
  const pt=i=>[i/(N-1)*100, MID-Math.max(0,Math.min(1,peaks[i]))*A];
  let d=`M ${pt(0)[0].toFixed(2)} ${pt(0)[1].toFixed(2)}`;
  for(let i=0;i<N-1;i++){
    const p0=pt(Math.max(0,i-1)),p1=pt(i),p2=pt(i+1),p3=pt(Math.min(N-1,i+2));
    const c1x=p1[0]+(p2[0]-p0[0])/6, c1y=p1[1]+(p2[1]-p0[1])/6;
    const c2x=p2[0]-(p3[0]-p1[0])/6, c2y=p2[1]-(p3[1]-p1[1])/6;
    d+=` C ${c1x.toFixed(2)} ${c1y.toFixed(2)}, ${c2x.toFixed(2)} ${c2y.toFixed(2)}, ${p2[0].toFixed(2)} ${p2[1].toFixed(2)}`;
  }
  // mirror down the center for a filled silhouette
  let dn=`L 100 ${MID}`;
  for(let i=N-1;i>=0;i--){const p=pt(i); dn+=` L ${p[0].toFixed(2)} ${(2*MID-p[1]).toFixed(2)}`;}
  return d+` L 100 ${MID} `+dn+" Z";
}
function card(g,ci){
  const dur = g.duration || Math.max(...g.windows.map(w=>w.end), 1);
  let viol = 0;
  for(let i=0;i<g.windows.length;i++)
    for(let j=i+1;j<g.windows.length;j++)
      if(overlaps([g.windows[i].start,g.windows[i].end],[g.windows[j].start,g.windows[j].end])) viol++;
  const rows = g.windows.map((w,i)=>{
    const c=COLORS[i%COLORS.length];
    const dis = w.mixAudio ? "" : ' style="cursor:default;opacity:.55"';
    const btn = w.mixAudio
      ? `<button class="playbtn" data-card="${ci}" data-win="${i}"><span class="ico-slot">${ICO_PLAY}</span><span class="btn-txt">play</span></button>`
      : `<span class="playbtn" style="opacity:.5">no audio</span>`;
    return `<div class="row" data-card="${ci}" data-win="${i}"${dis}>`+
      `<span class="swatch" style="background:${c}"></span>`+
      `<span class="lab">window <b>${i+1}</b></span>`+
      `<span class="tt">${w.start.toFixed(2)}&ndash;${w.end.toFixed(2)}s</span>`+
      `<span class="mos">${(w.end-w.start).toFixed(2)}s &middot; MOS ${w.mos==null?"-":w.mos}</span>`+
      `${btn}</div>`;
  }).join("");
  // Waveform layer: smooth clean-REF silhouette + clickable overlay windows.
  let waveBlock = "";
  if(g.peaks && g.peaks.length){
    const H=104;
    const path=wavePath(g.peaks,H);
    const overlays=g.windows.map((w,i)=>{
      const left=(w.start/dur*100).toFixed(2), width=Math.max(0.6,(w.end-w.start)/dur*100).toFixed(2);
      const c=COLORS[i%COLORS.length];
      const t=w.mixAudio?`title="play degraded window ${i+1}"`:'title="no audio"';
      return `<div class="win" data-card="${ci}" data-win="${i}" ${t} style="left:${left}%;width:${width}%;background:${c}"></div>`+
             `<div class="winlabel" style="left:calc(${left}% + 4px);color:${c}">${i+1}</div>`;
    }).join("");
    waveBlock=`<p class="hint">Click a band to play / pause that degraded mix &middot; click the track to <kbd>seek</kbd></p>
    <div class="wave" data-wave="${ci}">
      <div class="grid"></div>
      <svg viewBox="0 0 100 ${H}" preserveAspectRatio="none">
        <defs><linearGradient id="wg-${ci}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="var(--wave)" stop-opacity="0.85"/>
          <stop offset="1" stop-color="var(--wave)" stop-opacity="0.30"/>
        </linearGradient></defs>
        <path d="${path}" fill="url(#wg-${ci})"/>
      </svg>
      ${overlays}
      <div class="playhead" id="ph-${ci}"></div>
    </div>
    <div class="axis"><span>0.00s</span><span>clean reference waveform &middot; bands = degraded windows</span><span>${dur.toFixed(2)}s</span></div>
    <audio id="au-${ci}" preload="none"></audio>`;
  }
  return `<div class="card" id="card-${ci}" data-ref="${g.ref.toLowerCase()}" data-n="${g.windows.length}">
    <h2>${g.ref}</h2>
    <div class="meta">
      <span class="pill"><b>${g.windows.length}</b> window${g.windows.length>1?"s":""}</span>
      <span class="pill">duration <b>${dur.toFixed(2)}s</b></span>
      ${viol>0?`<span class="pill" style="border-color:var(--warn);color:var(--warn)">OVERLAP &times;${viol}</span>`:`<span class="pill" style="border-color:rgba(86,224,200,.4);color:var(--accent)">&check; non-overlapping</span>`}
    </div>
    ${waveBlock}
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
// --- playback engine: one card active at a time, rAF-smooth playhead ----------
let RAF=0;
function tickPlayhead(){
  let anyPlaying=false;
  document.querySelectorAll(".wave[data-wave]").forEach(wave=>{
    const ci=wave.dataset.wave;
    const au=document.getElementById("au-"+ci), ph=document.getElementById("ph-"+ci);
    if(!au||!ph) return;
    const d=au.duration;
    if(au.src && d && isFinite(d)){
      ph.classList.add("on");
      ph.style.left=(Math.min(1,au.currentTime/d)*100)+"%";
      if(!au.paused && !au.ended) anyPlaying=true;
    }
  });
  RAF = anyPlaying ? requestAnimationFrame(tickPlayhead) : 0;
}
function startRAF(){ if(!RAF) RAF=requestAnimationFrame(tickPlayhead); }
function setRowState(ci,wi,playing){
  const card=document.getElementById("card-"+ci);
  if(!card) return;
  card.classList.toggle("active", playing);
  card.querySelectorAll(".win,.row").forEach(e=>e.classList.remove("playing"));
  card.querySelectorAll(".playbtn .btn-txt").forEach(t=>t.textContent="play");
  card.querySelectorAll(".playbtn .ico-slot").forEach(s=>s.innerHTML=ICO_PLAY);
  if(playing){
    card.querySelectorAll(`[data-card="${ci}"][data-win="${wi}"]`).forEach(e=>{
      if(e.classList.contains("win")||e.classList.contains("row")) e.classList.add("playing");
    });
    const row=card.querySelector(`.row[data-win="${wi}"]`);
    if(row){
      const slot=row.querySelector(".playbtn .ico-slot"); if(slot) slot.innerHTML=ICO_PAUSE;
      const txt=row.querySelector(".playbtn .btn-txt"); if(txt) txt.textContent="pause";
    }
  }
}
function loadWin(ci,wi){
  const au=document.getElementById("au-"+ci), w=DATA[ci].windows[wi];
  if(au.dataset.win!==String(wi)){ au.src=w.mixAudio; au.dataset.win=String(wi); }
  return au;
}
function toggleWindow(ci,wi){
  const g=DATA[ci]; if(!g) return;
  const w=g.windows[wi]; if(!w||!w.mixAudio) return;
  const au=document.getElementById("au-"+ci);
  // Same window already loaded -> toggle pause/resume (playhead freezes on pause).
  if(au.dataset.win===String(wi) && au.src){
    if(au.paused){ au.play().catch(()=>{}); setRowState(ci,wi,true); startRAF(); }
    else{ au.pause(); setRowState(ci,wi,false); }
    return;
  }
  // Stop any other card that is playing (single active player).
  document.querySelectorAll("audio[id^='au-']").forEach(a=>{ if(a!==au){ a.pause(); }});
  loadWin(ci,wi);
  au.play().catch(()=>{});
  setRowState(ci,wi,true);
  startRAF();
}
function seekWave(ci,frac){
  const g=DATA[ci]; if(!g) return;
  const au=document.getElementById("au-"+ci); if(!au) return;
  let wi=au.dataset.win?parseInt(au.dataset.win,10):-1;
  if(wi<0){ wi=g.windows.findIndex(w=>w.mixAudio); if(wi<0) return; loadWin(ci,wi); }
  const apply=()=>{
    const d=au.duration;
    if(d&&isFinite(d)){ au.currentTime=Math.max(0,Math.min(d-0.01,frac*d)); au.play().catch(()=>{}); setRowState(ci,wi,true); startRAF(); }
  };
  if(au.readyState>=1) apply(); else au.addEventListener("loadedmetadata",apply,{once:true});
}
function wireCard(ci){
  const au=document.getElementById("au-"+ci); if(!au||au.dataset.wired) return;
  au.dataset.wired="1";
  au.addEventListener("ended",()=>{ const wi=au.dataset.win?parseInt(au.dataset.win,10):0; setRowState(ci,wi,false);
    const ph=document.getElementById("ph-"+ci); if(ph){ph.style.left="0%";} });
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
  list.innerHTML = shown ? html : `<div class="empty">No references match &ldquo;${q}&rdquo;.</div>`;
  DATA.forEach((g,ci)=>{ if(document.getElementById("au-"+ci)) wireCard(ci); });
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
  document.getElementById("mult").textContent=(nmix/Math.max(1,DATA.length)).toFixed(2)+"x";
  const v=document.getElementById("nviol"); v.textContent=viol;
  const vs=document.getElementById("violstat"); vs.classList.add(viol>0?"bad":"ok");
  document.getElementById("q").addEventListener("input",render);
  document.getElementById("onlyMulti").addEventListener("change",render);
  // Click a band/row -> toggle play/pause that mix; click bare track -> seek.
  document.getElementById("list").addEventListener("click",ev=>{
    const winEl=ev.target.closest("[data-card][data-win]");
    if(winEl){ toggleWindow(parseInt(winEl.dataset.card,10), parseInt(winEl.dataset.win,10)); return; }
    const waveEl=ev.target.closest(".wave[data-wave]");
    if(waveEl){
      const rect=waveEl.getBoundingClientRect();
      const frac=Math.max(0,Math.min(1,(ev.clientX-rect.left)/rect.width));
      seekWave(parseInt(waveEl.dataset.wave,10), frac);
    }
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
