"""Three thesis figures for the temporal SFT data-size sweep (relaxed palette)."""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Relaxed, muted palette (no pure red/blue)
SAGE   = "#6b9080"   # IoU
CLAY   = "#cb997e"   # MSE
INK    = "#3d3d46"   # text/axes
GRID   = "#b7b7b7"
LINE5  = ["#6b9080", "#a4ac86", "#cb997e", "#8d9db6", "#b5838d"]

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "figure.dpi": 150,
})

base = Path("/work3/s234817/automatic-speech-assessment/results/analysis/datasize_sweep")
out = base / "figures"; out.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(base/"metrics.csv")))
sizes = [int(r["size"]) for r in rows]
iou   = [float(r["iou_mean"]) for r in rows]
mse   = [float(r["mos_mse"]) for r in rows]
AUDIO_BLIND = 0.22
labels = [f"{s:,}" for s in sizes]

def label_bars(ax, bars, vals, fmt):
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, fmt.format(v), ha="center",
                va="bottom", fontsize=9, color=INK)

# Fig 1: size vs IoU
fig, ax = plt.subplots(figsize=(6.2,4))
ax.plot(sizes, iou, "o-", color=SAGE, lw=2.2, ms=8, mfc="white", mec=SAGE, mew=2)
ax.axhline(AUDIO_BLIND, ls=(0,(4,3)), color=GRID, lw=1.3)
ax.text(sizes[0], AUDIO_BLIND+0.02, "audio-blind floor (0.22)", color="#7a7a7a", fontsize=8.5)
ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels(labels)
ax.set_xlabel("Training clips (1 per reference)"); ax.set_ylabel("Mean temporal IoU")
ax.set_ylim(0,1); ax.grid(axis="y", color=GRID, alpha=0.35, lw=0.7)
fig.tight_layout(); fig.savefig(out/"datasize_iou_curve.png"); plt.close(fig)

# Fig 2: training-loss curves
losses = json.load(open(base/"loss_series.json"))
fig, ax = plt.subplots(figsize=(6.2,4))
for i,k in enumerate(sorted(losses, key=lambda k:int(k))):
    s = losses[k]
    if not s: continue
    ax.plot([p[0] for p in s], [p[1] for p in s], color=LINE5[i%len(LINE5)],
            lw=2, label=f"{int(k):,} clips")
ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
ax.legend(fontsize=8.5, frameon=False); ax.grid(axis="y", color=GRID, alpha=0.35, lw=0.7)
fig.tight_layout(); fig.savefig(out/"datasize_loss_curves.png"); plt.close(fig)

# Fig 3 (Fig 4.4): IoU + MOS-MSE bars, side by side, relaxed colors
fig, ax = plt.subplots(1,2, figsize=(9.5,4.2))
x = range(len(sizes))
b0 = ax[0].bar(x, iou, color=SAGE, width=0.68)
ax[0].axhline(AUDIO_BLIND, ls=(0,(4,3)), color=GRID, lw=1.3)
ax[0].text(len(sizes)-0.5, AUDIO_BLIND+0.015, "audio-blind 0.22", color="#7a7a7a", fontsize=8.5, ha="right")
ax[0].set_ylabel("Mean temporal IoU"); ax[0].set_title("Localization (IoU)"); ax[0].set_ylim(0,1.0)
label_bars(ax[0], b0, iou, "{:.3f}")
b1 = ax[1].bar(x, mse, color=CLAY, width=0.68)
ax[1].set_ylabel("MOS MSE"); ax[1].set_title("Rating (MOS MSE)"); ax[1].set_ylim(0, max(mse)*1.18)
label_bars(ax[1], b1, mse, "{:.3f}")
for a in ax:
    a.set_xticks(list(x)); a.set_xticklabels(labels, rotation=0); a.set_xlabel("Training clips")
    a.grid(axis="y", color=GRID, alpha=0.35, lw=0.7)
fig.tight_layout(); fig.savefig(out/"datasize_iou_mse_bars.png"); plt.close(fig)
print("figures regenerated with relaxed palette:", sorted(p.name for p in out.glob("*.png")))
