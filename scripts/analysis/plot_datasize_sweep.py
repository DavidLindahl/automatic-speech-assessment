"""Three thesis figures for the temporal SFT data-size sweep."""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = Path("/work3/s234817/automatic-speech-assessment/results/analysis/datasize_sweep")
out = base / "figures"; out.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(base/"metrics.csv")))
sizes = [int(r["size"]) for r in rows]
iou   = [float(r["iou_mean"]) for r in rows]
mse   = [float(r["mos_mse"]) for r in rows]
uniq  = [int(r["n_unique"]) for r in rows]
AUDIO_BLIND = 0.22  # best-constant-interval audio-blind baseline

# Fig 1: size vs IoU
plt.figure(figsize=(6,4))
plt.plot(sizes, iou, "o-", color="#1f77b4", lw=2, ms=7)
plt.axhline(AUDIO_BLIND, ls="--", color="gray", lw=1)
plt.text(sizes[0], AUDIO_BLIND+0.015, "audio-blind floor (0.22)", color="gray", fontsize=8)
plt.xscale("log"); plt.xticks(sizes, [str(s) for s in sizes])
plt.xlabel("Training clips (1 per reference)"); plt.ylabel("Mean temporal IoU")
plt.title("Temporal IoU vs training-set size")
plt.ylim(0,1); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(out/"datasize_iou_curve.png", dpi=150); plt.close()

# Fig 2: overlaid training-loss curves
losses = json.load(open(base/"loss_series.json"))
plt.figure(figsize=(6,4))
cmap = plt.cm.viridis
keys = sorted(losses, key=lambda k:int(k))
for i,k in enumerate(keys):
    series = losses[k]
    if not series: continue
    ep = [p[0] for p in series]; ls = [p[1] for p in series]
    plt.plot(ep, ls, color=cmap(i/max(1,len(keys)-1)), lw=1.8, label=f"{k} clips")
plt.xlabel("Epoch"); plt.ylabel("Training loss")
plt.title("Training loss by data-set size")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(out/"datasize_loss_curves.png", dpi=150); plt.close()

# Fig 3: IoU + MOS-MSE barplots
fig, ax = plt.subplots(1,2, figsize=(9,4))
x = range(len(sizes)); labels=[str(s) for s in sizes]
ax[0].bar(x, iou, color="#1f77b4"); ax[0].axhline(AUDIO_BLIND, ls="--", color="gray", lw=1)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel("Mean temporal IoU"); ax[0].set_title("IoU by data-set size"); ax[0].set_ylim(0,1); ax[0].grid(alpha=0.3, axis="y")
ax[1].bar(x, mse, color="#d62728")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_ylabel("MOS MSE"); ax[1].set_title("MOS MSE by data-set size"); ax[1].grid(alpha=0.3, axis="y")
for a in ax: a.set_xlabel("Training clips")
plt.tight_layout(); plt.savefig(out/"datasize_iou_mse_bars.png", dpi=150); plt.close()
print("figures written:", [p.name for p in out.glob("*.png")])
