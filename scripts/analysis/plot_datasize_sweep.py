"""Three thesis figures for the temporal SFT data-size sweep (relaxed palette)."""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAGE = "#6b9080"; CLAY = "#d4a373"; INK = "#3d3d46"; GRID = "#c2c2c2"
LINE5 = ["#6b9080", "#a4ac86", "#d4a373", "#8d9db6", "#b5838d"]
plt.rcParams.update({"font.size":11,"axes.edgecolor":INK,"axes.labelcolor":INK,
    "xtick.color":INK,"ytick.color":INK,"text.color":INK,"axes.titlesize":12.5,
    "figure.dpi":150,"font.family":"DejaVu Sans"})

base = Path("/work3/s234817/automatic-speech-assessment/results/analysis/datasize_sweep")
out = base/"figures"; out.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(base/"metrics.csv")))
sizes=[int(r["size"]) for r in rows]; iou=[float(r["iou_mean"]) for r in rows]; mse=[float(r["mos_mse"]) for r in rows]
labels=[f"{s:,}" for s in sizes]; AUDIO_BLIND=0.22

# Fig 1: IoU curve
fig,ax=plt.subplots(figsize=(6.2,4))
ax.plot(sizes,iou,"o-",color=SAGE,lw=2.2,ms=8,mfc="white",mec=SAGE,mew=2)
ax.axhline(AUDIO_BLIND,ls=(0,(4,3)),color=GRID,lw=1.3)
ax.text(0.015,0.30,"audio-blind floor (0.22)",transform=ax.transAxes,color="#8a8a8a",fontsize=8.5)
ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels(labels)
ax.set_xlabel("Training clips"); ax.set_ylabel(r"Mean temporal IoU $\uparrow$")
ax.set_ylim(0,1); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",color=GRID,alpha=0.4,lw=0.7)
fig.tight_layout(); fig.savefig(out/"datasize_iou_curve.png"); plt.close(fig)

# Fig 2: loss curves
losses=json.load(open(base/"loss_series.json"))
fig,ax=plt.subplots(figsize=(6.2,4))
for i,k in enumerate(sorted(losses,key=lambda k:int(k))):
    s=losses[k]
    if s: ax.plot([p[0] for p in s],[p[1] for p in s],color=LINE5[i%len(LINE5)],lw=2,label=f"{int(k):,} clips")
ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
ax.legend(fontsize=8.5,frameon=False); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",color=GRID,alpha=0.4,lw=0.7)
fig.tight_layout(); fig.savefig(out/"datasize_loss_curves.png"); plt.close(fig)

# Fig 4.4: grouped bars, dual axis, arrows on axes, audio-blind label top-left
fig,ax1=plt.subplots(figsize=(7.8,4.7)); ax2=ax1.twinx()
x=list(range(len(sizes))); w=0.38; xl=[i-w/2 for i in x]; xr=[i+w/2 for i in x]
b1=ax1.bar(xl,iou,w,color=SAGE,label="Mean IoU (localization)",zorder=3)
b2=ax2.bar(xr,mse,w,color=CLAY,label="MOS MSE (rating)",zorder=3)
ax1.axhline(AUDIO_BLIND,ls=(0,(4,3)),color="#9a9a9a",lw=1.2,zorder=2)
ax1.text(0.015,0.93,"dashed: audio-blind IoU floor (0.22)",transform=ax1.transAxes,color="#8a8a8a",fontsize=8.2)
for xx,v in zip(xl,iou): ax1.text(xx,v+0.012,f"{v:.2f}",ha="center",va="bottom",fontsize=8.5,color="#4a6b5d")
for xx,v in zip(xr,mse): ax2.text(xx,v+0.006,f"{v:.2f}",ha="center",va="bottom",fontsize=8.5,color="#a9774a")
ax1.set_ylabel(r"Mean temporal IoU  $\uparrow$",color=SAGE)
ax2.set_ylabel(r"MOS MSE  $\downarrow$",color=CLAY)
ax1.tick_params(axis="y",colors=SAGE); ax2.tick_params(axis="y",colors=CLAY)
ax1.set_ylim(0,1.0); ax2.set_ylim(0,max(mse)*1.6)
ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_xlabel("Training clips")
ax1.set_title("Localization vs. rating across training-set size")
ax1.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
ax1.grid(axis="y",color=GRID,alpha=0.35,lw=0.7,zorder=0)
# big directional arrows just outside each axis
ax1.annotate("",xy=(-0.105,0.80),xytext=(-0.105,0.45),xycoords="axes fraction",arrowprops=dict(arrowstyle="-|>",color=SAGE,lw=2.4))
ax2.annotate("",xy=(1.105,0.45),xytext=(1.105,0.80),xycoords="axes fraction",arrowprops=dict(arrowstyle="-|>",color=CLAY,lw=2.4))
ax1.legend([b1,b2],[h.get_label() for h in [b1,b2]],frameon=False,fontsize=9,loc="upper center",bbox_to_anchor=(0.5,-0.13),ncol=2)
fig.tight_layout(); fig.savefig(out/"datasize_iou_mse_bars.png",bbox_inches="tight"); plt.close(fig)
print("done")
