"""Extract data-size sweep metrics + training-loss series for thesis figures."""
import json, re, glob
from pathlib import Path

RUNS = [
    (500,   "sft_gc_timelast_sweep500",   "logs/sft_gc_sweep500_*.out"),
    (1000,  "sft_gc_timelast_sweep1000",  "logs/sft_gc_sweep1000_*.out"),
    (2500,  "sft_gc_timelast_sweep2500",  "logs/sft_gc_sweep2500_*.out"),
    (5105,  "sft_gc_timelast_sweep5105",  "logs/sft_gc_sweep5105_*.out"),
    (13495, "sft_gc_timelast_full13495",  "logs/sft_gc_full13495_*.out"),
]
SETS = ["FOR", "LIVE", "P501"]
base = Path("/work3/s234817/automatic-speech-assessment")

def metrics(model):
    d = base / f"results/evaluation/temporal/{model}_greedy_timelast_600tok"
    ious, mses, maes, uniq, tot = [], [], [], 0, 0
    for s in SETS:
        fs = list(d.glob(f"test_{s}_*results.json"))
        if not fs: return None
        recs = json.load(open(fs[0]))
        recs = recs.get("results", recs) if isinstance(recs, dict) else recs
        t = [r["tiou"] for r in recs if r.get("tiou") is not None]
        pm = [(r["pred_mos"], r["mos"]) for r in recs if r.get("pred_mos") is not None]
        ious.append(sum(t)/len(t))
        mses.append(sum((a-b)**2 for a,b in pm)/len(pm))
        maes.append(sum(abs(a-b) for a,b in pm)/len(pm))
        uniq += len(set(r.get("predicted_response","") for r in recs)); tot += len(recs)
    return dict(iou=sum(ious)/3, mse=sum(mses)/3, mae=sum(maes)/3,
                iou_per=ious, n_unique=uniq, n_total=tot)

rows, losses = [], {}
for size, model, logpat in RUNS:
    m = metrics(model)
    rows.append((size, m))
    fs = glob.glob(str(base/logpat))
    series = []
    if fs:
        txt = open(fs[0]).read().replace("\r","\n")
        for mm in re.finditer(r"loss.:\s*([0-9.]+),.*?epoch.:\s*([0-9.]+)", txt):
            series.append((float(mm.group(2)), float(mm.group(1))))
    losses[size] = series

out = base/"results/analysis/datasize_sweep"
out.mkdir(parents=True, exist_ok=True)
with open(out/"metrics.csv","w") as f:
    f.write("size,iou_mean,iou_FOR,iou_LIVE,iou_P501,mos_mse,mos_mae,n_unique,n_total\n")
    for size,m in rows:
        if m: f.write("%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d\n"%(size,m["iou"],m["iou_per"][0],m["iou_per"][1],m["iou_per"][2],m["mse"],m["mae"],m["n_unique"],m["n_total"]))
json.dump(losses, open(out/"loss_series.json","w"))
print(open(out/"metrics.csv").read())
print("loss series points:", {k:len(v) for k,v in losses.items()})
