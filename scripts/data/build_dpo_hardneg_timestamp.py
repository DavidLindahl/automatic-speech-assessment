"""Filter the timestamp-cycle DPO set to hard negatives by sample-vs-gold IoU.

The full timestamp cycle (train_dpo_armA_cycle_timestamp_sampled.json) uses the
model own sampled interval as the rejected, single-factor (caption gold, only the
<aN><fK> interval differs). Its sampled intervals sit at mean IoU ~0.74 vs gold,
so most pairs carry near-zero preference signal and the cycle never beats SFT
(t-IoU 0.743 vs 0.884). This keeps only the pairs where the sample is genuinely
wrong (IoU(chosen, rejected) < threshold), concentrating the DPO gradient on real
on-task localization errors while staying on the model own output manifold.
"""
import json
import re
from pathlib import Path

import typer

app = typer.Typer(help="Build hard-negative timestamp-cycle DPO pairs.")
P = re.compile(r"<a(\d+)><f(\d+)>")


def interval(text: str):
    m = P.findall(text)
    if len(m) < 2:
        return None
    a = int(m[-2][0]) + int(m[-2][1]) / 10.0
    b = int(m[-1][0]) + int(m[-1][1]) / 10.0
    return (a, b) if b > a else None


def iou(g, r):
    if not g or not r:
        return None
    inter = max(0.0, min(g[1], r[1]) - max(g[0], r[0]))
    union = (g[1] - g[0]) + (r[1] - r[0]) - inter
    return inter / union if union > 0 else 0.0


@app.command()
def main(
    src: Path = typer.Option(
        Path("data/processed/dpo/train_dpo_armA_cycle_timestamp_sampled.json"),
        help="Source timestamp-cycle JSONL.",
    ),
    out: Path = typer.Option(..., help="Output filtered JSONL."),
    max_iou: float = typer.Option(0.6, help="Keep pairs with sample IoU below this."),
):
    recs = [json.loads(l) for l in src.open() if l.strip()]
    kept, dropped, unparsed = [], 0, 0
    for r in recs:
        g, s = interval(r["chosen"]), interval(r["rejected"])
        v = iou(g, s)
        if v is None:
            unparsed += 1
            continue
        if v < max_iou:
            kept.append(r)
        else:
            dropped += 1
    with out.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(
        f"src={len(recs)} kept={len(kept)} dropped={dropped} "
        f"unparsed={unparsed} max_iou={max_iou} -> {out}"
    )


if __name__ == "__main__":
    app()
