"""Subsample the temporal SFT train set to N clips, 1 clip per reference file.

Data-size ablation (Carl 2026-06-16): the full set is ~13.5k clips from ~5.1k
refs (~2.6/ref). To study how training-set SIZE affects temporal performance, we
build a 1-clip-per-ref pool (prefer a single-splice mix) and emit deterministic
subsets of several sizes for a size-vs-IoU curve.
"""
import json
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    src: Path = typer.Option(
        Path("data/processed/temporal/train_nisqa_temporal_gc_timelast_aug_anchoroffset.json")
    ),
    sizes: str = typer.Option("500,1000,2500,5105", help="Comma-sep subset sizes."),
    out_prefix: str = typer.Option(
        "data/processed/temporal/train_gc_timelast_sweep_"
    ),
):
    recs = [json.loads(l) for l in src.open() if l.strip()]
    # one record per ref: prefer fewest degradation segments (single splice),
    # tie-break deterministically by id.
    best = {}
    for r in recs:
        ref = r.get("filename_ref")
        nseg = len(r.get("mix_deg_segments") or [])
        key = (nseg, str(r.get("id")))
        if ref not in best or key < best[ref][0]:
            best[ref] = (key, r)
    pool = [v[1] for v in best.values()]
    pool.sort(key=lambda r: str(r.get("id")))  # deterministic order
    print(f"src={len(recs)} refs/pool={len(pool)}")
    for s in [int(x) for x in sizes.split(",")]:
        n = min(s, len(pool))
        out = Path(f"{out_prefix}{n}.json")
        with out.open("w") as f:
            for r in pool[:n]:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {n} -> {out}")


if __name__ == "__main__":
    app()
