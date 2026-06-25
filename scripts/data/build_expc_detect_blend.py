#!/usr/bin/env python
"""Exp C: blend degraded (with interval) + preprocessed clean (no interval) records.

Clean records come from preprocess_clean_refs.py (16 kHz / 6 s, matched to the
mixes). Degraded stream = headline MOS<=3 timelast-anchoroffset set, verbatim.
"""
import json, random, argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degraded-jsonl", default="data/processed/temporal/train_nisqa_temporal_gc_timelast_aug_anchoroffset.json")
    ap.add_argument("--clean-records", default="data/processed/temporal/expc_clean_records.jsonl")
    ap.add_argument("--output-jsonl", default="data/processed/temporal/train_nisqa_temporal_expc_detect_blend.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    deg = [json.loads(l) for l in open(args.degraded_jsonl)]
    for r in deg: r["is_clean"] = False
    clean = [json.loads(l) for l in open(args.clean_records)]
    blend = deg + clean
    rng.shuffle(blend)
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w") as f:
        for r in blend: f.write(json.dumps(r) + "\n")
    print(f"degraded={len(deg)} clean={len(clean)} total={len(blend)} clean_frac={len(clean)/len(blend):.3f}")
    print("output:", args.output_jsonl)

if __name__ == "__main__": main()
