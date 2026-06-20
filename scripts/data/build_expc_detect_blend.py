#!/usr/bin/env python
"""Experiment C: blend degraded (with interval) + clean (no interval) records."""
import json, random, argparse, os

CLEAN_TEMPLATES = [
    "This synthesized speech is clean and natural throughout. It has a high overall MOS score of 5.0. There is no degradation in the clip.",
    "This speech sounds clear and natural with no audible artifacts. The overall MOS score is 5.0. There is no degradation in the clip.",
    "The speech is clean and continuous, free of noise or distortion. Its overall MOS score is 5.0. There is no degradation in the clip.",
    "This synthesized speech is high quality, clear and natural. It reaches an overall MOS score of 5.0. There is no degradation in the clip.",
    "The clip is clean throughout, with natural and intelligible speech. The overall MOS score is 5.0. There is no degradation in the clip.",
    "This speech is clear, natural, and free of any audible degradation. It has a high overall MOS score of 5.0. There is no degradation in the clip.",
]
QUERY = "Please describe and evaluate the synthetic speech, and identify when the degradation occurs.<audio>"

def rel(p):
    return p.split("/data/", 1)[1] if "/data/" in p else p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degraded-jsonl", default="data/processed/temporal/train_nisqa_temporal_gc_timelast_aug_anchoroffset.json")
    ap.add_argument("--caption-jsonl", default="data/processed/sft/train_nisqa_llama_10k.json")
    ap.add_argument("--output-jsonl", default="data/processed/temporal/train_nisqa_temporal_expc_detect_blend.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    deg = [json.loads(l) for l in open(args.degraded_jsonl)]
    for r in deg:
        r["is_clean"] = False
    cap = [json.loads(l) for l in open(args.caption_jsonl)]
    seen, clean = set(), []
    for c in cap:
        cp = c.get("clean_path")
        if not cp:
            continue
        p = cp[0] if isinstance(cp, list) else cp
        if p in seen:
            continue
        seen.add(p)
        clean.append({"id": f"clean_{len(clean):05d}", "audios": [rel(p)], "response": rng.choice(CLEAN_TEMPLATES), "query": QUERY, "mos": 5.0, "is_clean": True})
    blend = deg + clean
    rng.shuffle(blend)
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w") as f:
        for r in blend:
            f.write(json.dumps(r) + "\n")
    print(f"degraded={len(deg)} clean={len(clean)} total={len(blend)} clean_frac={len(clean)/len(blend):.3f}")
    print("output:", args.output_jsonl)

if __name__ == "__main__":
    main()
