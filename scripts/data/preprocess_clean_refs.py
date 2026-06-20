#!/usr/bin/env python
"""Exp C: resample unique clean refs to 16 kHz / 6 s (match the mixes), emit clean records.

Matching SR and duration to the degraded mixes removes a 3-4x training slowdown
AND removes clip-length as a trivial clean-vs-degraded cue.
"""
import json, os, random, argparse
import numpy as np, soundfile as sf
import librosa

CLEAN_TEMPLATES = [
    "This synthesized speech is clean and natural throughout. It has a high overall MOS score of 5.0. There is no degradation in the clip.",
    "This speech sounds clear and natural with no audible artifacts. The overall MOS score is 5.0. There is no degradation in the clip.",
    "The speech is clean and continuous, free of noise or distortion. Its overall MOS score is 5.0. There is no degradation in the clip.",
    "This synthesized speech is high quality, clear and natural. It reaches an overall MOS score of 5.0. There is no degradation in the clip.",
    "The clip is clean throughout, with natural and intelligible speech. The overall MOS score is 5.0. There is no degradation in the clip.",
    "This speech is clear, natural, and free of any audible degradation. It has a high overall MOS score of 5.0. There is no degradation in the clip.",
]
QUERY = "Please describe and evaluate the synthetic speech, and identify when the degradation occurs.<audio>"
SR=16000; DUR=6.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--caption-jsonl", default="data/processed/sft/train_nisqa_llama_10k.json")
    ap.add_argument("--out-dir", default="data/processed/temporal/clean_refs_16k")
    ap.add_argument("--out-records", default="data/processed/temporal/expc_clean_records.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    a=ap.parse_args()
    rng=random.Random(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    cap=[json.loads(l) for l in open(a.caption_jsonl)]
    seen=set(); recs=[]; n=0
    target=int(SR*DUR)
    for c in cap:
        cp=c.get("clean_path")
        if not cp: continue
        p=cp[0] if isinstance(cp,list) else cp
        if p in seen: continue
        seen.add(p)
        y,sr=sf.read(p)
        if y.ndim>1: y=y.mean(axis=1)
        if sr!=SR: y=librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=SR)
        y=y[:target]
        if len(y)<target: y=np.pad(y,(0,target-len(y)))
        name=f"clean_{n:05d}.wav"
        outp=os.path.join(a.out_dir,name)
        sf.write(outp,y,SR)
        rel=outp.split("/data/",1)[1] if "/data/" in outp else ("data/"+outp if not outp.startswith("data/") else outp)
        # outp already starts with data/, make data-root-relative (strip leading data/)
        rel = outp[len("data/"):] if outp.startswith("data/") else outp
        recs.append({"id":f"clean_{n:05d}","audios":[rel],"response":rng.choice(CLEAN_TEMPLATES),"query":QUERY,"mos":5.0,"is_clean":True})
        n+=1
        if n%2000==0: print("done",n,flush=True)
    with open(a.out_records,"w") as f:
        for r in recs: f.write(json.dumps(r)+"\n")
    print(f"clean records={len(recs)} out_dir={a.out_dir} records={a.out_records}")

if __name__=="__main__": main()
