#!/usr/bin/env python
"""Exp C: clean test set from held-out NISQA_VAL_SIM refs (disjoint from training).

Resample to 16kHz/6s (match mixes). Response = clean template (no interval), so
ground-truth interval = None. False-alarm rate = fraction where the model still
emits an interval (read samples_with_parsed_prediction_interval/samples_total).
"""
import json, os, random, glob
import numpy as np, soundfile as sf, librosa

QUERY="Please describe and evaluate the synthetic speech, and identify when the degradation occurs.<audio>"
CLEAN_RESP="This synthesized speech is clean and natural throughout. It has a high overall MOS score of 5.0. There is no degradation in the clip."
SR=16000; DUR=6.0; N=250; SEED=123
out_dir="data/processed/temporal/clean_test_16k"
out_json="data/processed/temporal/test_CLEAN_valsim_16k.json"
os.makedirs(out_dir, exist_ok=True)
refs=sorted(glob.glob("data/raw/NISQA_Corpus/NISQA_VAL_SIM/ref/*.wav"))
random.Random(SEED).shuffle(refs)
refs=refs[:N]
target=int(SR*DUR)
recs=[]
for i,p in enumerate(refs):
    y,sr=sf.read(p)
    if y.ndim>1: y=y.mean(axis=1)
    if sr!=SR: y=librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=SR)
    y=y[:target]
    if len(y)<target: y=np.pad(y,(0,target-len(y)))
    name=f"cleantest_{i:04d}.wav"; outp=os.path.join(out_dir,name)
    sf.write(outp,y,SR)
    rel=outp[len("data/"):]
    recs.append({"id":f"cleantest_{i:04d}","audios":[rel],"response":CLEAN_RESP,"query":QUERY,"mos":5.0,"is_clean":True})
with open(out_json,"w") as f:
    for r in recs: f.write(json.dumps(r)+"\n")
print(f"clean test clips={len(recs)} -> {out_json}")
