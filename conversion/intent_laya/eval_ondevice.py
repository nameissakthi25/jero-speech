#!/usr/bin/env python3
"""On-device eval: board encoder last_hidden -> host heads -> compare top-1 intent to FP32.
Runs on the A100. Board outputs pulled to OUTDIR (Result_0..119)."""
import os, json, glob, numpy as np, onnxruntime as ort
from collections import Counter
D="/home/raemox/jero/intent/qnn/encsplit"
OUTDIR=os.environ.get("OUTDIR","/home/raemox/jero/intent/qnn/enc_dlc/board_out")
VER=D+"/verify"
ENC_OUT="/model/encoder/final_norm/LayerNormalization_output_0"
hd=ort.InferenceSession(D+"/intent_heads.onnx",providers=["CPUExecutionProvider"])
ref=json.load(open(VER+"/fp32_ref.json")); id2i=json.load(open("/home/raemox/jero/intent/export/label_maps.json"))["intent_id2label"]
im=0; sm=0; st=0; qh=Counter(); miss=0
for r in ref:
    i=r["i"]
    rd=os.path.join(OUTDIR,"Result_%d"%i)
    fs=glob.glob(os.path.join(rd,"*.raw"))
    if not fs: miss+=1; continue
    lh=np.fromfile(fs[0],dtype=np.float32).reshape(1,32,768)
    msk=np.fromfile(os.path.join(VER,"mask_%03d.raw"%i),dtype=np.int32).reshape(1,32).astype(np.int64)
    o=hd.run(None,{ENC_OUT:lh,"attention_mask":msk})
    qi=int(o[0][0].argmax()); qh[qi]+=1
    im+=(qi==r["fp32_intent"])
    n=r["nreal"]; st+=n
    sm+=int((np.array(o[1][0].argmax(-1)[:n])==np.array(r["fp32_slots"])).sum())
N=len(ref)
print("=== ON-DEVICE (QCS6490/v68) split encoder w8a16 vs FP32 (120 utts) ===")
print("missing results:",miss)
print("top-1 intent agreement: %d/%d = %.2f%%"%(im,N,100*im/N))
print("slot per-token agreement: %d/%d = %.2f%%"%(sm,st,100*sm/st))
print("device intent hist:",dict(qh.most_common()))
