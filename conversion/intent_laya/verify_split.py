#!/usr/bin/env python3
"""Parity check: host-embed -> encoder(fp32) -> heads(fp32) must equal the full fp32 ONNX."""
import json, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer

EXPORT="/home/raemox/jero/intent/export"; TEST="/home/raemox/jero/intent/data/intent/test.jsonl"
D="/home/raemox/jero/intent/qnn/encsplit"
tok=AutoTokenizer.from_pretrained(EXPORT)
labels=json.load(open(EXPORT+"/label_maps.json")); id2i=labels["intent_id2label"]
W=np.load(D+"/embed_table.npy")  # [50368,768]
full=ort.InferenceSession(EXPORT+"/intent.onnx",providers=["CPUExecutionProvider"])
enc =ort.InferenceSession(D+"/intent_encoder.onnx",providers=["CPUExecutionProvider"])
hd  =ort.InferenceSession(D+"/intent_heads.onnx",providers=["CPUExecutionProvider"])
ENC_OUT="/model/encoder/final_norm/LayerNormalization_output_0"

lines=[json.loads(l) for l in open(TEST) if l.strip()][:120]
im=0; sm=0; st=0; ii_match=0
for r in lines:
    e=tok(r["text"],padding="max_length",truncation=True,max_length=32,return_tensors="np")
    ids=e["input_ids"].astype(np.int64); msk=e["attention_mask"].astype(np.int64); n=int(msk.sum())
    # full fp32
    f=full.run(None,{"input_ids":ids,"attention_mask":msk})
    fi=int(f[0][0].argmax())
    # split fp32: host embed -> encoder -> heads
    emb=W[ids[0]].reshape(1,32,768).astype(np.float32)
    lh=enc.run(None,{"inputs_embeds":emb,"attention_mask":msk})[0]
    o=hd.run(None,{ENC_OUT:lh,"attention_mask":msk})
    si=int(o[0][0].argmax())
    im+=(fi==si)
    st+=n; sm+=int((f[1][0].argmax(-1)[:n]==o[1][0].argmax(-1)[:n]).sum())
    ii_match+=(fi==id2i and False)  # placeholder
N=len(lines)
print("=== SPLIT FP32 PARITY vs FULL FP32 ===")
print("intent argmax match: %d/%d = %.2f%%"%(im,N,100*im/N))
print("slot per-token match: %d/%d = %.2f%%"%(sm,st,100*sm/st))
print("DONE_PARITY")
