#!/usr/bin/env python3
"""Prepare verification: pick N test utterances, write int32 raws + input_list,
and compute the FP32 ONNX reference (intent argmax, slot argmax per real token)."""
import os, json, numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort

EXPORT="/home/raemox/jero/intent/export"
TEST="/home/raemox/jero/intent/data/intent/test.jsonl"
VDIR="/home/raemox/jero/intent/qnn/verify"
RAW=os.path.join(VDIR,"raw")
N=25
os.makedirs(RAW,exist_ok=True)
tok=AutoTokenizer.from_pretrained(EXPORT)
labels=json.load(open(os.path.join(EXPORT,"label_maps.json")))
id2intent=labels["intent_id2label"]

lines=[json.loads(l) for l in open(TEST) if l.strip()]
# use a spread offset from calibration (calib used [::step]); take an interior window
sel=lines[5:5+N]
so=ort.InferenceSession(os.path.join(EXPORT,"intent.onnx"),providers=["CPUExecutionProvider"])

listf=open(os.path.join(VDIR,"verify_list.txt"),"w")
ref=[]
for i,r in enumerate(sel):
    e=tok(r["text"],padding="max_length",truncation=True,max_length=32,return_tensors="np")
    ids=e["input_ids"].astype(np.int32); msk=e["attention_mask"].astype(np.int32)
    pids=os.path.join(RAW,"ids_%03d.raw"%i); pmsk=os.path.join(RAW,"mask_%03d.raw"%i)
    ids.tofile(pids); msk.tofile(pmsk)
    listf.write("input_ids:=%s attention_mask:=%s\n"%(pids,pmsk))
    out=so.run(None,{"input_ids":ids.astype(np.int64),"attention_mask":msk.astype(np.int64)})
    intent_logits=out[0][0]; slot_logits=out[1][0]
    nreal=int(msk.sum())
    ref.append({
        "i":i,"text":r["text"],"gold_intent":r.get("intent"),
        "fp32_intent":int(intent_logits.argmax()),
        "fp32_intent_name":id2intent[str(int(intent_logits.argmax()))],
        "nreal":nreal,
        "fp32_slots":[int(x) for x in slot_logits.argmax(-1)[:nreal]],
    })
listf.close()
json.dump(ref,open(os.path.join(VDIR,"fp32_ref.json"),"w"),indent=1)
print("prepared",len(sel),"samples;",VDIR)
print("outputs order:",[o.name for o in so.get_outputs()])
