#!/usr/bin/env python3
"""Generate 120-utt board test bundle: int32 raws + input_list (board paths) + FP32 reference."""
import os, json, numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
EXPORT="/home/raemox/jero/intent/export"; TEST="/home/raemox/jero/intent/data/intent/test.jsonl"
VB="/home/raemox/jero/intent/qnn/verify_board"; RAW=os.path.join(VB,"vraw")
BOARD_DIR="/tmp/jero_intent"            # where files will live on the board
os.makedirs(RAW,exist_ok=True)
tok=AutoTokenizer.from_pretrained(EXPORT)
labels=json.load(open(os.path.join(EXPORT,"label_maps.json"))); id2i=labels["intent_id2label"]
lines=[json.loads(l) for l in open(TEST) if l.strip()]
sel=lines[:120]
so=ort.InferenceSession(os.path.join(EXPORT,"intent.onnx"),providers=["CPUExecutionProvider"])
lf=open(os.path.join(VB,"input_list_board.txt"),"w")
ref=[]
for i,r in enumerate(sel):
    e=tok(r["text"],padding="max_length",truncation=True,max_length=32,return_tensors="np")
    ids=e["input_ids"].astype(np.int32); msk=e["attention_mask"].astype(np.int32)
    ids.tofile(os.path.join(RAW,"ids_%03d.raw"%i)); msk.tofile(os.path.join(RAW,"mask_%03d.raw"%i))
    lf.write("input_ids:=%s/vraw/ids_%03d.raw attention_mask:=%s/vraw/mask_%03d.raw\n"%(BOARD_DIR,i,BOARD_DIR,i))
    out=so.run(None,{"input_ids":ids.astype(np.int64),"attention_mask":msk.astype(np.int64)})
    n=int(msk.sum())
    ref.append({"i":i,"text":r["text"],"gold":r.get("intent"),
                "fp32_intent":int(out[0][0].argmax()),
                "fp32_intent_name":id2i[str(int(out[0][0].argmax()))],
                "nreal":n,"fp32_slots":[int(x) for x in out[1][0].argmax(-1)[:n]]})
lf.close()
json.dump(ref,open(os.path.join(VB,"fp32_ref_board.json"),"w"))
print("wrote",len(sel),"utts to",RAW)
print("input_list:",os.path.join(VB,"input_list_board.txt"))
print("output order:",[o.name for o in so.get_outputs()])
