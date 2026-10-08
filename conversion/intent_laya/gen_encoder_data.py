#!/usr/bin/env python3
"""Generate encoder calibration (100) + verification (120) raws: inputs_embeds[f32] + attention_mask[i32].
Also write fp32 head reference for on-device agreement, and board input_lists."""
import os, json, numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort

EXPORT="/home/raemox/jero/intent/export"; TEST="/home/raemox/jero/intent/data/intent/test.jsonl"
D="/home/raemox/jero/intent/qnn/encsplit"
CAL=os.path.join(D,"calib"); VER=os.path.join(D,"verify")
BOARD="/tmp/jero_enc"           # board staging dir
os.makedirs(CAL,exist_ok=True); os.makedirs(VER,exist_ok=True)
tok=AutoTokenizer.from_pretrained(EXPORT)
labels=json.load(open(EXPORT+"/label_maps.json")); id2i=labels["intent_id2label"]
W=np.load(D+"/embed_table.npy")
full=ort.InferenceSession(EXPORT+"/intent.onnx",providers=["CPUExecutionProvider"])
lines=[json.loads(l) for l in open(TEST) if l.strip()]

def emb_mask(text):
    e=tok(text,padding="max_length",truncation=True,max_length=32,return_tensors="np")
    ids=e["input_ids"].astype(np.int64); msk=e["attention_mask"]
    ie=W[ids[0]].reshape(1,32,768).astype(np.float32)
    return ie, msk.astype(np.int32), ids

# calibration: 100 spread across file
sel=lines[::max(1,len(lines)//100)][:100]
cl=open(os.path.join(CAL,"input_list.txt"),"w")
clb=open(os.path.join(CAL,"input_list_board.txt"),"w")
for i,r in enumerate(sel):
    ie,msk,_=emb_mask(r["text"])
    pie=os.path.join(CAL,"ie_%03d.raw"%i); pm=os.path.join(CAL,"mask_%03d.raw"%i)
    ie.tofile(pie); msk.tofile(pm)
    cl.write("inputs_embeds:=%s attention_mask:=%s\n"%(pie,pm))
    clb.write("inputs_embeds:=%s/calib/ie_%03d.raw attention_mask:=%s/calib/mask_%03d.raw\n"%(BOARD,i,BOARD,i))
cl.close(); clb.close()
print("calib:",len(sel))

# verification: first 120 + fp32 intent/slot reference
ver=lines[:120]
vl=open(os.path.join(VER,"input_list_board.txt"),"w")
ref=[]
for i,r in enumerate(ver):
    ie,msk,ids=emb_mask(r["text"])
    ie.tofile(os.path.join(VER,"ie_%03d.raw"%i)); msk.tofile(os.path.join(VER,"mask_%03d.raw"%i))
    vl.write("inputs_embeds:=%s/verify/ie_%03d.raw attention_mask:=%s/verify/mask_%03d.raw\n"%(BOARD,i,BOARD,i))
    f=full.run(None,{"input_ids":ids,"attention_mask":msk.astype(np.int64)})
    n=int(msk.sum())
    ref.append({"i":i,"text":r["text"],"gold":r.get("intent"),
                "fp32_intent":int(f[0][0].argmax()),
                "fp32_intent_name":id2i[str(int(f[0][0].argmax()))],
                "nreal":n,"fp32_slots":[int(x) for x in f[1][0].argmax(-1)[:n]]})
vl.close()
json.dump(ref,open(os.path.join(VER,"fp32_ref.json"),"w"))
print("verify:",len(ver))
print("ie bytes:",os.path.getsize(os.path.join(VER,"ie_000.raw")),"mask bytes:",os.path.getsize(os.path.join(VER,"mask_000.raw")))
print("DONE_GEN")
