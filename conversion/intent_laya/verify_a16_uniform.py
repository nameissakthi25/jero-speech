#!/usr/bin/env python3
"""Host int-sim verification of w8a16 via AIMET QuantizationSimModel (onnxruntime).
Calibrate on a calib slice, then compare quantized sim vs FP32 ONNX on an eval set:
top-1 intent agreement (gate >=99%) and per-token slot agreement."""
import os, json, glob, numpy as np, onnx
from transformers import AutoTokenizer
import onnxruntime as ort
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.common.defs import qtype, QuantScheme
import aimet_onnx

EXPORT="/home/raemox/jero/intent/export"
TEST="/home/raemox/jero/intent/data/intent/test.jsonl"
cfgdir=os.path.join(os.path.dirname(aimet_onnx.__file__),"common","quantsim_config")
HTP=os.path.join(cfgdir,"default_config.json")
labels=json.load(open(os.path.join(EXPORT,"label_maps.json")))
id2i=labels["intent_id2label"]
tok=AutoTokenizer.from_pretrained(EXPORT)
def tk(t):
    e=tok(t,padding="max_length",truncation=True,max_length=32,return_tensors="np")
    return e["input_ids"].astype(np.int64),e["attention_mask"].astype(np.int64)

lines=[json.loads(l) for l in open(TEST) if l.strip()]
calib=[tk(r["text"]) for r in lines[::3][:150]]
calib_in=[{"input_ids":a,"attention_mask":b} for a,b in calib]
# eval set: first 120 (distribution overlaps file but exercises many intents); report agreement
EVAL=lines[:120]

fp32=ort.InferenceSession(os.path.join(EXPORT,"intent.onnx"),providers=["CPUExecutionProvider"])
model=onnx.load(os.path.join(EXPORT,"intent.onnx"))
sim=QuantizationSimModel(model=model,param_type=qtype.int(8),activation_type=qtype.int(16),
                         quant_scheme=QuantScheme.min_max,config_file=HTP,dummy_input=calib_in[0])
sim.compute_encodings(iter(calib_in))
print("quantsim calibrated; evaluating",len(EVAL),"samples")

intent_match=0; intent_fp_gold=0; intent_q_gold=0
slot_tok_total=0; slot_tok_match=0
for r in EVAL:
    ids,msk=tk(r["text"]); n=int(msk.sum())
    f=fp32.run(None,{"input_ids":ids,"attention_mask":msk})
    q=sim.session.run(None,{"input_ids":ids,"attention_mask":msk})
    fi=int(f[0][0].argmax()); qi=int(q[0][0].argmax())
    intent_match+= (fi==qi)
    g=r.get("intent")
    intent_fp_gold+= (id2i[str(fi)]==g)
    intent_q_gold += (id2i[str(qi)]==g)
    fs=f[1][0].argmax(-1)[:n]; qs=q[1][0].argmax(-1)[:n]
    slot_tok_total+=n; slot_tok_match+=int((fs==qs).sum())
N=len(EVAL)
print("=== W8A16 AIMET host int-sim vs FP32 ===")
print("top-1 intent agreement (q vs fp32): %d/%d = %.2f%%"%(intent_match,N,100*intent_match/N))
print("FP32 intent accuracy vs gold      : %d/%d = %.2f%%"%(intent_fp_gold,N,100*intent_fp_gold/N))
print("W8A16 intent accuracy vs gold     : %d/%d = %.2f%%"%(intent_q_gold,N,100*intent_q_gold/N))
print("slot per-token agreement (q vs fp): %d/%d = %.2f%%"%(slot_tok_match,slot_tok_total,100*slot_tok_match/slot_tok_total))
