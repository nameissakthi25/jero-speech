#!/usr/bin/env python3
"""Verify the mask-fix graph: (1) FP32 maskfix == FP32 original (lossless surgery),
(2) W8A16 quantsim on maskfix vs FP32 original -> top-1 intent agreement (gate>=99%)."""
import os, json, numpy as np, onnx
from collections import Counter
from transformers import AutoTokenizer
import onnxruntime as ort
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.common.defs import qtype, QuantScheme
import aimet_onnx

EXPORT="/home/raemox/jero/intent/export"; TEST="/home/raemox/jero/intent/data/intent/test.jsonl"
ORIG=os.path.join(EXPORT,"intent.onnx")
FIX="/home/raemox/jero/intent/qnn/intent_maskfix.onnx"
cfgdir=os.path.join(os.path.dirname(aimet_onnx.__file__),"common","quantsim_config")
HTP=os.path.join(cfgdir,"htp_quantsim_config_v68.json")   # deployment-realistic config
labels=json.load(open(os.path.join(EXPORT,"label_maps.json"))); id2i=labels["intent_id2label"]
tok=AutoTokenizer.from_pretrained(EXPORT)
def tk(t):
    e=tok(t,padding="max_length",truncation=True,max_length=32,return_tensors="np")
    return e["input_ids"].astype(np.int64),e["attention_mask"].astype(np.int64)
lines=[json.loads(l) for l in open(TEST) if l.strip()]
calib=[{"input_ids":a,"attention_mask":b} for a,b in (tk(r["text"]) for r in lines[::3][:150])]
EVAL=lines[:120]

fp32=ort.InferenceSession(ORIG,providers=["CPUExecutionProvider"])
fp32fix=ort.InferenceSession(FIX,providers=["CPUExecutionProvider"])

# (1) surgery lossless check
same=0; slot_same=0; slot_tot=0
for r in EVAL:
    ids,msk=tk(r["text"]); n=int(msk.sum())
    a=fp32.run(None,{"input_ids":ids,"attention_mask":msk})
    b=fp32fix.run(None,{"input_ids":ids,"attention_mask":msk})
    same+=(int(a[0][0].argmax())==int(b[0][0].argmax()))
    slot_tot+=n; slot_same+=int((a[1][0].argmax(-1)[:n]==b[1][0].argmax(-1)[:n]).sum())
print("surgery lossless: intent %d/%d  slot-tok %d/%d"%(same,len(EVAL),slot_same,slot_tot))

# (2) W8A16 quantsim on maskfix vs fp32 original
model=onnx.load(FIX)
sim=QuantizationSimModel(model=model,param_type=qtype.int(8),activation_type=qtype.int(16),
                         quant_scheme=QuantScheme.min_max,config_file=HTP,dummy_input=calib[0])
sim.compute_encodings(iter(calib))
m=0; q_gold=0; fp_gold=0; st=0; sm=0; qh=Counter()
for r in EVAL:
    ids,msk=tk(r["text"]); n=int(msk.sum())
    f=fp32.run(None,{"input_ids":ids,"attention_mask":msk})
    q=sim.session.run(None,{"input_ids":ids,"attention_mask":msk})
    fi=int(f[0][0].argmax()); qi=int(q[0][0].argmax())
    m+=(fi==qi); qh[id2i[str(qi)]]+=1
    g=r.get("intent"); fp_gold+=(id2i[str(fi)]==g); q_gold+=(id2i[str(qi)]==g)
    st+=n; sm+=int((f[1][0].argmax(-1)[:n]==q[1][0].argmax(-1)[:n]).sum())
N=len(EVAL)
print("=== W8A16 (mask-fixed) AIMET host int-sim vs FP32 ===")
print("top-1 intent agreement (q vs fp32): %d/%d = %.2f%%"%(m,N,100*m/N))
print("FP32 intent acc vs gold           : %d/%d = %.2f%%"%(fp_gold,N,100*fp_gold/N))
print("W8A16 intent acc vs gold          : %d/%d = %.2f%%"%(q_gold,N,100*q_gold/N))
print("slot per-token agreement          : %d/%d = %.2f%%"%(sm,st,100*sm/st))
print("q pred hist:",dict(qh.most_common()))
print("DONE_MASKFIX")
