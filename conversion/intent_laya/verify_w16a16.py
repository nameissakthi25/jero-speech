#!/usr/bin/env python3
"""Discriminator: W16A16 (near-lossless) vs FP32, plus W8A16 prediction histogram.
Isolates whether 8-bit WEIGHTS or an ACTIVATION quantizer causes the collapse."""
import os, json, numpy as np, onnx
from collections import Counter
from transformers import AutoTokenizer
import onnxruntime as ort
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.common.defs import qtype, QuantScheme
import aimet_onnx

EXPORT="/home/raemox/jero/intent/export"; TEST="/home/raemox/jero/intent/data/intent/test.jsonl"
cfgdir=os.path.join(os.path.dirname(aimet_onnx.__file__),"common","quantsim_config")
DEF=os.path.join(cfgdir,"default_config.json")
labels=json.load(open(os.path.join(EXPORT,"label_maps.json"))); id2i=labels["intent_id2label"]
tok=AutoTokenizer.from_pretrained(EXPORT)
def tk(t):
    e=tok(t,padding="max_length",truncation=True,max_length=32,return_tensors="np")
    return e["input_ids"].astype(np.int64),e["attention_mask"].astype(np.int64)
lines=[json.loads(l) for l in open(TEST) if l.strip()]
calib=[{"input_ids":a,"attention_mask":b} for a,b in (tk(r["text"]) for r in lines[::3][:150])]
EVAL=lines[:120]
fp32=ort.InferenceSession(os.path.join(EXPORT,"intent.onnx"),providers=["CPUExecutionProvider"])

def run_cfg(tag, pbw, abw):
    model=onnx.load(os.path.join(EXPORT,"intent.onnx"))
    sim=QuantizationSimModel(model=model,param_type=qtype.int(pbw),activation_type=qtype.int(abw),
                             quant_scheme=QuantScheme.min_max,config_file=DEF,dummy_input=calib[0])
    sim.compute_encodings(iter(calib))
    m=0; qhist=Counter(); fhist=Counter()
    for r in EVAL:
        ids,msk=tk(r["text"])
        f=fp32.run(None,{"input_ids":ids,"attention_mask":msk})
        q=sim.session.run(None,{"input_ids":ids,"attention_mask":msk})
        fi=int(f[0][0].argmax()); qi=int(q[0][0].argmax())
        m+=(fi==qi); qhist[id2i[str(qi)]]+=1; fhist[id2i[str(fi)]]+=1
    N=len(EVAL)
    print("[%s W%dA%d] top-1 agreement q-vs-fp32: %d/%d = %.2f%%"%(tag,pbw,abw,m,N,100*m/N))
    print("   q pred hist:",dict(qhist.most_common()))
    print("   fp pred hist:",dict(fhist.most_common()))
    return 100*m/N

run_cfg("CONTROL",16,16)
print("DONE_W16A16")
