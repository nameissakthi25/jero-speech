#!/usr/bin/env python3
"""AIMET w8a16 quantsim for Jero intent model (ModernBERT joint intent+slot).
W8 weights / A16 activations, HTP v68 config. Exports .encodings + onnx pair."""
import os, glob, json, sys
import numpy as np
import onnx
from transformers import AutoTokenizer
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.common.defs import qtype, QuantScheme

EXPORT = "/home/raemox/jero/intent/export"
ONNX   = "/home/raemox/jero/intent/qnn/intent_maskfix.onnx"
TEST   = "/home/raemox/jero/intent/data/intent/test.jsonl"
OUTDIR = "/home/raemox/jero/intent/qnn/aimet_out_maskfix"
MAXLEN = 32
N_CALIB = 150
os.makedirs(OUTDIR, exist_ok=True)

# ---- locate v68 HTP config ----
import aimet_onnx
cfgdir = os.path.join(os.path.dirname(aimet_onnx.__file__), "common", "quantsim_config")
HTP_CONFIG = os.path.join(cfgdir, "htp_quantsim_config_v68.json")
assert os.path.isfile(HTP_CONFIG), HTP_CONFIG
print("HTP config:", HTP_CONFIG)

# ---- tokenizer + calibration set from real test distribution ----
tok = AutoTokenizer.from_pretrained(EXPORT)
def tokenize(text):
    e = tok(text, padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="np")
    return {"input_ids": e["input_ids"].astype(np.int64),
            "attention_mask": e["attention_mask"].astype(np.int64)}

lines = [json.loads(l) for l in open(TEST) if l.strip()]
# spread across the file for distribution coverage
step = max(1, len(lines)//N_CALIB)
calib_lines = lines[::step][:N_CALIB]
calib_inputs = [tokenize(r["text"]) for r in calib_lines]
print(f"calibration samples: {len(calib_inputs)} (from {len(lines)} test rows)")

# ---- quantsim: W8 / A16 ----
model = onnx.load(ONNX)
dummy = calib_inputs[0]
sim = QuantizationSimModel(
    model=model,
    param_type=qtype.int(8),        # W8
    activation_type=qtype.int(16),  # A16 (INT16 fixed point, NOT fp16)
    quant_scheme=QuantScheme.min_max,
    config_file=HTP_CONFIG,
    dummy_input=dummy,
)
print("quantsim built. computing encodings...")

# ---- calibration ----
try:
    sim.compute_encodings(iter(calib_inputs))
except TypeError as ex:
    print("inputs-form failed, using callback:", ex)
    def cb(session, _=None):
        for s in calib_inputs:
            session.run(None, s)
    sim.compute_encodings(cb)
print("encodings computed.")

# ---- export ----
sim.export(path=OUTDIR, filename_prefix="intent")
enc_path = os.path.join(OUTDIR, "intent.encodings")
enc = json.load(open(enc_path))
act = enc.get("activation_encodings", {})
par = enc.get("param_encodings", {})
nact = len(act) if not isinstance(act, list) else len(act)
npar = len(par) if not isinstance(par, list) else len(par)
print(f"EXPORTED {enc_path}")
print(f"activation encodings: {nact}")
print(f"param encodings     : {npar}")

# report bitwidths seen
def bws(x):
    s=set()
    if isinstance(x, dict):
        for v in x.values():
            if isinstance(v, list):
                for r in v: s.add(r.get("bitwidth") or r.get("bw"))
            elif isinstance(v, dict):
                s.add(v.get("bitwidth") or v.get("bw"))
    elif isinstance(x, list):
        for r in x: s.add(r.get("bitwidth") or r.get("bw"))
    return sorted(b for b in s if b is not None)
print("activation bitwidths:", bws(act))
print("param bitwidths     :", bws(par))
assert nact > 0, "zero activation encodings -> compute_encodings did not run"
print("DONE")
