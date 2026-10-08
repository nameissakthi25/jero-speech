#!/usr/bin/env python3
"""Export the QAT'd Jero intent model -> fp32 ONNX (sdpa, opset17, [1,32]) ->
mask surgery (-3.4e38 -> -1e4). Mirrors training/intent/export.py + mask_surgery.py."""
import os, sys, json
sys.path.insert(0, "/home/raemox/jero/intent/training/intent")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch, onnx
from onnx import numpy_helper
import onnxruntime as ort
from transformers import AutoTokenizer
from labels import ID2INTENT, ID2SLOT
from model import LayaJointModel, ExportWrapper

BEST = "/home/raemox/jero/intent/runs/best"
CKPT = "/home/raemox/jero/intent/qnn/qat/ckpt_qat3_best.pt"
OUTDIR = "/home/raemox/jero/intent/qnn/qat"
L = 32
ONNX_RAW = f"{OUTDIR}/intent_qat.onnx"
ONNX_FIX = f"{OUTDIR}/intent_qat_maskfix.onnx"

tok = AutoTokenizer.from_pretrained(BEST)
# fresh SDPA model (structure+config from best), then overwrite with QAT weights
model, cfg = LayaJointModel.from_pretrained_like(BEST)
qsd = torch.load(CKPT, map_location="cpu")
# QAT ckpt wrapped Linears in QuantLinear whose .weight IS the orig param, so
# the weight keys match; observer buffers are extra -> strict=False.
clean = {k: v for k, v in qsd.items()
         if not any(s in k for s in (".out_obs.", ".q_obs.", ".k_obs.",
                                     ".p_obs.", ".v_obs.", ".s_obs.", ".sm_obs."))}
res = model.load_state_dict(clean, strict=False)
print("missing:", res.missing_keys[:6], "unexpected:", res.unexpected_keys[:6])
model.eval()
wrapper = ExportWrapper(model).eval()

dummy_ids = torch.ones(1, L, dtype=torch.long)
dummy_mask = torch.ones(1, L, dtype=torch.long)
torch.onnx.export(wrapper, (dummy_ids, dummy_mask), ONNX_RAW,
                  input_names=["input_ids", "attention_mask"],
                  output_names=["intent_logits", "slot_logits"],
                  opset_version=17, do_constant_folding=True, dynamic_axes=None)
print("exported", ONNX_RAW)

# verify ORT vs torch on 20 eval utts
sess = ort.InferenceSession(ONNX_RAW, providers=["CPUExecutionProvider"])
rows = [json.loads(l) for l in open("/home/raemox/jero/intent/data/intent/test.jsonl")][:20]
ia = 0
for r in rows:
    e = tok(r["text"].lower(), truncation=True, max_length=L, padding="max_length", return_tensors="np")
    ids = e["input_ids"].astype(np.int64); am = e["attention_mask"].astype(np.int64)
    with torch.no_grad():
        ti, _ = wrapper(torch.tensor(ids), torch.tensor(am))
    oi, _ = sess.run(None, {"input_ids": ids, "attention_mask": am})
    ia += int(ti.numpy().argmax(-1)[0] == oi.argmax(-1)[0])
print("ORT vs torch intent argmax agree: %d/20" % ia)

# ---- mask surgery: |x|>1e30 fills -> -1e4 ----
m = onnx.load(ONNX_RAW)
BIG = 1e30; NEW = -1.0e4; ni = nc = 0
for init in m.graph.initializer:
    arr = numpy_helper.to_array(init)
    if arr.dtype in (np.float32, np.float16, np.float64) and np.any(np.abs(arr) > BIG):
        a = arr.copy(); msk = np.abs(a) > BIG; a[msk] = np.sign(a[msk]) * abs(NEW)
        init.CopyFrom(numpy_helper.from_array(a.astype(arr.dtype), init.name)); ni += 1
for node in m.graph.node:
    if node.op_type != "Constant":
        continue
    for att in node.attribute:
        if att.name == "value" and att.t.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE):
            arr = numpy_helper.to_array(att.t)
            if np.any(np.abs(arr) > BIG):
                a = arr.copy(); msk = np.abs(a) > BIG; a[msk] = np.sign(a[msk]) * abs(NEW)
                att.t.CopyFrom(numpy_helper.from_array(a.astype(arr.dtype))); nc += 1
print("mask surgery: patched initializers=%d constants=%d" % (ni, nc))
onnx.save(m, ONNX_FIX)
print("saved", ONNX_FIX)
