#!/usr/bin/env python3
"""Split intent ModernBERT into: host embedding lookup + NPU encoder body + host heads.
Reason: V68 HTP returns WRONG ROWS for on-chip integer Gather, so the token-embedding
lookup must run on the host. NPU gets inputs_embeds and runs only the transformer encoder."""
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

SRC = "/home/raemox/jero/intent/export/intent.onnx"
OUT = "/home/raemox/jero/intent/qnn/encsplit"
import os; os.makedirs(OUT, exist_ok=True)
L, H = 32, 768
GATHER_OUT = "/model/encoder/embeddings/tok_embeddings/Gather_output_0"
ENC_OUT = "/model/encoder/final_norm/LayerNormalization_output_0"
BIG = 1e30; NEWMASK = -1.0e4

m = onnx.load(SRC); g = m.graph

# --- save embedding table for host-side lookup ---
tbl_name = "model.encoder.embeddings.tok_embeddings.weight"
tbl = next(numpy_helper.to_array(i) for i in g.initializer if i.name == tbl_name)
np.save(os.path.join(OUT, "embed_table.npy"), tbl.astype(np.float32))
print("embed_table.npy", tbl.shape, tbl.dtype)

# --- embed surgery: remove tok_embeddings Gather, expose inputs_embeds ---
GE = [n for n in g.node if n.op_type == "Gather" and "tok_embeddings/Gather" in n.name]
assert len(GE) == 1
gath = GE[0]; out = gath.output[0]
assert out == GATHER_OUT, out
g.node.remove(gath)
for n in g.node:
    n.input[:] = ["inputs_embeds" if i == out else i for i in n.input]
g.input.extend([helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, L, H])])

# --- mask surgery: -3.4e38 -> -1e4 (Constant nodes) ---
patched = 0
for node in g.node:
    if node.op_type != "Constant": continue
    for att in node.attribute:
        if att.name == "value" and att.t.data_type in (TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.FLOAT16):
            arr = numpy_helper.to_array(att.t)
            if np.any(np.abs(arr) > BIG):
                a = arr.copy(); mk = np.abs(a) > BIG; a[mk] = np.sign(a[mk]) * abs(NEWMASK)
                att.t.CopyFrom(numpy_helper.from_array(a.astype(arr.dtype)))
                patched += 1
print("mask constants patched:", patched)

# prune unused inputs/initializers
used = set(o.name for o in g.output)
for n in g.node:
    for i in n.input: used.add(i)
keepin = [i for i in g.input if i.name in used or i.name == "inputs_embeds"]
del g.input[:]; g.input.extend(keepin)
keepi = [i for i in g.initializer if i.name in used]
del g.initializer[:]; g.initializer.extend(keepi)
EMB = os.path.join(OUT, "intent_emb.onnx")
onnx.save(m, EMB)
print("intent_emb.onnx inputs:", [i.name for i in g.input])

# --- cut encoder body: [inputs_embeds, attention_mask] -> ENC_OUT ---
onnx.utils.extract_model(EMB, os.path.join(OUT, "intent_encoder.onnx"),
    ["inputs_embeds", "attention_mask"], [ENC_OUT])
enc = onnx.load(os.path.join(OUT, "intent_encoder.onnx"))
print("ENCODER in:", [i.name for i in enc.graph.input], "out:", [o.name for o in enc.graph.output])

# --- cut heads (host): [ENC_OUT, attention_mask] -> intent_logits, slot_logits ---
onnx.utils.extract_model(EMB, os.path.join(OUT, "intent_heads.onnx"),
    [ENC_OUT, "attention_mask"], ["intent_logits", "slot_logits"])
hd = onnx.load(os.path.join(OUT, "intent_heads.onnx"))
print("HEADS in:", [i.name for i in hd.graph.input], "out:", [o.name for o in hd.graph.output])
print("DONE_SPLIT")
