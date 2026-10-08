# Laya-style split surgery for the Jero intent ModernBERT-base encoder.
# host: embedding lookup -> inputs_embeds ; attention_mask -> mask bias (Expand) ; + heads
# board: encoder body (inputs_embeds, maskbias) -> last_hidden_state  [quantized w8a16]
import onnx, numpy as np
from onnx import helper, TensorProto, numpy_helper
SRC="/home/raemox/jero/intent/export/intent.onnx"
OUT="/home/raemox/jero/intent/qnn/laya_split"
import os; os.makedirs(OUT, exist_ok=True)
EMB_GATHER="/model/encoder/embeddings/tok_embeddings/Gather"
EMB_OUT="/model/encoder/embeddings/tok_embeddings/Gather_output_0"
MASKBIAS="/model/encoder/Expand_output_0"
ENC_OUT="/model/encoder/final_norm/LayerNormalization_output_0"

m=onnx.load(SRC); g=m.graph
# save embedding table for host-side lookup
et=[numpy_helper.to_array(i) for i in g.initializer if i.name=="model.encoder.embeddings.tok_embeddings.weight"][0]
np.save(f"{OUT}/embed_table.npy", et); print("embed_table", et.shape, et.dtype)
# save head weights for host-side heads
inits={i.name:numpy_helper.to_array(i) for i in g.initializer}
import json
head_names=[n for n in inits if "intent_head" in n or "slot_head" in n]
np.savez(f"{OUT}/heads.npz", **{n:inits[n] for n in head_names})
print("head tensors:", head_names)

# 1) embedding surgery: remove the tok_embeddings Gather, feed inputs_embeds
gath=[n for n in g.node if n.name==EMB_GATHER][0]; g.node.remove(gath)
H=et.shape[1]
for n in g.node:
    n.input[:]=["inputs_embeds" if i==EMB_OUT else i for i in n.input]
g.input.extend([helper.make_tensor_value_info("inputs_embeds",TensorProto.FLOAT,[1,32,H])])
# mask surgery: -3.4e38 (abs>1e30) fills -> -1e4 in initializers (so host-fed maskbias is sane)
BIG=1e30
for init in g.initializer:
    a=numpy_helper.to_array(init)
    if a.dtype in (np.float32,np.float16) and np.any(np.abs(a)>BIG):
        b=a.copy(); b[np.abs(b)>BIG]=np.sign(b[np.abs(b)>BIG])*1e4
        init.CopyFrom(numpy_helper.from_array(b.astype(a.dtype),init.name))
onnx.save(m,f"{OUT}/intent_emb.onnx")
print("intent_emb.onnx saved; inputs:", [i.name for i in g.input])

# 2) extract encoder body: [inputs_embeds, maskbias] -> last_hidden_state
onnx.utils.extract_model(f"{OUT}/intent_emb.onnx", f"{OUT}/intent_encoder.onnx",
    ["inputs_embeds", MASKBIAS], [ENC_OUT])
enc=onnx.load(f"{OUT}/intent_encoder.onnx")
print("intent_encoder.onnx in:", [i.name for i in enc.graph.input], "out:", [o.name for o in enc.graph.output])

# 3) mask-only model: attention_mask -> maskbias (host computes the additive bias)
onnx.utils.extract_model(SRC, f"{OUT}/mask_only.onnx", ["attention_mask"], [MASKBIAS])
# mask surgery on the mask-only graph too
mo=onnx.load(f"{OUT}/mask_only.onnx")
for init in mo.graph.initializer:
    a=numpy_helper.to_array(init)
    if a.dtype in (np.float32,np.float16) and np.any(np.abs(a)>BIG):
        b=a.copy(); b[np.abs(b)>BIG]=np.sign(b[np.abs(b)>BIG])*1e4
        init.CopyFrom(numpy_helper.from_array(b.astype(a.dtype),init.name))
onnx.save(mo,f"{OUT}/mask_only.onnx")
print("mask_only.onnx in:", [i.name for i in mo.graph.input], "out:", [o.name for o in mo.graph.output])
print("DONE surgery")
