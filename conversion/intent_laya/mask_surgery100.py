#!/usr/bin/env python3
"""Graph surgery: replace the ~-3.4e38 (FP32 finfo.min) attention-mask fill value
with a modest -10000 so int16 min-max quantization keeps a sane range.
Softmax(score + -1e4) ~ 0 for masked positions, same as -inf, so FP32 output is unchanged."""
import numpy as np, onnx
from onnx import numpy_helper
SRC="/home/raemox/jero/intent/export/intent.onnx"
DST="/home/raemox/jero/intent/qnn/intent_maskfix100.onnx"
BIG=1e30
NEW=-1.0e2

m=onnx.load(SRC)
g=m.graph
n_init=0; n_const=0; hits=[]
# 1) initializers
for init in g.initializer:
    arr=numpy_helper.to_array(init)
    if arr.dtype in (np.float32,np.float16,np.float64) and np.any(np.abs(arr)>BIG):
        a=arr.copy(); mask=np.abs(a)>BIG
        a[mask]=np.sign(a[mask])*abs(NEW)
        # ensure negative fills become NEW (they are negative)
        new_init=numpy_helper.from_array(a.astype(arr.dtype), init.name)
        init.CopyFrom(new_init); n_init+=1; hits.append((init.name,"init",int(mask.sum()),float(arr.flat[np.argmax(np.abs(arr))])))
# 2) Constant nodes
for node in g.node:
    if node.op_type!="Constant": continue
    for att in node.attribute:
        if att.name=="value" and att.t.data_type in (onnx.TensorProto.FLOAT,onnx.TensorProto.FLOAT16,onnx.TensorProto.DOUBLE):
            arr=numpy_helper.to_array(att.t)
            if np.any(np.abs(arr)>BIG):
                a=arr.copy(); mask=np.abs(a)>BIG
                a[mask]=np.sign(a[mask])*abs(NEW)
                att.t.CopyFrom(numpy_helper.from_array(a.astype(arr.dtype)))
                n_const+=1; hits.append((node.name,"const",int(mask.sum()),float(arr.flat[np.argmax(np.abs(arr))])))
print("patched initializers:",n_init," constant-nodes:",n_const)
for h in hits[:20]: print("  ",h)
onnx.checker.check_model(m) if m.ByteSize()<2**31 else print("(skip checker: >2GB)")
onnx.save(m,DST)
print("saved",DST)
