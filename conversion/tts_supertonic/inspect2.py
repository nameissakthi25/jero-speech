import onnx, numpy as np
from onnx import numpy_helper
C="/home/raemox/.cache/supertonic3/onnx/"
def big_negs(g):
    hits=[]
    for t in g.initializer:
        a=numpy_helper.to_array(t)
        if a.dtype.kind=="f" and a.size>0 and np.nanmin(a)< -1e4:
            hits.append((t.name, float(np.nanmin(a)), list(a.shape)))
    for n in g.node:
        if n.op_type=="Constant":
            for att in n.attribute:
                if att.name=="value":
                    a=numpy_helper.to_array(att.t)
                    if a.dtype.kind=="f" and a.size>0 and np.nanmin(a)< -1e4:
                        hits.append((n.name+"(const)", float(np.nanmin(a)), list(a.shape)))
    return hits
for name in ["text_encoder","duration_predictor","vector_estimator","vocoder"]:
    m=onnx.load(C+name+".onnx"); g=m.graph
    print("\n####", name)
    print(" IN:",[(i.name,[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in g.input])
    print(" OUT:",[(o.name,[d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in g.output])
    inits={t.name:t for t in g.initializer}
    # token-embed gathers: Gather whose data is a 2D float initializer and idx is a graph input or int tensor
    for n in g.node:
        if n.op_type=="Gather" and n.input[0] in inits:
            arr=numpy_helper.to_array(inits[n.input[0]])
            if arr.ndim==2 and arr.shape[0]>1000:
                print("  TOKEN-EMB Gather:",n.name,"data",n.input[0],list(arr.shape),"idx",n.input[1],"->",n.output[0])
    bn=big_negs(g)
    print("  big-neg(<-1e4) count:",len(bn))
    for h in bn[:6]: print("     ",h)
    # Where nodes (mask fills)
    wh=[n for n in g.node if n.op_type=="Where"]
    print("  Where nodes:",len(wh))
