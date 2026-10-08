import onnx, sys
from onnx import numpy_helper
C="/home/raemox/.cache/supertonic3/onnx/"
for name in ["text_encoder","duration_predictor"]:
    m=onnx.load(C+name+".onnx"); g=m.graph
    print("\n########", name, "########")
    print("INPUTS:", [(i.name,[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in g.input])
    print("OUTPUTS:", [(o.name,[d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in g.output])
    inits={t.name:t for t in g.initializer}
    print("Gather nodes:")
    for n in g.node:
        if n.op_type=="Gather":
            data=n.input[0]; idx=n.input[1]
            dshape = list(numpy_helper.to_array(inits[data]).shape) if data in inits else "(dynamic)"
            print("  ",n.name,"| data=",data,dshape,"| idx=",idx,"| out=",n.output[0])
    # find which gather consumes text_ids (directly or via Cast)
    print("nodes consuming text_ids:")
    for n in g.node:
        if "text_ids" in n.input:
            print("  ",n.op_type,n.name,"inputs=",list(n.input))
