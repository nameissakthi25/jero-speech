import json, onnx
enc=json.load(open("aimet_out/intent.encodings"))
act={r["name"]:r for r in enc["activation_encodings"]}
par={r["name"]:r for r in enc["param_encodings"]}
# any encoding on inputs / embedding indices?
for key in ["input_ids","attention_mask"]:
    print(key, "in act:", key in act)
# find gather/embedding related encoded tensors
m=onnx.load("aimet_out/intent.onnx")
g=m.graph
prod={o:n for n in g.node for o in n.output}
# the 8-bit activations
eights=[r for r in enc["activation_encodings"] if r["bw"]==8]
print("num 8-bit activations:",len(eights))
for r in eights[:50]:
    n=prod.get(r["name"])
    print("  8bit:",r["name"],"| producer:", n.op_type if n else "INPUT/INIT")
# check embedding gather output encoding bw
emb=[n for n in g.node if n.op_type=="Gather"]
print("num Gather:",len(emb))
for n in emb[:6]:
    out=n.output[0]
    e=act.get(out)
    print("  Gather out",out,"bw",e["bw"] if e else "none")
# softmax / layernorm outputs
for opt in ["Softmax","LayerNormalization","Add","Mul"]:
    nodes=[n for n in g.node if n.op_type==opt]
    bws=[act[n.output[0]]["bw"] for n in nodes if n.output[0] in act]
    from collections import Counter
    print(opt,"count",len(nodes),"out-bw hist",dict(Counter(bws)))
