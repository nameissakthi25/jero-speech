import onnx, json
m = onnx.load("aimet_out/intent.onnx")
g = m.graph
exp = [n for n in g.node if n.op_type == "Expand"]
print("num Expand nodes:", len(exp))
prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n
init_names = {i.name for i in g.initializer}
enc = json.load(open("aimet_out/intent.encodings"))
act = {r["name"]: r for r in enc["activation_encodings"]}
par = {r["name"]: r for r in enc["param_encodings"]}

def tag(name):
    if name in act:
        return "ACT bw%d" % act[name]["bw"]
    if name in par:
        return "PAR bw%d" % par[name]["bw"]
    if name in init_names:
        return "INIT"
    return "none"

for n in exp:
    print("----", n.name)
    for i, inp in enumerate(n.input):
        p = prod.get(inp)
        pt = p.op_type if p else ("INIT" if inp in init_names else "GRAPH_IN")
        print("  in[%d] %s  producer=%s  enc=%s" % (i, inp, pt, tag(inp)))
    for o in n.output:
        cons = [c.op_type for c in g.node if o in c.input]
        print("  out   %s  enc=%s  consumers=%s" % (o, tag(o), cons))
