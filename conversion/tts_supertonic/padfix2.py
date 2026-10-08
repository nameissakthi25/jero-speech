import onnx,os
from onnx import helper
O=os.path.expanduser('~/stwork/onnx/')
def toposort(g):
    produced=set(i.name for i in g.input)|set(t.name for t in g.initializer)
    nodes=list(g.node); ordered=[]
    # Constant/no-input nodes first
    while nodes:
        prog=[];rest=[]
        for n in nodes:
            if all((i=='' or i in produced) for i in n.input): prog.append(n)
            else: rest.append(n)
        if not prog:
            ordered+=rest; break
        for n in prog: produced.update(n.output)
        ordered+=prog; nodes=rest
    del g.node[:]; g.node.extend(ordered)
def fix(nm):
    m=onnx.load(O+nm+'.onnx'); g=m.graph
    inits={t.name for t in g.initializer}|{n.output[0] for n in g.node if n.op_type=='Constant'}
    cnt=0
    for n in g.node:
        if n.op_type=='Pad' and len(n.input)>0 and n.input[0] in inits:
            dst=n.input[0]+'__idbuf'
            g.node.append(helper.make_node('Identity',[n.input[0]],[dst],name='IdPad%d'%cnt))
            n.input[0]=dst; cnt+=1
    toposort(g)
    m=onnx.shape_inference.infer_shapes(m)
    out=O+nm+'_id.onnx'; onnx.save(m,out); return cnt,out
for nm in ['duration_predictor_cf','text_encoder_cf','vector_estimator_cf','vocoder_cf']:
    c,o=fix(nm); print('%-24s identities=%d'%(nm,c))
