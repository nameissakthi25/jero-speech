import onnx,os,sys
from onnx import helper
O=os.path.expanduser('~/stwork/onnx/')
def fix(nm):
    m=onnx.load(O+nm+'.onnx'); g=m.graph
    inits={t.name:t for t in g.initializer}
    # initializers consumed as data(input[0]) of a Pad -> make Constant nodes
    targets=set()
    for n in g.node:
        if n.op_type=='Pad' and len(n.input)>0 and n.input[0] in inits:
            targets.add(n.input[0])
    if not targets:
        onnx.save(m,O+nm.replace('_opt','_cv')+'.onnx'); return 0
    newnodes=[]
    for name in targets:
        t=inits[name]
        cst=helper.make_node('Constant',[],[name],name='Const_'+name.replace('.','_'),value=t)
        newnodes.append(cst)
        g.initializer.remove(t)
    g.node.extend(newnodes)  # Constants with no inputs can go anywhere; topo-sort by onnx
    m=onnx.shape_inference.infer_shapes(m)
    # topological sort: onnx.utils doesn't sort; use a simple reorder putting Constants first
    consts=[n for n in g.node if n.op_type=='Constant']
    others=[n for n in g.node if n.op_type!='Constant']
    del g.node[:]; g.node.extend(consts+others)
    onnx.save(m,O+nm.replace('_opt','_cv')+'.onnx')
    return len(targets)
for nm in ['duration_predictor_opt','text_encoder_opt','vector_estimator_opt','vocoder_opt']:
    try: print(nm,'-> fixed',fix(nm),'pad-initializers')
    except Exception as e: print(nm,'ERR',e)
