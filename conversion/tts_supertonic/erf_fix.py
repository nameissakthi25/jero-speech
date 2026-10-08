import onnx,os,numpy as np
from onnx import helper,TensorProto,numpy_helper
O=os.path.expanduser('~/stwork/onnx/')
def C(name,val):
    return helper.make_node('Constant',[],[name],value=numpy_helper.from_array(np.array(val,np.float32),name+'_v'))
def fix(nm):
    m=onnx.load(O+nm+'.onnx'); g=m.graph
    erfs=[n for n in g.node if n.op_type=='Erf']
    newnodes=[]
    for i,n in enumerate(erfs):
        u=n.input[0]; out=n.output[0]; p='erf%d_'%i
        c1=C(p+'c1',1.1283792); c2=C(p+'c2',0.1009)
        u2=helper.make_node('Mul',[u,u],[p+'u2']); u3=helper.make_node('Mul',[p+'u2',u],[p+'u3'])
        t1=helper.make_node('Mul',[u,p+'c1'],[p+'t1']); t2=helper.make_node('Mul',[p+'u3',p+'c2'],[p+'t2'])
        s=helper.make_node('Add',[p+'t1',p+'t2'],[p+'s']); th=helper.make_node('Tanh',[p+'s'],[out])
        g.node.remove(n)
        newnodes+=[c1,c2,u2,u3,t1,t2,s,th]
    g.node.extend(newnodes)
    m=onnx.shape_inference.infer_shapes(m)
    # topo: onnx requires defs before use; do a simple kahn sort
    nodes=list(g.node); produced=set(i.name for i in g.input)|set(t.name for t in g.initializer)
    ordered=[]; pend=nodes
    while pend:
        prog=[]; rest=[]
        for nd in pend:
            if all((inp=='' or inp in produced) for inp in nd.input): prog.append(nd)
            else: rest.append(nd)
        if not prog: ordered+=rest; break
        for nd in prog: produced.update(nd.output)
        ordered+=prog; pend=rest
    del g.node[:]; g.node.extend(ordered)
    out=O+nm.replace('_st','_f')+'.onnx'; onnx.save(m,out)
    print('%-26s Erf replaced=%d -> %s'%(nm,len(erfs),os.path.basename(out)))
for nm in ['text_encoder_st','duration_predictor_st','vector_estimator_st','vocoder_st']:
    fix(nm)
