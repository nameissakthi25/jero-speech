import onnx,os,numpy as np,onnxruntime as ort
from onnx import helper,TensorProto,numpy_helper
O=os.path.expanduser('~/stwork/onnx/')
so=ort.SessionOptions(); so.log_severity_level=3
def erf_swap(g):
    erfs=[n for n in g.node if n.op_type=='Erf']; nn=[]
    for i,n in enumerate(erfs):
        u=n.input[0]; out=n.output[0]; p='erf%d_'%i
        def Cst(nm,v): return helper.make_node('Constant',[],[nm],value=numpy_helper.from_array(np.array(v,np.float32),nm+'_v'))
        nn+=[Cst(p+'c1',1.1283792),Cst(p+'c2',0.1009),
             helper.make_node('Mul',[u,u],[p+'u2']),helper.make_node('Mul',[p+'u2',u],[p+'u3']),
             helper.make_node('Mul',[u,p+'c1'],[p+'t1']),helper.make_node('Mul',[p+'u3',p+'c2'],[p+'t2']),
             helper.make_node('Add',[p+'t1',p+'t2'],[p+'s']),helper.make_node('Tanh',[p+'s'],[out])]
        g.node.remove(n)
    g.node.extend(nn); return len(erfs)
def const_fold(m):
    g=m.graph
    const={t.name:numpy_helper.to_array(t) for t in g.initializer}
    for n in g.node:
        if n.op_type=='Constant':
            const[n.output[0]]=numpy_helper.to_array(n.attribute[0].t)
    changed=True; folded=0
    while changed:
        changed=False
        for n in list(g.node):
            if n.op_type in ('Constant',): continue
            if not n.input: continue
            if all((i=='' or i in const) for i in n.input) and not all(o in const for o in n.output):
                # build single-node model and run
                try:
                    ins=[helper.make_tensor_value_info(i,onnx.TensorProto.FLOAT,None) for i in n.input if i]
                    inits=[numpy_helper.from_array(const[i].astype(const[i].dtype),i) for i in n.input if i]
                    outs=[helper.make_tensor_value_info(o,onnx.TensorProto.FLOAT,None) for o in n.output]
                    sub=helper.make_graph([n],'s',[],outs,initializer=inits)
                    mm=helper.make_model(sub,opset_imports=m.opset_import)
                    r=ort.InferenceSession(mm.SerializeToString(),so).run(None,{})
                    for o,val in zip(n.output,r): const[o]=val
                    folded+=1; changed=True
                except Exception as e:
                    const.setdefault('__fail_'+n.name,True)
    # now rebuild graph: drop folded nodes, add their terminal outputs as initializers where consumed
    keepnodes=[]; produced_by_node=set()
    for n in g.node:
        if n.op_type=='Constant': 
            keepnodes.append(n); continue
        if all(o in const for o in n.output) and n.input and all((i=='' or i in const) for i in n.input):
            continue # folded away
        keepnodes.append(n)
        produced_by_node.update(n.output)
    # determine which const tensors are needed as inputs to kept nodes but not already initializers/constants
    init_names=set(t.name for t in g.initializer)|set(n.output[0] for n in g.node if n.op_type=='Constant')
    needed=set()
    for n in keepnodes:
        for i in n.input:
            if i in const and i not in init_names and i not in produced_by_node:
                needed.add(i)
    del g.node[:]; g.node.extend(keepnodes)
    for nm in needed:
        g.initializer.append(numpy_helper.from_array(const[nm].astype(const[nm].dtype),nm))
    return folded,len(needed)
for nm in ['text_encoder_srg','duration_predictor_srg','vector_estimator_srg','vocoder_srg']:
    m=onnx.load(O+nm+'.onnx'); ne=erf_swap(m.graph)
    m=onnx.shape_inference.infer_shapes(m)
    nf,nn=const_fold(m)
    m=onnx.shape_inference.infer_shapes(m)
    out=O+nm.replace('_srg','_cf')+'.onnx'; onnx.save(m,out)
    print('%-24s erf=%d folded=%d new_init=%d -> %s'%(nm,ne,nf,nn,os.path.basename(out)))
