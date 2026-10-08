import onnx,numpy as np,os,shutil
from onnx import helper,TensorProto,numpy_helper
C=os.path.expanduser('~/.cache/supertonic3/onnx/'); O=os.path.expanduser('~/stwork/onnx/')
os.makedirs(O,exist_ok=True)
def embed_surgery(src,dst,gather_sub,emb_dim):
    m=onnx.load(C+src); g=m.graph
    GE=[n for n in g.node if n.op_type=='Gather' and gather_sub in n.name]
    assert len(GE)==1,[n.name for n in g.node if n.op_type=='Gather' and 'char_embedder' in n.name]
    gath=GE[0]; out=gath.output[0]
    g.node.remove(gath)
    for n in g.node:
        n.input[:]=['inputs_embeds' if i==out else i for i in n.input]
    g.input.extend([helper.make_tensor_value_info('inputs_embeds',TensorProto.FLOAT,[1,'text_length',emb_dim])])
    used=set(o for n in g.node for o in n.input)|set(o.name for o in g.output)
    keep=[i for i in g.input if i.name in used]; del g.input[:]; g.input.extend(keep)
    ki=[t for t in g.initializer if t.name in used]; del g.initializer[:]; g.initializer.extend(ki)
    onnx.save(m,O+dst)
    print(dst,'in:',[i.name for i in g.input],'out:',[o.name for o in g.output])
embed_surgery('text_encoder.onnx','text_encoder_srg.onnx','text_embedder/char_embedder/Gather',256)
embed_surgery('duration_predictor.onnx','duration_predictor_srg.onnx','text_embedder/char_embedder/Gather',64)
m=onnx.load(C+'vector_estimator.onnx'); g=m.graph; nfix=0; NEG=-1.0e4
for t in g.initializer:
    a=numpy_helper.to_array(t)
    if a.dtype.kind=='f' and a.size>0 and np.nanmin(a)<=-1e30:
        t.CopyFrom(numpy_helper.from_array(np.where(a<=-1e30,np.float32(NEG),a).astype(a.dtype),t.name)); nfix+=1
for n in g.node:
    if n.op_type=='Constant':
        for att in n.attribute:
            if att.name=='value':
                a=numpy_helper.to_array(att.t)
                if a.dtype.kind=='f' and a.size>0 and np.nanmin(a)<=-1e30:
                    att.t.CopyFrom(numpy_helper.from_array(np.where(a<=-1e30,np.float32(NEG),a).astype(a.dtype))); nfix+=1
onnx.save(m,O+'vector_estimator_srg.onnx'); print('vector_estimator: fixed',nfix,'neg-inf tensors -> -1e4')
shutil.copy(C+'vocoder.onnx',O+'vocoder_srg.onnx'); print('vocoder copied')
