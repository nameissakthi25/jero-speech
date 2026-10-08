import onnx,os,onnxsim
O=os.path.expanduser('~/stwork/onnx/')
shapes={
 'text_encoder_srg':{'inputs_embeds':[1,64,256],'style_ttl':[1,50,256],'text_mask':[1,1,64]},
 'duration_predictor_srg':{'inputs_embeds':[1,64,64],'style_dp':[1,8,16],'text_mask':[1,1,64]},
 'vector_estimator_srg':{'noisy_latent':[1,144,64],'text_emb':[1,256,64],'style_ttl':[1,50,256],'latent_mask':[1,1,64],'text_mask':[1,1,64],'current_step':[1],'total_step':[1]},
 'vocoder_srg':{'latent':[1,144,64]},
}
for nm,sh in shapes.items():
    m=onnx.load(O+nm+'.onnx')
    ms,ok=onnxsim.simplify(m,overwrite_input_shapes=sh)
    assert ok,'simplify check failed '+nm
    out=O+nm.replace('_srg','_st')+'.onnx'
    onnx.save(ms,out)
    npad=sum(1 for n in ms.graph.node if n.op_type=='Pad')
    print('%-24s -> %s  Pads=%d  %dKB'%(nm,os.path.basename(out),npad,os.path.getsize(out)//1024))
