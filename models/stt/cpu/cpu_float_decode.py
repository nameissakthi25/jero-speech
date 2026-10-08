import os,sys,glob,time,numpy as np,onnxruntime as ort
MD="onnx_test/model"
enc=ort.InferenceSession(f"{MD}/encoder.onnx",providers=["CPUExecutionProvider"])
dec=ort.InferenceSession(f"{MD}/decoder.onnx",providers=["CPUExecutionProvider"])
joi=ort.InferenceSession(f"{MD}/joiner.onnx",providers=["CPUExecutionProvider"])
enc_in=[i.name for i in enc.get_inputs()]; state_in=[n for n in enc_in if n!="x"]
T_WIN,CHUNK,CTX,BLANK=39,32,2,0
def shp(ix):
    return [1 if (isinstance(d,str) or d is None) else d for d in ix.shape]
def init_states():
    st={}
    for i in enc.get_inputs():
        if i.name=="x": continue
        st[i.name]=np.zeros(shp(i),np.float32 if i.type=="tensor(float)" else np.int64)
    return st
def load_tokens(p):
    id2t={}
    for l in open(p,encoding="utf-8"):
        a=l.split()
        if len(a)>=2: id2t[int(a[1])]=a[0]
    return id2t
def ids2txt(ids,id2t): return "".join(id2t.get(i,"") for i in ids).replace("▁"," ").strip()
def decode(feats,id2t):
    nf=feats.shape[0]; npad=(-(nf-T_WIN))%CHUNK
    if nf<T_WIN: npad=T_WIN-nf
    if npad: feats=np.concatenate([feats,np.full((npad,80),feats.min(),np.float32)],0); nf=feats.shape[0]
    st=init_states(); hyp=[BLANK]*CTX
    do=dec.run(None,{"y":np.array([hyp[-CTX:]],np.int64)})[0]; emitted=[]
    i=0
    while i+T_WIN<=nf:
        x=feats[i:i+T_WIN][None,:,:].astype(np.float32)
        out=enc.run(None,{"x":x,**st})
        onames=[o.name for o in enc.get_outputs()]; od=dict(zip(onames,out))
        eo=od["encoder_out"]
        st={s:od["new_"+s] for s in state_in}
        for t in range(eo.shape[1]):
            lg=joi.run(None,{"encoder_out":eo[:,t,:],"decoder_out":do})[0]
            k=int(np.argmax(lg[0]))
            if k!=BLANK: hyp.append(k); emitted.append(k); do=dec.run(None,{"y":np.array([hyp[-CTX:]],np.int64)})[0]
        i+=CHUNK
    return emitted
id2t=load_tokens(f"{MD}/tokens.txt")
rows=[l.split("\t") for l in open(sys.argv[1]).read().strip().split("\n")]
out=open(sys.argv[2],"w"); ta=0;t0=time.time()
for uid,_ in rows:
    f=f"evaluation/utterances/{uid}.npy"; feats=np.load(f).astype(np.float32); ta+=len(feats)/100.0
    txt=ids2txt(decode(feats,id2t),id2t); out.write(f"{uid}\t{txt}\n"); print(f"{uid}\t{txt}",flush=True)
wall=time.time()-t0
print(f"CPU_FLOAT_DONE utts={len(rows)} audio={ta:.1f}s wall={wall:.1f}s RTF={wall/ta:.3f}")
