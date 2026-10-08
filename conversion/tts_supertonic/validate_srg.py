import os,numpy as np,onnxruntime as ort
C=os.path.expanduser('~/.cache/supertonic3/onnx/'); O=os.path.expanduser('~/stwork/onnx/'); W=os.path.expanduser('~/stwork/')
d=np.load(W+'raw/canon_inputs.npy',allow_pickle=True).item()
te_emb=np.load(W+'te_emb.npy'); dp_emb=np.load(W+'dp_emb.npy')
ie_te=d['ie_te']; ie_dp=d['ie_dp']; sttl=d['sttl']; sdp=d['sdp']; tm=d['tmask']; Tr=d['Tr']
def corr(a,b): return np.corrcoef(a.ravel(),b.ravel())[0,1]
def mx(a,b): return float(np.max(np.abs(a-b)))
so=ort.SessionOptions(); so.log_severity_level=3
# reconstruct text_ids from inputs_embeds not needed; run original with text_ids via padded ids
# we have ie; to run original te we need text_ids -> rebuild from nearest? Instead reuse capture: original te takes text_ids. Load padded ids by re-embedding match: argmin. Simpler: run original on padded text_ids stored? Not saved. Rebuild via matching rows.
# Build padded text_ids by matching ie_te rows to te_emb table (exact rows).
def ids_from_ie(ie,tbl):
    T=ie.shape[1]; ids=np.zeros((1,T),np.int64)
    for t in range(T):
        ids[0,t]=int(np.argmin(np.abs(tbl-ie[0,t]).sum(1)))
    return ids
tid=ids_from_ie(ie_te,te_emb)
# TEXT ENCODER
o_orig=ort.InferenceSession(C+'text_encoder.onnx',so).run(None,{'text_ids':tid,'style_ttl':sttl,'text_mask':tm})[0]
o_srg =ort.InferenceSession(O+'text_encoder_srg.onnx',so).run(None,{'inputs_embeds':ie_te,'style_ttl':sttl,'text_mask':tm})[0]
print('text_encoder  corr=%.6f maxabs=%.3e  (valid region corr=%.6f)'%(corr(o_orig,o_srg),mx(o_orig,o_srg),corr(o_orig[0,:,:Tr],o_srg[0,:,:Tr])))
# DURATION
d_orig=ort.InferenceSession(C+'duration_predictor.onnx',so).run(None,{'text_ids':tid,'style_dp':sdp,'text_mask':tm})[0]
d_srg =ort.InferenceSession(O+'duration_predictor_srg.onnx',so).run(None,{'inputs_embeds':ie_dp,'style_dp':sdp,'text_mask':tm})[0]
print('duration      orig=%.5f srg=%.5f  abs_diff=%.3e'%(float(d_orig[0]),float(d_srg[0]),abs(float(d_orig[0])-float(d_srg[0]))))
# VECTOR ESTIMATOR (padded calib sample 000_0)
CAL=W+'calib/'
def ld(n,shape): return np.fromfile(CAL+n+'.raw',np.float32).reshape(shape)
ve_in={'noisy_latent':ld('ve_noisy_000_0',(1,144,64)),'text_emb':ld('ve_temb_000_0',(1,256,64)),'style_ttl':ld('ve_sttl_000_0',(1,50,256)),'latent_mask':ld('ve_lmask_000_0',(1,1,64)),'text_mask':ld('ve_tmask_000_0',(1,1,64)),'current_step':ld('ve_cs_000_0',(1,)),'total_step':ld('ve_ts_000_0',(1,))}
v_orig=ort.InferenceSession(C+'vector_estimator.onnx',so).run(None,ve_in)[0]
v_srg =ort.InferenceSession(O+'vector_estimator_srg.onnx',so).run(None,ve_in)[0]
Lr=int(ve_in['latent_mask'].sum())
print('vector_est    corr=%.6f maxabs=%.3e (valid L=%d corr=%.6f)'%(corr(v_orig,v_srg),mx(v_orig,v_srg),Lr,corr(v_orig[0,:,:Lr],v_srg[0,:,:Lr])))
