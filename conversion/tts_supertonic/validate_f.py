import os,numpy as np,onnxruntime as ort,wave
W=os.path.expanduser('~/stwork/'); O=W+'onnx/'; C=os.path.expanduser('~/.cache/supertonic3/onnx/')
so=ort.SessionOptions(); so.log_severity_level=3
def S(p): return ort.InferenceSession(p,so)
def corr(a,b): return float(np.corrcoef(a.ravel(),b.ravel())[0,1])
d=np.load(W+'raw/canon_inputs.npy',allow_pickle=True).item()
ie_te,ie_dp,sttl,sdp,tm,Tr,dur=d['ie_te'],d['ie_dp'],d['sttl'],d['sdp'],d['tmask'],d['Tr'],d['dur']
# per-model corr (f static vs original)
te_f=S(O+'text_encoder_f.onnx').run(None,{'inputs_embeds':ie_te,'style_ttl':sttl,'text_mask':tm})[0]
te_o=S(O+'text_encoder_srg.onnx').run(None,{'inputs_embeds':ie_te,'style_ttl':sttl,'text_mask':tm})[0]
print('text_encoder  f-vs-fp32 corr=%.6f (valid %.6f)'%(corr(te_f,te_o),corr(te_f[0,:,:Tr],te_o[0,:,:Tr])))
dp_f=S(O+'duration_predictor_f.onnx').run(None,{'inputs_embeds':ie_dp,'style_dp':sdp,'text_mask':tm})[0]
dp_o=S(O+'duration_predictor_srg.onnx').run(None,{'inputs_embeds':ie_dp,'style_dp':sdp,'text_mask':tm})[0]
print('duration      f=%.5f fp32=%.5f'%(float(dp_f[0]),float(dp_o[0])))
# full fp32 static pipeline with _f models
SR=44100; chunk=512*6
te=S(O+'text_encoder_f.onnx'); ve=S(O+'vector_estimator_f.onnx'); vo=S(O+'vocoder_f.onnx'); dpS=S(O+'duration_predictor_f.onnx')
dd=float(dpS.run(None,{'inputs_embeds':ie_dp,'style_dp':sdp,'text_mask':tm})[0][0])
temb=te.run(None,{'inputs_embeds':ie_te,'style_ttl':sttl,'text_mask':tm})[0]
Lr=int((dd*SR+chunk-1)//chunk); Lr=min(Lr,64)
rng=np.random.RandomState(1000)  # match capture seed for line 0
nl=rng.randn(1,144,Lr).astype(np.float32)
lm=np.zeros((1,1,64),np.float32); lm[0,0,:Lr]=1
xt=np.pad(nl,[(0,0),(0,0),(0,64-Lr)]).astype(np.float32)*lm
tot=np.array([8],np.float32)
for s in range(8):
    xt=ve.run(None,{'noisy_latent':xt,'text_emb':temb,'style_ttl':sttl,'text_mask':tm,'latent_mask':lm,'current_step':np.array([s],np.float32),'total_step':tot})[0]
wav=vo.run(None,{'latent':xt})[0]
nsamp=int(dd*SR)
x=np.clip(wav[0,:nsamp],-1,1); pcm=(x*32767).astype('<i2')
with wave.open(W+'raw/cpu_fp32static_M1.wav','wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
# compare to cpu_padded_M1.wav
def rd(p):
    with wave.open(p,'rb') as r: return np.frombuffer(r.readframes(r.getnframes()),'<i2').astype(np.float32)/32767
a=rd(W+'raw/cpu_fp32static_M1.wav'); b=rd(W+'raw/cpu_padded_M1.wav'); n=min(len(a),len(b))
print('fp32-static wav vs cpu_padded: len',len(a),len(b),'corr=%.4f'%corr(a[:n],b[:n]))
print('dur used=%.3f Lr=%d'%(dd,Lr))
