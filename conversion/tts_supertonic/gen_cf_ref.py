import os,numpy as np,onnxruntime as ort,wave
W=os.path.expanduser('~/stwork/'); O=W+'onnx/'
so=ort.SessionOptions(); so.log_severity_level=3
def S(p): return ort.InferenceSession(p,so)
def corr(a,b): return float(np.corrcoef(a.ravel(),b.ravel())[0,1])
d=np.load(W+'raw/canon_inputs.npy',allow_pickle=True).item()
ie_te,ie_dp,sttl,sdp,tm,Tr=d['ie_te'],d['ie_dp'],d['sttl'],d['sdp'],d['tmask'],d['Tr']
# verify _cf text_encoder vs _srg(exact)
te_cf=S(O+'text_encoder_cf.onnx').run(None,{'inputs_embeds':ie_te,'style_ttl':sttl,'text_mask':tm})[0]
te_o =S(O+'text_encoder_srg.onnx').run(None,{'inputs_embeds':ie_te,'style_ttl':sttl,'text_mask':tm})[0]
print('_cf text_encoder vs exact corr=%.6f (valid %.6f)'%(corr(te_cf,te_o),corr(te_cf[0,:,:Tr],te_o[0,:,:Tr])))
SR=44100; chunk=512*6
te=S(O+'text_encoder_cf.onnx'); ve=S(O+'vector_estimator_cf.onnx'); vo=S(O+'vocoder_cf.onnx'); dp=S(O+'duration_predictor_cf.onnx')
dd=float(dp.run(None,{'inputs_embeds':ie_dp,'style_dp':sdp,'text_mask':tm})[0][0])/1.05
temb=te.run(None,{'inputs_embeds':ie_te,'style_ttl':sttl,'text_mask':tm})[0]
Lr=min(int((dd*SR+chunk-1)//chunk),64)
rng=np.random.RandomState(1000); nl=rng.randn(1,144,Lr).astype(np.float32)
lm=np.zeros((1,1,64),np.float32); lm[0,0,:Lr]=1
xt=np.pad(nl,[(0,0),(0,0),(0,64-Lr)]).astype(np.float32)*lm
tot=np.array([8],np.float32)
for s in range(8):
    xt=ve.run(None,{'noisy_latent':xt,'text_emb':temb,'style_ttl':sttl,'text_mask':tm,'latent_mask':lm,'current_step':np.array([s],np.float32),'total_step':tot})[0]
wav=vo.run(None,{'latent':xt})[0]; nsamp=int(dd*SR)
x=np.clip(wav[0,:nsamp],-1,1); pcm=(x*32767).astype('<i2')
with wave.open(W+'raw/cpu_cf_M1.wav','wb') as w: w.setnchannels(1);w.setsampwidth(2);w.setframerate(SR);w.writeframes(pcm.tobytes())
print('cpu_cf dur=%.3f Lr=%d (cpu_padded dur was 2.478)'%(dd,Lr))
