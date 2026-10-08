import os,sys,subprocess,numpy as np,wave,shutil,glob
W=os.path.expanduser('~/stwork/'); DLC=W+'dlc/'
B=os.path.expanduser('~/qairt/2.37.1.250807/')
BACKEND=B+'lib/x86_64-linux-clang/libQnnCpu.so'; MODEL=B+'lib/x86_64-linux-clang/libQnnModelDlc.so'
NETRUN=B+'bin/x86_64-linux-clang/qnn-net-run'
TMP=W+'simtmp/'; os.makedirs(TMP,exist_ok=True)
env=dict(os.environ); env['LD_LIBRARY_PATH']=B+'lib/x86_64-linux-clang:'+env.get('LD_LIBRARY_PATH','')
def runqnn(dlcname,inputs):
    d=TMP+dlcname; 
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)
    line=[]
    for k,v in inputs.items():
        p=d+'/%s.raw'%k; v.astype(np.float32).tofile(p); line.append('%s:=%s'%(k,p))
    open(d+'/il.txt','w').write(' '.join(line)+'\n')
    r=subprocess.run([NETRUN,'--backend',BACKEND,'--model',MODEL,'--dlc_path',DLC+dlcname+'.dlc','--input_list',d+'/il.txt','--output_dir',d+'/out'],env=env,capture_output=True,text=True)
    outs=sorted(glob.glob(d+'/out/Result_0/*.raw'))
    if not outs:
        print('FAIL',dlcname,r.stdout[-500:],r.stderr[-300:]); sys.exit(1)
    return outs
d=np.load(W+'raw/canon_inputs.npy',allow_pickle=True).item()
ie_te,ie_dp,sttl,sdp,tm=d['ie_te'],d['ie_dp'],d['sttl'],d['sdp'],d['tmask']
SR=44100; chunk=512*6
# DP
dur_raw=runqnn('dp_w8a16',{'inputs_embeds':ie_dp,'style_dp':sdp,'text_mask':tm})[0]
dd=float(np.fromfile(dur_raw,np.float32)[0])
# TE
te_raw=runqnn('te_w8a16',{'inputs_embeds':ie_te,'style_ttl':sttl,'text_mask':tm})[0]
temb=np.fromfile(te_raw,np.float32).reshape(1,256,64)
Lr=min(int((dd*SR+chunk-1)//chunk),64)
rng=np.random.RandomState(1000); nl=rng.randn(1,144,Lr).astype(np.float32)
lm=np.zeros((1,1,64),np.float32); lm[0,0,:Lr]=1
xt=np.pad(nl,[(0,0),(0,0),(0,64-Lr)]).astype(np.float32)*lm
tot=np.array([8],np.float32)
for s in range(8):
    o=runqnn('ve_w8a16',{'noisy_latent':xt,'text_emb':temb,'style_ttl':sttl,'latent_mask':lm,'text_mask':tm,'current_step':np.array([s],np.float32),'total_step':tot})[0]
    xt=np.fromfile(o,np.float32).reshape(1,144,64)
wav_raw=runqnn('vo_w8a16',{'latent':xt})[0]
wav=np.fromfile(wav_raw,np.float32)
nsamp=int(dd*SR); wav=wav[:nsamp]
x=np.clip(wav,-1,1); pcm=(x*32767).astype('<i2')
with wave.open(W+'raw/sim_cpu_w8a16_M1.wav','wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
def rd(p):
    with wave.open(p,'rb') as r: return np.frombuffer(r.readframes(r.getnframes()),'<i2').astype(np.float32)/32767
ref=rd(W+'raw/cpu_padded_M1.wav'); a=rd(W+'raw/sim_cpu_w8a16_M1.wav'); n=min(len(a),len(ref))
print('DP dur sim=%.3f (fp32 ref ~2.48)'%dd)
print('wav corr (w8a16 CPUsim vs cpu_padded)=%.4f  lens %d/%d'%(float(np.corrcoef(a[:n],ref[:n])[0,1]),len(a),len(ref)))
print('wav RMS sim=%.4f ref=%.4f'%(np.sqrt((a**2).mean()),np.sqrt((ref**2).mean())))
