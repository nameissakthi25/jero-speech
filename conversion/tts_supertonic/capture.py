import os,sys,json,numpy as np
sys.path.insert(0,"/home/raemox/supertonic/.venv/lib/python3.10/site-packages")
from supertonic import TTS
import wave
W=os.path.expanduser("~/stwork"); R=W+"/raw"; CAL=W+"/calib"
T_BUCKET=64; L_BUCKET=64
os.makedirs(R,exist_ok=True); os.makedirs(CAL,exist_ok=True)
tts=TTS(auto_download=False); core=tts.model; tp=core.text_processor
te_emb=np.load(W+"/te_emb.npy"); dp_emb=np.load(W+"/dp_emb.npy")
style=tts.get_voice_style("M1")
SR=core.sample_rate; chunk=core.base_chunk_size*core.chunk_compress_factor
def wwrite(path,sr,x):
    x=np.clip(x,-1,1); pcm=(x*32767).astype("<i2")
    with wave.open(path,"wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
def embed(tid,tbl): return tbl[tid[0]][None].astype(np.float32)  # [1,T,E]
def pad2(a,axis,n):
    pad=[(0,0)]*a.ndim; pad[axis]=(0,n-a.shape[axis]); return np.pad(a,pad)
lines=["Hi, I am Jero! Lets play.","Hello there friend.","I am ready now.","Lets go outside.","That is so cool!","Can you help me?","Good morning everyone.","I love robots.","Watch me jump high.","Time for a story.","Where did it go?","You are my friend.","Lets count to ten.","The sky is blue.","I feel very happy.","Please follow me.","What is your name?","See you tomorrow.","That was amazing!","Do not be afraid.","Lets build a tower.","I found a treasure.","Come play with me.","It is getting dark.","Sing a happy song.","Run fast now.","Look at the stars.","I am a robot.","Give me five.","Lets explore."]
def raw(name,arr): arr.astype(np.float32).tofile(CAL+"/"+name+".raw")
te_list=[]; dp_list=[]; ve_list=[]; vo_list=[]
corrs=[]
for li,ln in enumerate(lines):
    tid,tmask=tp([ln],"en"); Tr=tid.shape[1]
    assert Tr<=T_BUCKET, (ln,Tr)
    # ---- reference (unpadded) ----
    dur_ref,*_=core.dp_ort.run(None,{"text_ids":tid,"style_dp":style.dp,"text_mask":tmask}); dur_ref=dur_ref/1.05
    te_ref,*_=core.text_enc_ort.run(None,{"text_ids":tid,"style_ttl":style.ttl,"text_mask":tmask})
    # ---- padded bucket mode ----
    tidP=pad2(tid,1,T_BUCKET); tmP=pad2(tmask,2,T_BUCKET)
    dur_p,*_=core.dp_ort.run(None,{"text_ids":tidP,"style_dp":style.dp,"text_mask":tmP}); dur_p=dur_p/1.05
    teP,*_=core.text_enc_ort.run(None,{"text_ids":tidP,"style_ttl":style.ttl,"text_mask":tmP})
    # compare valid region
    c_te=np.corrcoef(te_ref[0,:,:Tr].ravel(),teP[0,:,:Tr].ravel())[0,1]
    corrs.append((c_te,float(dur_ref[0]),float(dur_p[0])))
    # host embeds for surgered-model calib
    ieT=pad2(embed(tidP,te_emb),1,T_BUCKET)        # [1,64,256]
    ieD=pad2(embed(tidP,dp_emb),1,T_BUCKET)        # [1,64,64]
    # latent setup from padded duration
    Lr=int((dur_p.max()*SR+chunk-1)//chunk); Lr=min(Lr,L_BUCKET)
    rng=np.random.RandomState(1000+li)
    nl=rng.randn(1,144,Lr).astype(np.float32)
    lm=np.zeros((1,1,L_BUCKET),np.float32); lm[0,0,:Lr]=1
    nlP=pad2(nl,2,L_BUCKET)*lm
    xt=nlP.copy(); tot=np.array([8],np.float32)
    # estimator loop padded
    for s in range(8):
        cs=np.array([s],np.float32)
        if li<24: # dump a subset of steps for calib (every line, steps 0,3,7)
            if s in (0,3,7):
                tag="%03d_%d"%(li,s)
                raw("ve_noisy_"+tag,xt); raw("ve_temb_"+tag,teP); raw("ve_sttl_"+tag,style.ttl)
                raw("ve_lmask_"+tag,lm); raw("ve_tmask_"+tag,tmP); raw("ve_cs_"+tag,cs); raw("ve_ts_"+tag,tot)
                ve_list.append(tag)
        xt,*_=core.vector_est_ort.run(None,{"noisy_latent":xt,"text_emb":teP,"style_ttl":style.ttl,"text_mask":tmP,"latent_mask":lm,"current_step":cs,"total_step":tot})
    wavP,*_=core.vocoder_ort.run(None,{"latent":xt})
    # calib dumps (per line)
    tag="%03d"%li
    raw("te_ie_"+tag,ieT); raw("te_sttl_"+tag,style.ttl); raw("te_tmask_"+tag,tmP); te_list.append(tag)
    raw("dp_ie_"+tag,ieD); raw("dp_sdp_"+tag,style.dp); raw("dp_tmask_"+tag,tmP); dp_list.append(tag)
    raw("vo_latent_"+tag,xt); vo_list.append(tag)
    if li==0:
        # save canonical padded-pipeline wav + unpadded ref for fair device compare
        nsamp=int(dur_p[0]*SR)
        wwrite(R+"/cpu_padded_M1.wav",SR,wavP[0,:nsamp].astype(np.float32))
        np.save(R+"/canon_inputs.npy",{"ie_te":ieT,"ie_dp":ieD,"sttl":style.ttl,"sdp":style.dp,"tmask":tmP,"Tr":Tr,"dur":float(dur_p[0]),"text":ln},allow_pickle=True)
print("padded-vs-ref text_emb corr: min %.4f mean %.4f"%(min(c[0] for c in corrs),np.mean([c[0] for c in corrs])))
print("dur ref vs padded (first 5):",[ (round(c[1],3),round(c[2],3)) for c in corrs[:5]])
# write input_list.txt for each model
def il(path,tags,fields):
    with open(path,"w") as f:
        for t in tags:
            f.write(" ".join("%s:=%s/%s_%s.raw"%(n,CAL,pfx,t) for n,pfx in fields)+"\n")
il(W+"/calib_te.txt",te_list,[("inputs_embeds","te_ie"),("style_ttl","te_sttl"),("text_mask","te_tmask")])
il(W+"/calib_dp.txt",dp_list,[("inputs_embeds","dp_ie"),("style_dp","dp_sdp"),("text_mask","dp_tmask")])
il(W+"/calib_ve.txt",ve_list,[("noisy_latent","ve_noisy"),("text_emb","ve_temb"),("style_ttl","ve_sttl"),("latent_mask","ve_lmask"),("text_mask","ve_tmask"),("current_step","ve_cs"),("total_step","ve_ts")])
il(W+"/calib_vo.txt",vo_list,[("latent","vo_latent")])
print("calib sizes: te",len(te_list),"dp",len(dp_list),"ve",len(ve_list),"vo",len(vo_list))
print("saved cpu_padded_M1.wav and canon_inputs.npy")
