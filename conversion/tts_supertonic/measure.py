import os,sys,glob,json,numpy as np, onnx
from onnx import numpy_helper
sys.path.insert(0,"/home/raemox/supertonic/.venv/lib/python3.10/site-packages")
from supertonic import TTS
tts=TTS(auto_download=False)
core=tts.model
tp=core.text_processor
styleM1=tts.get_voice_style("M1")
print("voices:",tts.voice_style_names)
print("style_ttl",styleM1.ttl.shape,styleM1.ttl.dtype,"style_dp",styleM1.dp.shape,styleM1.dp.dtype)
lines=["Hi, I am Jero! Lets play.","Hello there friend.","I am ready now.","Lets go outside.","That is so cool!","Can you help me?","Good morning everyone.","I love robots.","Watch me jump high.","Time for a story.","Where did it go?","You are my friend.","Lets count to ten.","The sky is blue.","I feel very happy.","Please follow me.","What is your name?","See you tomorrow.","That was amazing!","Do not be afraid.","Lets build a tower.","I found a treasure.","Come play with me.","It is getting dark.","Sing a happy song.","Run as fast as you can.","Look at the bright stars.","I am a little robot.","Give me a high five.","Lets explore the world."]
Ts=[]; Ls=[]
for ln in lines:
    tid,tmask=tp([ln],"en")
    T=tid.shape[1]; Ts.append(T)
    dur,*_=core.dp_ort.run(None,{"text_ids":tid,"style_dp":styleM1.dp,"text_mask":tmask})
    dur=dur/1.05
    chunk=core.base_chunk_size*core.chunk_compress_factor
    wav_len_max=dur.max()*core.sample_rate
    L=int((wav_len_max+chunk-1)//chunk); Ls.append(L)
print("T: min",min(Ts),"max",max(Ts),"mean",round(np.mean(Ts),1))
print("L: min",min(Ls),"max",max(Ls),"mean",round(np.mean(Ls),1))
print("sample (text,T,L,dur):")
for ln,T,L in list(zip(lines,Ts,Ls))[:6]:
    print("  ",repr(ln[:20]),T,L)
# dump embedding tables
for nm,fn in [("text_encoder","te_emb.npy"),("duration_predictor","dp_emb.npy")]:
    m=onnx.load(os.path.expanduser("~/.cache/supertonic3/onnx/%s.onnx"%nm))
    for t in m.graph.initializer:
        if "char_embedder.weight" in t.name:
            a=numpy_helper.to_array(t)
            np.save(os.path.expanduser("~/stwork/"+fn),a)
            print(nm,"emb table",a.shape,a.dtype,"->",fn)
