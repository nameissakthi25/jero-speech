import os, json, glob, numpy as np, sys
BASE="/home/raemox/jero/intent"
OUT=sys.argv[1] if len(sys.argv)>1 else f"{BASE}/qnn/qat/out_x86htp"
ref=json.load(open(f"{BASE}/qnn/deliver/fp32_ref_board.json"))
im=0; ig=0; N=len(ref); miss=0; from collections import Counter; h=Counter()
for r in ref:
    i=r["i"]; rd=os.path.join(OUT,"Result_%d"%i)
    fs=glob.glob(os.path.join(rd,"intent_logits*.raw"))
    if not fs: miss+=1; continue
    lg=np.fromfile(fs[0],dtype=np.float32)
    qi=int(lg.reshape(-1)[:12].argmax()); h[qi]+=1
    im+=(qi==r["fp32_intent"]); 
    gold=r["gold"]; 
    # gold index via name
id2={int(k):v for k,v in json.load(open(f"{BASE}/export/label_maps.json"))["intent_id2label"].items()}
name2id={v:k for k,v in id2.items()}
im=0; ig=0
for r in ref:
    i=r["i"]; rd=os.path.join(OUT,"Result_%d"%i)
    fs=glob.glob(os.path.join(rd,"intent_logits*.raw"))
    if not fs: continue
    lg=np.fromfile(fs[0],dtype=np.float32).reshape(-1)[:12]
    qi=int(lg.argmax())
    im+=(qi==r["fp32_intent"]); ig+=(qi==name2id[r["gold"]])
print("missing:",miss)
print("top-1 intent agreement vs FP32: %d/%d = %.2f%%"%(im,N,100*im/N))
print("top-1 absolute accuracy vs gold: %d/%d = %.2f%%"%(ig,N,100*ig/N))
print("pred hist:",dict(h.most_common()))
