import json, onnx
enc=json.load(open("aimet_out/intent.encodings"))
def rng(r):
    s=r["scale"][0] if isinstance(r["scale"],list) else r["scale"]
    o=r["offset"][0] if isinstance(r["offset"],list) else r["offset"]
    bw=r["bw"]; lo=s*o; hi=s*(o+(2**bw-1))
    return lo,hi,hi-lo,s
acts=[]
for r in enc["activation_encodings"]:
    lo,hi,span,s=rng(r)
    acts.append((span,s,lo,hi,r["bw"],r["name"]))
acts.sort(reverse=True)
print("=== TOP 15 widest-range activation tensors (precision killers) ===")
for span,s,lo,hi,bw,n in acts[:15]:
    print("span=%.3g lo=%.3g hi=%.3g scale=%.3g bw=%d  %s"%(span,lo,hi,s,bw,n))
print()
pars=[]
for r in enc["param_encodings"]:
    lo,hi,span,s=rng(r)
    pars.append((span,s,r["bw"],r["name"]))
pars.sort(reverse=True)
print("=== TOP 8 widest-range param tensors ===")
for span,s,bw,n in pars[:8]:
    print("span=%.3g scale=%.3g bw=%d  %s"%(span,s,bw,n))
