import qat, torch, sys
import transformers.models.modernbert.modeling_modernbert as mb
CKPT=sys.argv[1] if len(sys.argv)>1 else "/home/raemox/jero/intent/qnn/qat/ckpt_qat3_best.pt"
tok=qat.AutoTokenizer.from_pretrained(qat.EXPORT); ev=qat.load_eval_120(tok)
def build_and_load(bits):
    qat._ATT_BITS=bits
    mb.MODERNBERT_ATTENTION_FUNCTION["eager"]=qat.make_quant_eager()
    m,_=qat.build_model(faithful=True)
    sd=torch.load(CKPT,map_location="cpu"); m.load_state_dict(sd); m.to(qat.DEV)
    return m
print("checkpoint:",CKPT)
# fp32 path (quant disabled)
m=build_and_load(3)
for o in qat.all_observers(m): o.enabled=False
g,a,_=qat.evaluate(m,*ev); print("fp32-path (no quant)      gold=%.3f agree=%.3f"%(g,a))
for bits in [8,4,3,2]:
    m=build_and_load(bits); qat.calibrate(m,tok); g,a,_=qat.evaluate(m,*ev)
    print("w8a16 + attn%d-bit proxy   gold=%.3f agree=%.3f"%(bits,g,a))
