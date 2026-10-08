import qat, torch
import transformers.models.modernbert.modeling_modernbert as mb
tok=qat.AutoTokenizer.from_pretrained(qat.EXPORT); ev=qat.load_eval_120(tok)
qat.MASK_FILL=-1.0e4
def run(bits):
    qat._ATT_BITS=bits
    mb.MODERNBERT_ATTENTION_FUNCTION["eager"]=qat.make_quant_eager()
    m,_=qat.build_model(faithful=True)  # attaches obs with _ATT_BITS
    qat.calibrate(m,tok); g,ag,_=qat.evaluate(m,*ev); return g,ag
for b in [8,6,5,4,3,2]:
    g,ag=run(b); print("attn_bits=%d  gold=%.3f agree=%.3f"%(b,g,ag))
