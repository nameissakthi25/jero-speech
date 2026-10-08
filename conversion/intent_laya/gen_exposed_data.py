import os,json,numpy as np,onnxruntime as ort
from transformers import AutoTokenizer
EX="/home/raemox/jero/intent/export"; D="encsplit"
G1="/model/encoder/Where_1_output_0"; G2="/model/encoder/Where_2_output_0"
tok=AutoTokenizer.from_pretrained(EX); W=np.load(D+"/embed_table.npy")
mb=ort.InferenceSession(D+"/mask_builder.onnx",providers=["CPUExecutionProvider"])
full=ort.InferenceSession(EX+"/intent.onnx",providers=["CPUExecutionProvider"])
id2i=json.load(open(EX+"/label_maps.json"))["intent_id2label"]
lines=[json.loads(l) for l in open("/home/raemox/jero/intent/data/intent/test.jsonl") if l.strip()]
BOARD="/tmp/jero_enc"
def mk(text):
    e=tok(text,padding="max_length",truncation=True,max_length=32,return_tensors="np")
    ids=e["input_ids"].astype(np.int64); msk=e["attention_mask"].astype(np.int64)
    ie=W[ids[0]].reshape(1,32,768).astype(np.float32)
    g1,g2=mb.run(None,{"attention_mask":msk})
    return ie,g1.astype(np.float32),g2.astype(np.float32),ids,msk
for split,rows,outdir,listname in [("calib",lines[::max(1,len(lines)//100)][:100],D+"/ecalib","input_list.txt"),("verify",lines[:120],D+"/everify","input_list_board.txt")]:
    os.makedirs(outdir,exist_ok=True)
    host=open(outdir+"/input_list_host.txt","w"); board=open(outdir+"/"+listname,"w")
    ref=[]
    for i,r in enumerate(rows):
        ie,g1,g2,ids,msk=mk(r["text"])
        ie.tofile(outdir+"/ie_%03d.raw"%i); g1.tofile(outdir+"/g1_%03d.raw"%i); g2.tofile(outdir+"/g2_%03d.raw"%i)
        host.write("inputs_embeds:=%s/ie_%03d.raw %s:=%s/g1_%03d.raw %s:=%s/g2_%03d.raw\n"%(outdir,i,G1,outdir,i,G2,outdir,i))
        sub="ecalib" if split=="calib" else "everify"
        board.write("inputs_embeds:=%s/%s/ie_%03d.raw %s:=%s/%s/g1_%03d.raw %s:=%s/%s/g2_%03d.raw\n"%(BOARD,sub,i,G1,BOARD,sub,i,G2,BOARD,sub,i))
        if split=="verify":
            f=full.run(None,{"input_ids":ids,"attention_mask":msk}); n=int(msk.sum())
            ref.append({"i":i,"gold":r.get("intent"),"fp32_intent":int(f[0][0].argmax()),"nreal":n,"fp32_slots":[int(x) for x in f[1][0].argmax(-1)[:n]]})
    host.close(); board.close()
    if ref: json.dump(ref,open(outdir+"/fp32_ref.json","w"))
    print(split,"done",len(rows))
print("g1 bytes",os.path.getsize(D+"/everify/g1_000.raw"))
