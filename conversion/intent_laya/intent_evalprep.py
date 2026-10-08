import os, json, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer
BASE="/home/raemox/jero/intent"; SP=f"{BASE}/qnn/laya_split"; EV=f"{SP}/eval"; RAW=f"{EV}/raw"
os.makedirs(RAW, exist_ok=True)
tok=AutoTokenizer.from_pretrained(f"{BASE}/export")
embed=np.load(f"{SP}/embed_table.npy")
mask_sess=ort.InferenceSession(f"{SP}/mask_only.onnx", providers=["CPUExecutionProvider"])
fp32=ort.InferenceSession(f"{BASE}/export/intent.onnx", providers=["CPUExecutionProvider"])
# use the frozen test split
texts=[]; 
with open(f"{BASE}/data/intent/test.jsonl") as f:
    for l in f: texts.append(json.loads(l)["text"].lower())
texts=texts[:120]
il=open(f"{EV}/input_list.txt","w")
ams=[]; fp32_intent=[]
for j,t in enumerate(texts):
    enc=tok(t, truncation=True, max_length=32, padding="max_length", return_tensors="np")
    ids=enc["input_ids"].astype(np.int64); am=enc["attention_mask"].astype(np.int64)
    ie=embed[ids[0]].astype(np.float32)[None]
    mb=mask_sess.run(None,{"attention_mask":am})[0].astype(np.float32)
    ie.tofile(f"{RAW}/ie_{j}.raw"); mb.tofile(f"{RAW}/mb_{j}.raw")
    il.write(f"inputs_embeds:=/home/ubuntu/intent_htp/eval/raw/ie_{j}.raw /model/encoder/Expand_output_0:=/home/ubuntu/intent_htp/eval/raw/mb_{j}.raw\n")
    ams.append(am[0])
    il_, sl_ = fp32.run(None,{"input_ids":ids,"attention_mask":am})
    fp32_intent.append(int(il_[0].argmax()))
il.close()
np.save(f"{EV}/ams.npy", np.array(ams)); np.save(f"{EV}/fp32_intent.npy", np.array(fp32_intent))
# save intent head (Gemm: weight[12,768], bias[12]) for host post-step
hd=np.load(f"{SP}/heads.npz")
np.save(f"{EV}/ih_w.npy", hd["model.intent_head.weight"]); np.save(f"{EV}/ih_b.npy", hd["model.intent_head.bias"])
print("prepped", len(texts), "utts; ih_w", hd["model.intent_head.weight"].shape)
