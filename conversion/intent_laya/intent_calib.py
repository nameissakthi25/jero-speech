# Build calibration raws for the intent encoder: host embed + host maskbias.
import os, json, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer
BASE="/home/raemox/jero/intent"; OUT=f"{BASE}/qnn/laya_split"; CAL=f"{OUT}/calib"
os.makedirs(CAL, exist_ok=True)
tok=AutoTokenizer.from_pretrained(f"{BASE}/export")
embed=np.load(f"{OUT}/embed_table.npy")
mask_sess=ort.InferenceSession(f"{OUT}/mask_only.onnx", providers=["CPUExecutionProvider"])
# calib texts from test.jsonl (first 64)
texts=[]
with open(f"{BASE}/data/intent/test.jsonl") as f:
    for i,l in enumerate(f):
        if i>=64: break
        texts.append(json.loads(l)["text"].lower())
il=open(f"{OUT}/input_list.txt","w")
for j,t in enumerate(texts):
    enc=tok(t, truncation=True, max_length=32, padding="max_length", return_tensors="np")
    ids=enc["input_ids"].astype(np.int64); am=enc["attention_mask"].astype(np.int64)
    ie=embed[ids[0]].astype(np.float32)[None]            # [1,32,768]
    mb=mask_sess.run(None, {"attention_mask": am})[0].astype(np.float32)  # maskbias
    ie.tofile(f"{CAL}/ie_{j}.raw"); mb.tofile(f"{CAL}/mb_{j}.raw")
    il.write(f"inputs_embeds:={CAL}/ie_{j}.raw /model/encoder/Expand_output_0:={CAL}/mb_{j}.raw\n")
il.close()
print("calib raws:", len(texts), "ie shape", ie.shape, "mb shape", mb.shape)
