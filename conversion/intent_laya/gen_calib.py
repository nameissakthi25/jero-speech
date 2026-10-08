#!/usr/bin/env python3
"""Generate QNN raw calibration files + input_list for the intent model."""
import os, json, numpy as np
from transformers import AutoTokenizer

EXPORT = "/home/raemox/jero/intent/export"
TEST   = "/home/raemox/jero/intent/data/intent/test.jsonl"
RAW    = "/home/raemox/jero/intent/qnn/calib_raw"
MAXLEN = 32
N = 100
os.makedirs(RAW, exist_ok=True)
tok = AutoTokenizer.from_pretrained(EXPORT)
lines = [json.loads(l) for l in open(TEST) if l.strip()]
step = max(1, len(lines)//N)
sel = lines[::step][:N]
listf = open("/home/raemox/jero/intent/qnn/input_list.txt", "w")
for i, r in enumerate(sel):
    e = tok(r["text"], padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="np")
    ids = e["input_ids"].astype(np.int32)
    msk = e["attention_mask"].astype(np.int32)
    pids = os.path.join(RAW, "ids_%03d.raw" % i)
    pmsk = os.path.join(RAW, "mask_%03d.raw" % i)
    ids.tofile(pids)
    msk.tofile(pmsk)
    listf.write("input_ids:=%s attention_mask:=%s\n" % (pids, pmsk))
listf.close()
print("wrote", len(sel), "calibration pairs to", RAW)
print("input_list: /home/raemox/jero/intent/qnn/input_list.txt")
