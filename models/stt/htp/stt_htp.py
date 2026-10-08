#!/usr/bin/env python3
"""Streaming Zipformer2 RNN-T greedy decode on the QCS6490 HTP via qnn-net-run.
Runs ENCODER + DECODER + JOINER (all W8A16) on the Hexagon v68 HTP.
All quant/dequant + state transposes done in numpy; qnn-net-run does raw native I/O.
Runs entirely on the board (board-local subprocess calls)."""
import os, sys, json, glob, time, shutil, subprocess, numpy as np

RUN   = "/opt/stt"
MODEL = "/opt/stt"
BACKEND = "/usr/lib/libQnnHtp.so"
WORK  = "/tmp/htp_work"
QNR   = "qnn-net-run"
CTXUTIL = "qnn-context-binary-utility"

BINS = {"encoder": f"{MODEL}/encoder_ctx.bin.bin",
        "decoder": f"{MODEL}/decoder_ctx.bin.bin",
        "joiner":  f"{MODEL}/joiner_ctx.bin.bin"}

# ---- metadata ----
def meta(binpath, tag):
    jf = f"/opt/stt/meta/{tag}.json"   # precomputed (device lacks qnn-context-binary-utility)
    d = json.load(open(jf))
    g = d["info"]["graphs"][0]["info"]
    def tl(key):
        out = []
        for t in g[key]:
            ti = t["info"]; qp = ti.get("quantizeParams", {})
            so = qp.get("scaleOffset") if isinstance(qp, dict) else None
            out.append(dict(name=ti["name"], dims=list(ti["dimensions"]),
                            dtype=ti["dataType"],
                            scale=(so or {}).get("scale"), offset=(so or {}).get("offset")))
        return out
    return g["graphName"], tl("graphInputs"), tl("graphOutputs")

def npdt(dtype):
    if dtype == "QNN_DATATYPE_UFIXED_POINT_16": return np.uint16
    if dtype == "QNN_DATATYPE_INT_32": return np.int32
    if dtype == "QNN_DATATYPE_FLOAT_32": return np.float32
    raise ValueError(dtype)

def q16(real, scale, offset):
    return np.clip(np.rint(real/scale) - offset, 0, 65535).astype(np.uint16)
def dq16(q, scale, offset):
    return (q.astype(np.float32) + offset) * scale

# ---- qnn-net-run wrapper ----
def run_graph(binpath, gname, ins, outs_meta, inputs_native):
    """inputs_native: name -> np array already in the graph's native dtype/shape.
    returns name -> native np array (output)."""
    d = os.path.join(WORK, gname)
    if os.path.isdir(d): shutil.rmtree(d)
    os.makedirs(d)
    parts = []
    for t in ins:
        a = np.ascontiguousarray(inputs_native[t["name"]], dtype=npdt(t["dtype"]))
        p = os.path.join(d, t["name"] + ".raw"); a.tofile(p)
        parts.append(f'{t["name"]}:={p}')
    il = os.path.join(d, "in.txt")
    open(il, "w").write(" ".join(parts) + "\n")
    od = os.path.join(d, "out")
    r = subprocess.run([QNR, "--retrieve_context", binpath, "--backend", BACKEND,
                        "--input_list", il, "--output_dir", od,
                        "--use_native_input_files", "--use_native_output_files"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.stderr.write(r.stdout.decode(errors="replace")[-2000:]); raise RuntimeError(f"{gname} rc={r.returncode}")
    res = {}
    for t in outs_meta:
        fs = (glob.glob(os.path.join(od, "**", t["name"] + "_native.raw"), recursive=True)
              or glob.glob(os.path.join(od, "**", t["name"] + ".raw"), recursive=True)
              or glob.glob(os.path.join(od, "**", t["name"] + "*.raw"), recursive=True))
        if not fs: raise RuntimeError(f"missing output {t['name']} in {od}")
        a = np.fromfile(fs[0], dtype=npdt(t["dtype"]))
        res[t["name"]] = a.reshape([1 if (isinstance(x,str) or x in (0,None)) else x for x in t["dims"]])
    return res

# ---- load graph metas ----
G = {}
for tag, b in BINS.items():
    name, gi, go = meta(b, tag)
    G[tag] = dict(bin=b, gname=name, ins=gi, outs=go,
                  imap={t["name"]: t for t in gi}, omap={t["name"]: t for t in go})

# state family -> transpose from OUTPUT(new_) layout to INPUT layout
PERM = {"len": None, "avg": (0,2,1), "key": (0,2,3,1), "val": (0,2,3,1),
        "val2": (0,2,3,1), "conv1": (0,2,3,1), "conv2": (0,2,3,1)}
def fam(name):  # strip new_cached_/cached_ prefix and trailing _idx
    s = name.replace("new_cached_","").replace("cached_","")
    base = "_".join(s.split("_")[:-1])
    return base

# ---- backends ----
def encoder_fn(x_win_f, states_f):
    e = G["encoder"]
    native = {}
    # x: float [1,39,80] -> transpose to [1,80,39] -> quantize
    xi = e["imap"]["x"]
    xq = q16(np.transpose(x_win_f, (0,2,1)), xi["scale"], xi["offset"])
    native["x"] = xq
    for t in e["ins"]:
        n = t["name"]
        if n == "x": continue
        sf = states_f[n]
        if t["dtype"] == "QNN_DATATYPE_INT_32":
            native[n] = sf.astype(np.int32)
        else:
            native[n] = q16(sf, t["scale"], t["offset"])
    out = run_graph(e["bin"], e["gname"], e["ins"], e["outs"], native)
    # encoder_out -> float
    om = e["omap"]["encoder_out"]
    enc = dq16(out["encoder_out"], om["scale"], om["offset"])   # [1,8,512]
    # build next input states (float, input layout) from outputs (output layout)
    new_states = {}
    for t in e["ins"]:
        n = t["name"]
        if n == "x": continue
        on = "new_" + n
        om2 = e["omap"][on]
        if t["dtype"] == "QNN_DATATYPE_INT_32":
            arr = out[on].astype(np.float32)
        else:
            arr = dq16(out[on], om2["scale"], om2["offset"])
        p = PERM[fam(n)]
        if p is not None: arr = np.transpose(arr, p)
        new_states[n] = arr.reshape([1 if (isinstance(x,str) or x in (0,None)) else x for x in t["dims"]])
    return enc, new_states

def decoder_fn(y_i64):
    g = G["decoder"]
    native = {"y": y_i64.astype(np.int32)}
    out = run_graph(g["bin"], g["gname"], g["ins"], g["outs"], native)
    om = g["omap"]["decoder_out"]
    return dq16(out["decoder_out"], om["scale"], om["offset"])   # [1,512] float

def joiner_fn(enc_f, dec_f):
    g = G["joiner"]
    ei = g["imap"]["encoder_out"]; di = g["imap"]["decoder_out"]
    native = {"encoder_out": q16(enc_f, ei["scale"], ei["offset"]),
              "decoder_out": q16(dec_f, di["scale"], di["offset"])}
    out = run_graph(g["bin"], g["gname"], g["ins"], g["outs"], native)
    return out["logit"]  # raw uint16 [1,2000]; argmax over q == argmax over real

# ---- init states (float zeros, input layout) ----
def init_states():
    st = {}
    for t in G["encoder"]["ins"]:
        if t["name"] == "x": continue
        shp = [1 if (isinstance(x,str) or x in (0,None)) else x for x in t["dims"]]
        st[t["name"]] = np.zeros(shp, np.float32)
    return st

# ---- tokens ----
def load_tokens(p):
    id2t = {}
    for line in open(p, encoding="utf-8"):
        a = line.split()
        if len(a) >= 2: id2t[int(a[1])] = a[0]
    return id2t
def ids_to_text(ids, id2t):
    return "".join(id2t.get(i,"") for i in ids).replace("▁"," ").strip()

T_WIN, CHUNK, CTX, BLANK = 39, 32, 2, 0


def decode_feats(feats):
    feats = feats.astype(np.float32); nf = feats.shape[0]
    npad = (-(nf - T_WIN)) % CHUNK
    if nf < T_WIN: npad = T_WIN - nf
    if npad: feats = np.concatenate([feats, np.full((npad,80), feats.min(), np.float32)], 0); nf = feats.shape[0]
    states = init_states(); hyp=[BLANK]*CTX
    dec_out = decoder_fn(np.array([hyp[-CTX:]], np.int64)); emitted=[]
    i=0; t_enc=0.0; t0=time.time()
    while i + T_WIN <= nf:
        x=feats[i:i+T_WIN][None,:,:]
        te=time.time(); enc,states=encoder_fn(x,states); t_enc+=time.time()-te
        for t in range(enc.shape[1]):
            logit=joiner_fn(enc[:,t,:],dec_out); k=int(np.argmax(logit[0]))
            if k!=BLANK: hyp.append(k); emitted.append(k); dec_out=decoder_fn(np.array([hyp[-CTX:]],np.int64))
        i+=CHUNK
    return emitted, nf/100.0, t_enc, time.time()-t0

def batch():
    id2t=load_tokens(f"{MODEL}/tokens.txt")
    listfile=sys.argv[2]
    rows=[l.split("\t") for l in open(listfile).read().strip().split("\n") if l.strip()]
    out=open(sys.argv[3],"w"); tot_a=0.0; tot_w=0.0
    for uid,npy in rows:
        emitted,au,te,wa=decode_feats(np.load(npy)); txt=ids_to_text(emitted,id2t)
        out.write(f"{uid}\t{txt}\n"); out.flush(); tot_a+=au; tot_w+=wa
        print(f"{uid}\t{txt}\t(RTF {wa/au:.2f})",flush=True)
    print(f"BATCH_DONE utts={len(rows)} audio={tot_a:.1f}s wall={tot_w:.1f}s RTF={tot_w/tot_a:.3f}")
    out.close()

def main():
    os.makedirs(WORK, exist_ok=True)
    if "--test" in sys.argv:
        d = decoder_fn(np.array([[0,0]], np.int64))
        print("HTP decoder([0,0]) out[:6]", np.round(d[0,:6],4), "norm", round(float(np.linalg.norm(d)),4))
        return
    feats = np.load(sys.argv[1] if len(sys.argv)>1 else f"{RUN}/sample_feats.npy").astype(np.float32)
    nf = feats.shape[0]
    npad = (-(nf - T_WIN)) % CHUNK
    if nf < T_WIN: npad = T_WIN - nf
    if npad: feats = np.concatenate([feats, np.full((npad,80), feats.min(), np.float32)], 0); nf = feats.shape[0]
    id2t = load_tokens(f"{MODEL}/tokens.txt")

    states = init_states()
    hyp = [BLANK]*CTX
    dec_out = decoder_fn(np.array([hyp[-CTX:]], np.int64))
    emitted = []
    i = 0; nch = 0
    t_enc = 0.0; t0 = time.time()
    while i + T_WIN <= nf:
        x = feats[i:i+T_WIN][None,:,:]
        te = time.time(); enc, states = encoder_fn(x, states); t_enc += time.time()-te
        for t in range(enc.shape[1]):
            logit = joiner_fn(enc[:,t,:], dec_out)
            k = int(np.argmax(logit[0]))
            if k != BLANK:
                hyp.append(k); emitted.append(k)
                dec_out = decoder_fn(np.array([hyp[-CTX:]], np.int64))
        i += CHUNK; nch += 1
        print(f"  chunk {nch:2d} emitted={len(emitted)}", flush=True)
    wall = time.time() - t0
    audio_s = nf/100.0
    text = ids_to_text(emitted, id2t)
    print("="*60)
    print("TRANSCRIPTION:", text)
    print("="*60)
    print(f"ids={emitted}")
    print(f"chunks={nch} audio={audio_s:.2f}s encoder_HTP_wall={t_enc:.2f}s total_wall={wall:.2f}s RTF={wall/audio_s:.3f}")

if __name__ == "__main__":
    (batch() if "--batch" in sys.argv else main())
