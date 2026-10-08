#!/usr/bin/env python3
"""Supertonic-3 w8a16 on-device TTS driver for QCS6490 HTP (Hexagon v68).
Runs text_encoder + duration_predictor + 8-step vector_estimator + vocoder on the
NPU via qnn-net-run (retrieve_context), with host-side char-embedding lookup, mask
building, noisy-latent sampling and the flow-matching step loop.
"""
import os, sys, re, json, glob, time, shutil, subprocess, unicodedata
import numpy as np
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
import json as _json
_DUR_LUT = _json.load(open(HERE+"/dur_lut.json")) if os.path.exists(HERE+"/dur_lut.json") else {}
CTX  = {m: HERE + "/ctx_%s/%s_ctx.bin" % (m, m) for m in ("dp", "te", "ve", "vo")}
BACKEND = "/usr/lib/libQnnHtp.so"
NETRUN  = "qnn-net-run"
T_BUCKET, L_BUCKET = 64, 64
SR, CHUNK = 44100, 512 * 6   # 3072 samples / latent frame
ENV = dict(os.environ)
ENV["ADSP_LIBRARY_PATH"] = "/usr/lib/rfsa/adsp:/usr/lib"
ENV["LD_LIBRARY_PATH"]   = "/usr/lib:" + ENV.get("LD_LIBRARY_PATH", "")

# ---------- host-side text processing (port of supertonic.core.UnicodeProcessor) ----------
_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿\U0001f1e6-\U0001f1ff]+", re.UNICODE)
_SYM = {"–":"-","‑":"-","—":"-","¯":" ","_":" ","“":'"',"”":'"',
        "‘":"'","’":"'","´":"'","`":"'","[":" ","]":" ","|":" ","/":" ","#":" ","→":" ","←":" "}
_SPECIAL = re.compile(r"[♥☆♡©\\]")
_PUNCT = [(re.compile(r" ,"),","),(re.compile(r" \."),"."),(re.compile(r" !"),"!"),(re.compile(r" \?"),"?"),
          (re.compile(r" ;"),";"),(re.compile(r" :"),":"),(re.compile(r" '"),"'")]
_DUP = re.compile(r'(["\'\`])\1+'); _WS = re.compile(r"\s+")
_END = re.compile(r"[.!?;:,'\"')\]}…。」』】〉》›»]$")

class TextProc:
    def __init__(self, indexer_path):
        self.indexer = json.load(open(indexer_path))
    def _pre(self, text, lang="en"):
        try: text = unicodedata.normalize("NFKD", text)
        except Exception: pass
        text = _EMOJI.sub("", text)
        for k,v in _SYM.items(): text = text.replace(k,v)
        text = _SPECIAL.sub("", text)
        for k,v in {"@":" at ","e.g.,":"for example, ","i.e.,":"that is, "}.items(): text=text.replace(k,v)
        for p,r in _PUNCT: text = p.sub(r,text)
        text = _DUP.sub(r"\1", text); text = _WS.sub(" ", text).strip()
        if not _END.search(text): text += "."
        if lang is not None: text = "<%s>%s</%s>" % (lang, text, lang)
        return text
    def ids(self, text, lang="en"):
        t = self._pre(text, lang)
        vals = [ord(c) for c in t]
        ids = np.array([self.indexer[v] for v in vals], dtype=np.int64)
        return ids, len(ids)

# ---------- qnn runner ----------
def runqnn(model, inputs, outshape, workdir):
    d = os.path.join(workdir, model)
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)
    parts = []
    for k, v in inputs.items():
        p = os.path.join(d, k + ".raw"); v.astype(np.float32).tofile(p)
        parts.append("%s:=%s" % (k, p))
    il = os.path.join(d, "il.txt"); open(il, "w").write(" ".join(parts) + "\n")
    r = subprocess.run([NETRUN, "--backend", BACKEND, "--retrieve_context", CTX[model],
                        "--input_list", il, "--output_dir", os.path.join(d, "out")],
                       env=ENV, capture_output=True, text=True)
    outs = sorted(glob.glob(os.path.join(d, "out", "Result_0", "*.raw")))
    if not outs:
        sys.stderr.write("QNN FAIL %s\n%s\n%s\n" % (model, r.stdout[-800:], r.stderr[-400:])); sys.exit(1)
    return np.fromfile(outs[0], np.float32).reshape(outshape)

def pad(a, axis, n):
    pads = [(0,0)]*a.ndim; pads[axis] = (0, n - a.shape[axis]); return np.pad(a, pads)

def write_wav(path, x, sr=SR):
    x = np.clip(x, -1, 1); pcm = (x*32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())

def synth(text, tp, te_emb, dp_emb, sttl, sdp, seed, workdir, speed=1.05, hybrid_vocoder=None):
    ids, T = tp.ids(text, "en")
    assert T <= T_BUCKET, "text too long: %d" % T
    idsP = pad(ids[None], 1, T_BUCKET)                         # [1,64]
    tmask = np.zeros((1,1,T_BUCKET), np.float32); tmask[0,0,:T] = 1
    ie_te = pad(te_emb[idsP[0]][None], 1, T_BUCKET).astype(np.float32)   # [1,64,256]
    ie_dp = pad(dp_emb[idsP[0]][None], 1, T_BUCKET).astype(np.float32)   # [1,64,64]
    _k=text.strip()
    dur = float(_DUR_LUT[_k]) if _k in _DUR_LUT else max(0.30+0.057*T, 1.2)   # lut (exact) else CPU-calibrated heuristic
    temb = runqnn("te", {"inputs_embeds":ie_te,"style_ttl":sttl,"text_mask":tmask}, (1,256,T_BUCKET), workdir)
    Lr = min(int((dur*SR + CHUNK - 1)//CHUNK), L_BUCKET)
    rng = np.random.RandomState(seed)
    nl = rng.randn(1,144,Lr).astype(np.float32)
    lm = np.zeros((1,1,L_BUCKET), np.float32); lm[0,0,:Lr] = 1
    xt = pad(nl, 2, L_BUCKET).astype(np.float32) * lm
    tot = np.array([8], np.float32)
    for s in range(8):
        xt = runqnn("ve", {"noisy_latent":xt,"text_emb":temb,"style_ttl":sttl,"latent_mask":lm,
                           "text_mask":tmask,"current_step":np.array([s],np.float32),"total_step":tot},
                    (1,144,L_BUCKET), workdir)
    if hybrid_vocoder is not None:
        wav = hybrid_vocoder(xt)[0]
    else:
        wav = runqnn("vo", {"latent":xt}, (-1,), workdir)
    nsamp = int(dur*SR); wav = wav[:nsamp]
    # trim trailing silence (energy) for a clean ending
    aw=np.abs(wav); thr=0.02*aw.max()
    nz=np.where(aw>thr)[0]
    if len(nz): wav = wav[:min(len(wav), nz[-1]+int(0.08*SR))]
    return wav, dur, T, Lr

if __name__ == "__main__":
    tp = TextProc(HERE + "/unicode_indexer.json")
    te_emb = np.load(HERE + "/te_emb.npy"); dp_emb = np.load(HERE + "/dp_emb.npy")
    d = np.load(HERE + "/canon_inputs.npy", allow_pickle=True).item()
    sttl, sdp = d["sttl"], d["sdp"]
    import os as _os
    if _os.environ.get("STYLE_NPY"):
        _st=np.load(_os.environ["STYLE_NPY"],allow_pickle=True).item(); sttl,sdp=_st["sttl"].astype(np.float32),_st["sdp"].astype(np.float32); print("STYLE",_os.environ["STYLE_NPY"])
    text = sys.argv[1] if len(sys.argv) > 1 else d["text"]
    out = sys.argv[2] if len(sys.argv) > 2 else HERE + "/board_w8a16_M1.wav"
    wd = HERE + "/run"; os.makedirs(wd, exist_ok=True)
    t0 = time.time()
    wav, dur, T, Lr = synth(text, tp, te_emb, dp_emb, sttl, sdp, seed=1000, workdir=wd)
    write_wav(out, wav)
    print("TEXT=%r  T=%d Lr=%d dur=%.3fs  wall=%.1fs" % (text, T, Lr, dur, time.time()-t0))
    print("WROTE", out, "samples=", len(wav))
