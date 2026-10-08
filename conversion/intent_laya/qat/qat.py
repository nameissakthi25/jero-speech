#!/usr/bin/env python3
"""Device-faithful w8a16 QAT for Jero ModernBERT joint intent+slot model.

Faithful sim of QCS6490 / HTP v68:
  - Linear weights  -> int8  per-output-channel symmetric
  - Linear outputs  -> uint16 per-tensor asymmetric (A16)
  - Attention BMM operands Q, K, probs, V -> uint8 per-tensor asymmetric (the
    v68 8x8 activation-x-activation matmul constraint; THIS is the crux)
  - attention mask fill = -1e4 (not -inf / finfo.min)

Phase A: build sim, calibrate, verify it reproduces the ~42% on-device baseline.
Phase B: QAT - adapt weights (STE) under the frozen quant grid, joint loss,
         eval the 120-utt set every N steps, checkpoint best.
"""
import os, sys, json, time, math, argparse, copy
sys.path.insert(0, "/home/raemox/jero/intent/training/intent")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer
import transformers.models.modernbert.modeling_modernbert as mb
from model import LayaJointModel
from labels import INTENT2ID, ID2SLOT, IGNORE_INDEX
from align import encode_example

BEST   = "/home/raemox/jero/intent/runs/best"
DATA   = "/home/raemox/jero/intent/data/intent"
EXPORT = "/home/raemox/jero/intent/export"
OUT    = "/home/raemox/jero/intent/qnn/qat"
MAXLEN = 32
DEV    = "cuda"

# ---------------- fake quant primitives ----------------
class _FakeQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zp, qmin, qmax):
        q = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
        return (q - zp) * scale
    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None

def fq(x, scale, zp, qmin, qmax):
    return _FakeQ.apply(x, scale, zp, qmin, qmax)

def asym_qparams(mn, mx, qmin, qmax, eps=1e-8):
    mn = torch.minimum(mn, torch.zeros_like(mn))
    mx = torch.maximum(mx, torch.zeros_like(mx))
    scale = (mx - mn) / (qmax - qmin)
    scale = torch.clamp(scale, min=eps)
    zp = qmin - torch.round(mn / scale)
    return scale, zp

class ActObserver(nn.Module):
    """Per-tensor asymmetric activation fake-quant with frozen min/max buffers."""
    def __init__(self, bits):
        super().__init__()
        self.qmin, self.qmax = 0, (1 << bits) - 1
        self.register_buffer("mn", torch.tensor(float("inf")))
        self.register_buffer("mx", torch.tensor(float("-inf")))
        self.calibrating = False
        self.enabled = True
    def reset(self):
        self.mn.fill_(float("inf")); self.mx.fill_(float("-inf"))
    def forward(self, x):
        if self.calibrating:
            with torch.no_grad():
                self.mn = torch.minimum(self.mn, x.min())
                self.mx = torch.maximum(self.mx, x.max())
            return x
        if not self.enabled or not torch.isfinite(self.mn):
            return x
        scale, zp = asym_qparams(self.mn, self.mx, self.qmin, self.qmax)
        return fq(x, scale, zp, self.qmin, self.qmax)

def quant_weight_perchannel(w, bits=8):
    """int8 per-output-channel symmetric fake-quant (dynamic from current w)."""
    qmax = (1 << (bits - 1)) - 1  # 127
    wf = w.reshape(w.shape[0], -1)
    amax = wf.abs().max(dim=1).values.clamp(min=1e-8)
    scale = (amax / qmax).reshape([-1] + [1] * (w.dim() - 1))
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return _STEround.apply(w, q, scale)

class _STEround(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, scale):
        return q * scale
    @staticmethod
    def backward(ctx, g):
        return g, None, None

class QuantLinear(nn.Module):
    """Wrap an nn.Linear: int8 per-channel weights, uint16 output activation."""
    def __init__(self, lin, abits=16, wbits=8):
        super().__init__()
        self.weight = lin.weight
        self.bias = lin.bias
        self.wbits = wbits
        self.out_obs = ActObserver(abits)
    def forward(self, x):
        wq = quant_weight_perchannel(self.weight, self.wbits)
        out = F.linear(x, wq, self.bias)
        return self.out_obs(out)

# ---------------- faithful eager attention (8x8 BMM) ----------------
# _ATT_BITS=3 is the DEVICE-FAITHFUL PROXY: it reproduces the measured on-device
# ~42-45% w8a16 baseline. The v68 HTP graph-prepare converts the attention
# activation-x-activation matmul operands to per-tensor uint8 with a wide range
# reused from the 16-bit encoding (all 12 heads x 32 pos x 64 dim share one
# scale) -> effective precision ~3 bits. Tight-calibrated 8-bit (what AIMET /
# QNN-x86 do) keeps ~100% and does NOT match the device; 3-bit matches it.
MASK_FILL = -1.0e4
_ATT_BITS = 3

def make_quant_eager():
    apply_rope = mb.apply_rotary_pos_emb
    def quant_eager(module, qkv, attention_mask, sliding_window_mask,
                    position_ids, local_attention, bs, dim,
                    output_attentions=False, **kw):
        cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
        query, key, value = qkv.transpose(3, 1).unbind(dim=2)
        query, key = apply_rope(query, key, cos, sin)
        scale = module.head_dim ** -0.5
        # ---- 8-bit on BOTH Q@K^T operands (v68 constraint) ----
        q_q = module.q_obs(query)
        k_q = module.k_obs(key)
        attn_weights = torch.matmul(q_q, k_q.transpose(2, 3)) * scale
        am = attention_mask
        if local_attention != (-1, -1):
            am = sliding_window_mask
        am = am.clamp(min=MASK_FILL)  # -inf -> -1e4
        attn_weights = attn_weights + am
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=module.attention_dropout, training=module.training)
        # ---- 8-bit on BOTH attn@V operands ----
        p_q = module.p_obs(attn_weights)
        v_q = module.v_obs(value)
        attn_output = torch.matmul(p_q, v_q)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, -1, dim)
        if output_attentions:
            return (attn_output, attn_weights)
        return (attn_output,)
    return quant_eager

# ---------------- model surgery ----------------
def wrap_linears(module, abits=16, wbits=8, skip=()):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name not in skip:
            setattr(module, name, QuantLinear(child, abits, wbits))
        else:
            wrap_linears(child, abits, wbits, skip)

def attach_attn_observers(model):
    obs_mods = []
    for m in model.modules():
        if isinstance(m, mb.ModernBertAttention):
            for nm in ("q_obs", "k_obs", "p_obs", "v_obs"):
                o = ActObserver(_ATT_BITS)
                m.add_module(nm, o)
                obs_mods.append(o)
    return obs_mods

def all_observers(model):
    return [m for m in model.modules() if isinstance(m, ActObserver)]

def set_calib(model, flag):
    for o in all_observers(model):
        o.calibrating = flag

def reset_observers(model):
    for o in all_observers(model):
        o.reset()

# ---------------- data ----------------
def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def tokenize(tok, text):
    e = tok(text, padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")
    return e["input_ids"].long(), e["attention_mask"].long()

def build_train_batch(tok, rows):
    ids, am, sl, il = [], [], [], []
    for r in rows:
        try:
            enc = encode_example(tok, r["text"], r["slots"], MAXLEN)
        except AssertionError:
            continue
        ids.append(enc["input_ids"]); am.append(enc["attention_mask"])
        sl.append(enc["slot_labels"]); il.append(INTENT2ID[r["intent"]])
    return (torch.tensor(ids), torch.tensor(am), torch.tensor(sl), torch.tensor(il))

# ---------------- eval ----------------
def load_eval_120(tok):
    rows = read_jsonl(f"{DATA}/test.jsonl")[:120]
    ref = json.load(open("/home/raemox/jero/intent/qnn/deliver/fp32_ref_board.json"))
    ids, am, gold, fp32 = [], [], [], []
    for k, r in enumerate(rows):
        i, a = tokenize(tok, r["text"].lower())
        ids.append(i); am.append(a)
        gold.append(INTENT2ID[r["intent"]])
        fp32.append(ref[k]["fp32_intent"])
    return (torch.cat(ids), torch.cat(am),
            torch.tensor(gold), torch.tensor(fp32))

@torch.no_grad()
def evaluate(model, ids, am, gold, fp32, bs=60):
    model.eval()
    preds = []
    for s in range(0, ids.shape[0], bs):
        i = ids[s:s+bs].to(DEV); a = am[s:s+bs].to(DEV)
        out = model(input_ids=i, attention_mask=a)
        preds.append(out["intent_logits"].argmax(-1).cpu())
    preds = torch.cat(preds)
    acc_gold = (preds == gold).float().mean().item()
    agree = (preds == fp32).float().mean().item()
    return acc_gold, agree, preds

@torch.no_grad()
def calibrate(model, tok, n=200, bs=50):
    rows = read_jsonl(f"{DATA}/train.jsonl")
    step = max(1, len(rows) // n)
    rows = rows[::step][:n]
    ids = []; ams = []
    for r in rows:
        i, a = tokenize(tok, r["text"].lower())
        ids.append(i); ams.append(a)
    ids = torch.cat(ids); ams = torch.cat(ams)
    reset_observers(model); set_calib(model, True); model.eval()
    for s in range(0, ids.shape[0], bs):
        model(input_ids=ids[s:s+bs].to(DEV), attention_mask=ams[s:s+bs].to(DEV))
    set_calib(model, False)

# ---------------- build ----------------
def build_model(faithful=True):
    cfg = json.load(open(f"{BEST}/laya_config.json"))
    model = LayaJointModel(cfg["base_name"], attn_impl="eager")
    model.encoder.config.reference_compile = False  # no torch.compile wrap
    sd = torch.load(f"{BEST}/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(sd)
    if faithful:
        mb.MODERNBERT_ATTENTION_FUNCTION["eager"] = make_quant_eager()
        attach_attn_observers(model)
        wrap_linears(model)
    model.to(DEV)
    return model, cfg

def log(msg, f=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if f:
        f.write(line + "\n"); f.flush()

def main():
    global _ATT_BITS
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["verify", "qat"], default="verify")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--recalib_every", type=int, default=100)
    ap.add_argument("--slot_w", type=float, default=1.0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--attn_bits", type=int, default=_ATT_BITS)
    args = ap.parse_args()
    _ATT_BITS = args.attn_bits
    os.makedirs(OUT, exist_ok=True)
    lf = open(f"{OUT}/logs/{args.tag}.log", "a")
    tok = AutoTokenizer.from_pretrained(EXPORT)
    ev = load_eval_120(tok)

    model, cfg = build_model(faithful=True)
    log(f"mode={args.mode} tag={args.tag}", lf)

    # fp32 (no-quant) sanity: disable quant, eval
    for o in all_observers(model):
        o.enabled = False
    g0, a0, _ = evaluate(model, *ev)
    log(f"[fp32-path, quant disabled] gold_acc={g0:.4f} fp32_agree={a0:.4f}", lf)
    for o in all_observers(model):
        o.enabled = True

    calibrate(model, tok)
    g1, a1, _ = evaluate(model, *ev)
    log(f"[FAITHFUL w8a16 sim, PTQ] gold_acc={g1:.4f} fp32_agree={a1:.4f}  "
        f"(expect ~0.42 to match on-device 42.5%)", lf)

    if args.mode == "verify":
        log("verify done.", lf); return

    # ---------------- QAT ----------------
    train_rows = read_jsonl(f"{DATA}/train.jsonl")
    import random
    random.seed(0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    best = {"agree": a1, "gold": g1, "step": 0}
    torch.save(model.state_dict(), f"{OUT}/ckpt_{args.tag}_best.pt")
    curve = []
    step = 0
    log(f"start QAT steps={args.steps} bs={args.bs} lr={args.lr} slot_w={args.slot_w}", lf)
    while step < args.steps:
        random.shuffle(train_rows)
        for s in range(0, len(train_rows), args.bs):
            batch = train_rows[s:s+args.bs]
            ids, am, sl, il = build_train_batch(tok, batch)
            if ids.shape[0] == 0:
                continue
            ids = ids.to(DEV); am = am.to(DEV); sl = sl.to(DEV); il = il.to(DEV)
            model.train()
            out = model(input_ids=ids, attention_mask=am,
                        intent_labels=il, slot_labels=sl, slot_loss_w=args.slot_w)
            loss = out["loss"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 25 == 0:
                log(f"step {step} loss {loss.item():.4f}", lf)
            if step % args.recalib_every == 0:
                calibrate(model, tok)
            if step % args.eval_every == 0:
                g, a, _ = evaluate(model, *ev)
                curve.append({"step": step, "gold_acc": g, "fp32_agree": a,
                              "loss": float(loss.item())})
                json.dump(curve, open(f"{OUT}/logs/{args.tag}_curve.json", "w"), indent=2)
                star = ""
                if a > best["agree"]:
                    best = {"agree": a, "gold": g, "step": step}
                    torch.save(model.state_dict(), f"{OUT}/ckpt_{args.tag}_best.pt")
                    star = " *BEST*"
                log(f"[EVAL step {step}] gold_acc={g:.4f} fp32_agree={a:.4f}"
                    f"  best_agree={best['agree']:.4f}@{best['step']}{star}", lf)
            if step >= args.steps:
                break
    log(f"QAT done. best fp32_agree={best['agree']:.4f} gold_acc={best['gold']:.4f} "
        f"@step {best['step']}. ckpt={OUT}/ckpt_{args.tag}_best.pt", lf)

if __name__ == "__main__":
    main()
