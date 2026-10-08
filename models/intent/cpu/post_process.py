"""Runtime post-processing for the exported Laya intent.onnx.

Pipeline:
    intent_logits, slot_logits  (ONNXRuntime)
      -> softmax + temperature -> intent + confidence
      -> per-token BIO argmax -> surface spans (first-subword labels)
      -> normalized slot VALUES (direction/steps/duration_s/angle_deg/speed/
         style/target/emotion)
      -> {"intent","confidence","unclear","slots":[...]}

An intent with confidence < THRESHOLD (default 0.7) is marked unclear=True so
the robot can ask again instead of acting on a low-confidence guess.

This module is self-contained (label maps embedded) so it ships with the
export dir and needs only numpy + onnxruntime + a HF tokenizer at runtime.
"""
import re

INTENT_ID2LABEL = {
    0: "walk", 1: "turn", 2: "stop", 3: "dance", 4: "look", 5: "greet",
    6: "emote", 7: "yes", 8: "no", 9: "cancel", 10: "chit_chat", 11: "none",
}
SLOT_ID2LABEL = {
    0: "O", 1: "B-direction", 2: "I-direction", 3: "B-steps", 4: "I-steps",
    5: "B-duration", 6: "I-duration", 7: "B-angle", 8: "I-angle",
    9: "B-speed", 10: "I-speed", 11: "B-style", 12: "I-style",
    13: "B-target", 14: "I-target", 15: "B-emotion", 16: "I-emotion",
}

DEFAULT_T = 1.0
DEFAULT_THRESHOLD = 0.7

_NUMWORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a couple": 2,
    "ninety": 90, "forty five": 45, "quarter": 90, "half": 180,
}


def _softmax(x):
    x = x - x.max()
    e = __import__("numpy").exp(x)
    return e / e.sum()


# --------------------------- value normalizers ---------------------------
def norm_direction(s):
    s = s.lower()
    if "forward" in s or "ahead" in s or "straight" in s or "foward" in s:
        return "forward"
    if "back" in s:
        return "back"
    if "left" in s or "lift" in s:
        return "left"
    if "right" in s or "rite" in s:
        return "right"
    return s


def norm_steps(s):
    s = s.lower()
    if "a bit" in s:
        return 2
    if "a few" in s or "couple" in s:
        return 3
    m = re.search(r"\d+", s)
    if m:
        return int(m.group())
    for w, v in _NUMWORDS.items():
        if re.search(r"\b" + re.escape(w) + r"\b", s):
            return v
    if "step" in s:
        return 1
    return None


def norm_duration(s):
    s = s.lower()
    m = re.search(r"(\d+)\s*(second|sec|s)\b", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*(minute|min)", s)
    if m:
        return int(m.group(1)) * 60
    if "a minute" in s:
        return 60
    for w, v in {"five": 5, "ten": 10, "thirty": 30, "twenty": 20,
                 "a few": 3}.items():
        if w in s:
            return v
    if "a while" in s:
        return 10
    return None


def norm_angle(s):
    s = s.lower()
    if "around" in s or "all the way" in s:
        return 180
    if "full turn" in s:
        return 360
    if "halfway" in s or "half" in s:
        return 180
    if "quarter" in s:
        return 90
    m = re.search(r"\d+", s)
    if m:
        return int(m.group())
    if "ninety" in s:
        return 90
    return None


def norm_speed(s):
    s = s.lower()
    if "slow" in s:
        return "slow"
    if "fast" in s or "quick" in s:
        return "fast"
    return "normal"


def norm_style(s):
    s = s.lower()
    if "part" in s:
        return "party"
    if "slow" in s:
        return "slow"
    if "happy" in s or "happi" in s:
        return "happy"
    return s


def norm_target(s):
    s = s.lower()
    if "up" in s:
        return "up"
    if "down" in s:
        return "down"
    if "me" in s:
        return "me"
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    return s


def norm_emotion(s):
    s = s.lower()
    for e in ("happy", "sad", "angry", "curious", "sleepy"):
        if e in s:
            return e
    return s


NORMALIZERS = {
    "direction": ("direction", norm_direction),
    "steps": ("steps", norm_steps),
    "duration": ("duration_s", norm_duration),
    "angle": ("angle_deg", norm_angle),
    "speed": ("speed", norm_speed),
    "style": ("style", norm_style),
    "target": ("target", norm_target),
    "emotion": ("emotion", norm_emotion),
}


# --------------------------- BIO -> spans ---------------------------
def bio_to_spans(tags, offsets, word_ids, text=None):
    """Reduce first-subword BIO predictions to char spans.

    tags: list[int] per token (argmax of slot_logits)
    offsets: list[(start,end)] char offsets per token
    word_ids: list[int|None] per token
    text: original string (used to trim BPE leading-space offsets)
    """
    spans = []
    cur = None
    prev_wid = None
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid == prev_wid:  # continuation subword: ignore (first-subword only)
            continue
        prev_wid = wid
        tag = SLOT_ID2LABEL.get(int(tags[i]), "O")
        cs, ce = offsets[i]
        if text is not None:  # trim BPE leading space
            while cs < ce and cs < len(text) and text[cs] == " ":
                cs += 1
        if tag == "O":
            if cur:
                spans.append(cur); cur = None
            continue
        bio, typ = tag.split("-", 1)
        if bio == "B" or cur is None or cur["type"] != typ:
            if cur:
                spans.append(cur)
            cur = {"type": typ, "start": cs, "end": ce}
        else:  # I- same type
            cur["end"] = ce
    if cur:
        spans.append(cur)
    return spans


def decode(text, intent_logits, slot_logits, offsets, word_ids,
           temperature=DEFAULT_T, threshold=DEFAULT_THRESHOLD):
    """Full decode for one utterance. logits are 1-D / 2-D numpy arrays."""
    probs = _softmax(intent_logits / temperature)
    intent_id = int(probs.argmax())
    conf = float(probs[intent_id])
    intent = INTENT_ID2LABEL[intent_id]

    slot_tags = slot_logits.argmax(-1).tolist()
    spans = bio_to_spans(slot_tags, offsets, word_ids, text=text)
    slots = []
    for sp in spans:
        surface = text[sp["start"]:sp["end"]]
        key, fn = NORMALIZERS[sp["type"]]
        slots.append({
            "type": sp["type"], "surface": surface,
            "start": sp["start"], "end": sp["end"],
            "value_key": key, "value": fn(surface),
        })

    unclear = conf < threshold
    return {
        "text": text, "intent": intent if not unclear else "unclear",
        "raw_intent": intent, "confidence": round(conf, 4),
        "unclear": unclear, "slots": slots,
    }


# ------------------------- end-to-end convenience -------------------------
def run(text, session, tokenizer, max_len=32,
        temperature=DEFAULT_T, threshold=DEFAULT_THRESHOLD):
    """Run tokenizer -> ONNXRuntime session -> decode. Returns the result dict.

    `session` is an onnxruntime.InferenceSession for intent.onnx.
    """
    import numpy as np
    enc = tokenizer(text, truncation=True, max_length=max_len,
                    padding="max_length", return_offsets_mapping=True)
    ids = np.array([enc["input_ids"]], dtype=np.int64)
    mask = np.array([enc["attention_mask"]], dtype=np.int64)
    intent_logits, slot_logits = session.run(
        None, {"input_ids": ids, "attention_mask": mask})
    return decode(text, intent_logits[0], slot_logits[0],
                  enc["offset_mapping"], enc.word_ids(),
                  temperature=temperature, threshold=threshold)
