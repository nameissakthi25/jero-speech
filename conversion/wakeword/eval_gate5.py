#!/usr/bin/env python3
"""
eval_gate5.py -- Gate 5 evaluation harness for the "Hey Jero" wake-word model.

Measures the two gate-5 metrics:
  (a) MISS RATE on a held-out positive set
        miss = clip whose peak hey_jero score never crosses --threshold.
        Gate target: <= 1 miss in 10  (<= 10 %).
  (b) FALSE ACCEPTS PER HOUR over a long noise-only stream
        a false accept = a debounced activation (N consecutive frames over
        --threshold) while no wake word is present.
        Gate target: <= 1 FA / hour.

IMPORTANT -- provisional vs real gate 5
---------------------------------------
Run *now* (today) this harness is pointed at SYNTHETIC positives (Piper TTS) and
PUBLIC noise (AudioSet). The numbers it prints are therefore PROVISIONAL and are
NOT the real gate-5 result. The real gate 5 must be re-run with:
  * --positive-dir  = a folder of REAL team recordings of "Hey Jero"
  * --noise-wav/-dir = REAL hall / deployment-room noise (the long background stream)
The harness is identical either way -- only the input folders change. See the
"REAL GATE 5" section at the bottom for the exact re-run command.

Usage
-----
  python eval_gate5.py \
      --model models/hey_jero.onnx \
      --positive-dir models/train/hey_jero/positive_test \
      --noise-dir data/noise/audioset_16k \
      --threshold 0.5 --debounce 3
"""
from __future__ import annotations

import os
import glob
import json
import argparse

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
OWW_FRAME = 1280          # openWakeWord native step (80 ms) -> one score per frame


def _load_model(model_path: str):
    from openwakeword.model import Model
    m = Model(wakeword_models=[model_path], inference_framework="onnx", vad_threshold=0.0)
    key = os.path.splitext(os.path.basename(model_path))[0]
    return m, key


def _read_wav_16k(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    if sr != SAMPLE_RATE:
        # Resample defensively (eval audio should already be 16 kHz).
        import librosa
        data = (librosa.resample(data.astype(np.float32) / 32768.0,
                                 orig_sr=sr, target_sr=SAMPLE_RATE) * 32768.0).astype(np.int16)
    return data


def _peak_score_over_clip(model, key: str, pcm: np.ndarray, pad_ms: int = 500) -> float:
    # openWakeWord's classifier needs a full ~1.28 s (16-frame) context window. Short
    # isolated clips never fill it, so we pad with silence on each side to mirror both
    # training (clips padded to 2 s) and the always-on mic stream (continuous context).
    if pad_ms > 0:
        pad = np.zeros(int(pad_ms * SAMPLE_RATE / 1000), dtype=np.int16)
        pcm = np.concatenate([pad, pcm, pad])
    model.reset()
    peak = 0.0
    for i in range(0, len(pcm) - OWW_FRAME + 1, OWW_FRAME):
        s = model.predict(pcm[i:i + OWW_FRAME]).get(key, 0.0)
        peak = max(peak, float(s))
    return peak


def eval_miss_rate(model, key, positive_dir, threshold, pad_ms=500):
    clips = sorted(glob.glob(os.path.join(positive_dir, "*.wav")))
    if not clips:
        print(f"[miss-rate] no wavs in {positive_dir}"); return None
    misses, peaks = 0, []
    for c in clips:
        pk = _peak_score_over_clip(model, key, _read_wav_16k(c), pad_ms=pad_ms)
        peaks.append(pk)
        if pk < threshold:
            misses += 1
    n = len(clips)
    rate = misses / n
    print(f"[miss-rate] n={n}  misses={misses}  miss_rate={rate:.3%}  "
          f"(= {rate*10:.2f} per 10)   median_peak={np.median(peaks):.3f}  "
          f"p10_peak={np.percentile(peaks,10):.3f}")
    return {"n": n, "misses": misses, "miss_rate": rate,
            "misses_per_10": rate * 10, "median_peak": float(np.median(peaks))}


def eval_false_accepts(model, key, noise_paths, threshold, debounce):
    model.reset()
    total_samples = 0
    activations = 0
    above = 0
    for p in noise_paths:
        pcm = _read_wav_16k(p)
        total_samples += len(pcm)
        for i in range(0, len(pcm) - OWW_FRAME + 1, OWW_FRAME):
            s = model.predict(pcm[i:i + OWW_FRAME]).get(key, 0.0)
            if s >= threshold:
                above += 1
                if above >= debounce:
                    activations += 1
                    above = 0            # require a fresh run of frames for next FA
            else:
                above = 0
    hours = total_samples / SAMPLE_RATE / 3600.0
    fa_per_hour = activations / hours if hours > 0 else float("nan")
    print(f"[false-accepts] stream={hours:.3f} h  activations={activations}  "
          f"FA/hour={fa_per_hour:.3f}")
    return {"stream_hours": hours, "activations": activations, "fa_per_hour": fa_per_hour}


def main():
    ap = argparse.ArgumentParser(description="Gate 5 eval for Hey Jero wake word")
    ap.add_argument("--model", required=True)
    ap.add_argument("--positive-dir", required=True,
                    help="held-out positive wavs (synthetic now; REAL team clips for real gate 5)")
    ap.add_argument("--noise-dir", help="folder of noise wavs for the FA stream")
    ap.add_argument("--noise-wav", help="single long noise wav for the FA stream")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--debounce", type=int, default=3,
                    help="N consecutive 80 ms frames over threshold = one activation")
    ap.add_argument("--pad-ms", type=int, default=500,
                    help="silence padding each side of a positive clip so oww gets a full "
                         "1.28 s window (matches training + continuous mic). 0 = raw clip.")
    ap.add_argument("--max-noise-files", type=int, default=400,
                    help="cap noise files for a quick provisional run")
    ap.add_argument("--json-out", help="write metrics json here")
    ap.add_argument("--real", action="store_true",
                    help="label the run as the REAL gate 5 (real clips + real hall noise)")
    args = ap.parse_args()

    model, key = _load_model(args.model)
    tag = "REAL GATE 5" if args.real else "PROVISIONAL (synthetic positives + public noise)"
    print("=" * 70)
    print(f"Hey Jero -- Gate 5 eval  [{tag}]")
    print(f"model={args.model}  key={key}  threshold={args.threshold}  debounce={args.debounce}")
    print("=" * 70)

    miss = eval_miss_rate(model, key, args.positive_dir, args.threshold, pad_ms=args.pad_ms)

    fa = None
    noise_paths = []
    if args.noise_wav:
        noise_paths = [args.noise_wav]
    elif args.noise_dir:
        noise_paths = sorted(glob.glob(os.path.join(args.noise_dir, "*.wav")))[:args.max_noise_files]
    if noise_paths:
        fa = eval_false_accepts(model, key, noise_paths, args.threshold, args.debounce)
    else:
        print("[false-accepts] no noise provided; skipping FA metric")

    # Gate verdict
    print("-" * 70)
    miss_ok = miss is not None and miss["misses_per_10"] <= 1.0
    fa_ok = fa is not None and fa["fa_per_hour"] <= 1.0
    if miss is not None:
        print(f"  miss-rate gate (<=1/10): {'PASS' if miss_ok else 'FAIL'}  "
              f"({miss['misses_per_10']:.2f}/10)")
    if fa is not None:
        print(f"  FA/hour  gate (<=1/h) : {'PASS' if fa_ok else 'FAIL'}  "
              f"({fa['fa_per_hour']:.2f}/h)")
    if not args.real:
        print("  NOTE: PROVISIONAL numbers -- not the real-hall gate. Re-run with "
              "real team clips + real hall noise and --real.")
    print("=" * 70)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"tag": tag, "threshold": args.threshold, "debounce": args.debounce,
                       "miss": miss, "false_accepts": fa}, f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------- #
# REAL GATE 5 (run when real data is available):
#
#   python eval_gate5.py --real \
#       --model models/hey_jero.onnx \
#       --positive-dir data/real/team_hey_jero_clips \
#       --noise-wav   data/real/hall_noise_1hour.wav \
#       --threshold 0.5 --debounce 3 --json-out eval/real_gate5.json
#
# Gate passes when: miss-rate <= 1/10  AND  FA/hour <= 1.
# Tune --threshold (and --debounce) on the real mic audio to trade miss-rate
# against FA/hour before locking the deployed threshold into runtime.py.
# --------------------------------------------------------------------------- #
