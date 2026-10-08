# Jero Speech — Model Benchmarks (CPU vs HTP)

**Device:** RB3 Gen2 (QCS6490, Hexagon V68 HTP) · **Host:** macOS arm64 (CPU numbers) · **Updated:** 2026-10-08

Every model has a `cpu/` and an `htp/` folder under `models/`. This file collects the
measured numbers and the **correlation / agreement of HTP vs the CPU/float reference**.

> Honesty notes up front:
> - QCS6490 HTP is **w8a16 only** (int8 weights, int16 activations) — no fp32/fp16 on the NPU.
> - **Sim ≠ device.** All HTP numbers below are real on-device runs, not simulator.
> - Some HTP cells are **N/A by design** (VAD, wake word) or **deferred** (intent) — explained per model.

---

## Summary table

| Model | CPU | HTP (w8a16, real V68) | HTP vs CPU |
|---|---|---|---|
| **STT** (Zipformer, en-IN) | float WER 28.7% | w8a16 WER **27.0%** (same utts) | **~lossless** (HTP ≈ float, sometimes cleaner) |
| **TTS** (Supertonic-3) | ONNX Runtime fp32 reference | full 4-model w8a16 on HTP (M1 audio) | **log-mel corr 0.77** (DTW-aligned) |
| **VAD** (Silero v4) | acc 94.5%, AUC 0.94, 0.078 ms/frame | **N/A — CPU-only by design** | — |
| **Intent** (ModernBERT, "Laya") | fp32 100% / 28 ms | **QAT w8a16 = 100%** (120/120) | **PTQ 42% → QAT 100% on-device** |
| **Wake word** (openWakeWord) | always-on, provisional gate-5 pass | **N/A — always-on CPU by design** | — |

> **Honesty on the numbers:** STT/TTS/Intent HTP were run on a QDC RB3 Gen2 (V68, soc_id 498). STT CPU-vs-HTP uses the *same* Svarah utterances through an identical decoder (float-host vs w8a16-device). Intent 100% = w8a16 matches fp32 on a **held-out synthetic** test where fp32 is itself ~100% — it proves QAT removed the on-device quantization loss, **not** real-world accuracy (real gate needs ≥200 real recordings, still open). Slot head not yet scored on-device.

---

## STT — Zipformer (Indian-English, streaming RNN-T)

- **CPU vs HTP on identical data (the clean comparison)** — 15 Svarah Indian-English utterances, same pre-extracted fbank, same numpy greedy decoder; CPU = float ONNX on host, HTP = w8a16 ctx bins on the QDC V68:
  - **CPU float WER 28.7%**, **HTP w8a16 WER 27.0%** — within noise, and HTP is *sometimes cleaner* ("ERROR OF GLOBALIZATION" vs CPU's garbled "ERRISATION"). **w8a16 costs ~nothing.**
  - Drivers: `models/stt/htp/stt_htp.py` (device) and `models/stt/cpu/cpu_float_decode.py` (host). Bench: `bench/stt_cpu_vs_htp_device.json`.
  - The ~28% is on **hard long-form** sentences; short command words decode near-perfectly. Earlier token-exact check: `sample.wav` → "CHANGE LANGUAGE TO HINDI" (identical to float).
  - On-device RTF here is inflated by **context-reload-per-call**; the persistent-context deployment is ~0.05 (enc ~16.4 ms/chunk).
- **Verdict:** w8a16 STT on HTP is **lossless vs float**. Ship on HTP.

## TTS — Supertonic-3 (4 ONNX models: text-encoder / duration / vector-estimator / vocoder)

- **CPU:** Supertonic-3 fp32 ONNX pipeline is the reference (`cpu_cf_M1.wav`, seed 1000). Piper `en_US-amy-medium` is the lightweight CPU fallback the service also ships.
- **HTP (w8a16, all 4 models on V68, QDC device):**
  - **Full pipeline runs on HTP** and produces audio (M1 voice, chunking for long text, numpy spectral de-noiser + RMS loudness post-proc baked into `models/tts/htp/board_driver.py`).
  - **Fidelity vs CPU-fp32** (same noise seed; metric = DTW-aligned, per-mel-normalized log-mel correlation, because flow-matching is stochastic and durations differ):
    - raw HTP vs fp32: **0.754**
    - shipped (de-noised + loudness) vs fp32: **0.772**  (method sanity: cpu-vs-cpu = 1.000)
  - **Graph-surgery cost alone** (host-embed + erf-approx + −1e4 mask, fp32-vs-fp32, from build validation): log-mel corr **1.000**, MCD **2.15 dB** — negligible; the gap to 0.77 is the **w8a16 quantization** of the generative vocoder.
  - **Latency:** ~1.3× RTF cold per invocation (reloads context each call); in a persistent service the context loads once. 5 M1 demo lines (2.4–4.5 s audio) render in 4–8 s wall.
  - **Perceptual:** intelligible, playful robot voice; **small residual buzz** remains (w8a16 on a flow-matching vocoder is the known quality risk). sqnr-calibrated fallback DLCs are staged if we want to chase it further.
- **Verdict:** w8a16 TTS on HTP is **demo-ready** with a minor residual buzz; 0.77 log-mel corr is expected for a stochastic generative model quantized to int.

## VAD — Silero (v4)

- **CPU:** acc **94.5%**, precision **1.00**, recall 0.78 (lower bound†), **ROC-AUC 0.94**; mean per-frame latency **0.078 ms**, **RTF 0.0024** (~410× real-time). 8,736 speech frames (40 LibriSpeech clips) vs 26,208 non-speech frames (silence + white + pink noise). Zero false positives.
  - †recall is a floor: whole speech clips were labeled speech=1, but they contain internal/edge silence that the VAD correctly calls non-speech → counted as misses. True speech recall is higher.
- **HTP: N/A by design.** Silero is **1.7 MB and already sub-millisecond on CPU** (410× RT); there is no latency or power case for putting it on the NPU, and it must run always-on to gate the pipeline. **VAD stays on CPU.**

## Intent — ModernBERT-base, 2-head ("Laya")

12-class intent (walk/turn/stop/dance/look/greet/emote/yes/no/cancel/chit_chat/none) + 8-slot BIO head.

- **CPU (fp32):** mean **27.6 ms** (<100 ms gate), 10/10 intent+slots on Jero command lines, conf ~1.0.
- **HTP (w8a16) — SOLVED via QAT, verified on real V68:** **100% top-1** (120/120 vs fp32 and vs gold, 0 missing, all 12 classes used). PTQ was capped at **42.5%**; QAT against a **device-faithful ~3-bit-attention sim** recovered it (45.8% → 96.7%@100 → 100%@300 steps). Full-graph w8a16 DLC, context built on-board, 120 inputs in ~3 s. Recipe: `conversion/intent_laya/qat/`; bench: `bench/intent_htp_qat_device.json`.
- **Caveat:** 100% = w8a16 matches fp32 on **held-out synthetic** data (fp32 itself ~100% there) — proves the quantization loss is gone on-device, not real-world accuracy. Slot head not yet scored on-device.
- **Verdict:** intent can now **ship on HTP at 100%** (CPU fp32 remains a 28 ms fallback).

## Wake word — openWakeWord "Hey Jero"

- **CPU:** 205 KB model, always-on. Provisional gate-5 (synthetic + AudioSet noise): miss 2.2%, 0 false-accepts/hr.
- **HTP: N/A by design** — tiny, must be always-on; push-to-talk is primary at the event.

---

## Reproduce

- STT CPU: `scratchpad/bench_stt_cpu.py` → `bench_stt_cpu.json`
- VAD CPU: `scratchpad/bench_vad_cpu.*`
- Intent CPU: loader path `brain/speech/loaders.py::IntentModel` on `models/intent/cpu/`
- TTS HTP: `models/tts/htp/board_driver.py` on the RB3/QDC device (ctx bins built on-board)
- TTS corr: DTW-aligned log-mel correlation vs `cpu_cf_M1.wav` (seed 1000)
