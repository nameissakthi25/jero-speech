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

| Model | CPU runtime | CPU result | HTP (w8a16) | HTP vs CPU correlation / agreement |
|---|---|---|---|---|
| **STT** (Zipformer, en-IN) | sherpa-onnx, 4 threads | WER 34.8%* , RTF 0.037 | runs on V68, enc ~16.4 ms/chunk, RTF ~0.05 | **token-exact** (1.000) vs float decode |
| **TTS** (Supertonic-3) | ONNX Runtime fp32 | reference pipeline | full 4-model w8a16 on HTP | **log-mel corr 0.77** (DTW-aligned) |
| **VAD** (Silero v4) | onnxruntime, 1 thread | acc 94.5%, AUC 0.94, 0.078 ms/frame | **N/A — CPU-only by design** | — |
| **Intent** (ModernBERT, "Laya") | onnxruntime fp32 | 27.6 ms, 10/10 on Jero lines | builds+runs, top-1 **~41%** | hidden-state r≈0.88 → **deferred to CPU** |
| **Wake word** (openWakeWord) | onnxruntime, always-on | provisional gate-5 pass | **N/A — always-on CPU by design** | — |

*WER is a **cross-accent proxy** (Indian-English model scored on US-English LibriSpeech), not an in-domain number.

---

## STT — Zipformer (Indian-English, streaming RNN-T)

- **CPU** (`hf-internal-testing/librispeech_asr_dummy`, clean/validation, 73 clips, sherpa-onnx greedy, 4 threads):
  - **WER 34.8%** (proxy), **RTF 0.037** (~27× real-time), 481 s audio in 17.6 s wall.
  - Caveat: the model is trained for **Indian English**; LibriSpeech is US English, so this over-states error vs the target accent. It is a fast sanity proxy, not the event gate.
- **HTP** (verified on RB3 board, `~/zipformer_test/*_w8a16_ctx.bin`):
  - Encoder + decoder + joiner all execute **w8a16 on V68**.
  - **Token-exact match to the float reference**: `sample.wav` → "CHANGE LANGUAGE TO HINDI", ids `[804,1529,37,23,217,1949]` — **correlation 1.000** (identical token stream).
  - On-HTP encoder ≈ **16.4 ms/chunk**, encoder RTF ≈ **0.05** (one-time ~67 ms context load). fbank + greedy loop stay on CPU.
- **Verdict:** STT is the one model where w8a16 on HTP is **bit-faithful enough to be identical** to float decoding. Ship on HTP.

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

- **CPU (fp32, deployed):** mean **27.6 ms** (median 27.6, max 28.3) — well under the 100 ms gate. 10/10 correct intent + slots on Jero command lines (walk/turn/dance/stop/greet, with direction + steps slots), confidence ~1.0.
- **HTP (w8a16):** builds and runs cleanly on V68 (context binary, spill=0), but **on-device top-1 agreement vs fp32 ≈ 41%** — fails the ≥99% gate. Root cause is a **w8a16 fidelity ceiling**: hidden-state Pearson **r≈0.88**, fine for Laya's coarse typed decisions but not for a 12-way argmax on mean-pooled states. Two serious recipes both landed ~41%.
- **Verdict:** **run intent on CPU** (fp32, full accuracy, 28 ms). HTP path documented and deferred (levers: sqnr calibration, QAT).

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
