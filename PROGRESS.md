# Jero Speech Stack — Build Log & Status

**Owner:** Sakthi · **Target:** RB3 Gen2 (QCS6490 / Hexagon V68) · **Event:** Oct 10, 2026
**Last updated:** 2026-10-08

Sakthi delivers the `jero-speech` service: mic → wake/PTT → VAD → STT → intent → events, plus `say()` TTS.
Emits intent events; never sends robot commands (Dev's skill executor consumes the events).

---

## Component status

| Component | Model | Runtime decision | Status |
|---|---|---|---|
| **STT** | Zipformer Indian-English (70M) | **HTP (NPU) w8a16** | ✅ **Decodes on HTP** — verified |
| **Intent (Laya)** | ModernBERT-base, 2-head (12 intents + 17 BIO slots) | **CPU fp32** (HTP hit a fidelity ceiling) | ✅ built + wired (CPU) |
| **TTS** | Piper `en_US-amy-medium` via sherpa-onnx | CPU | ✅ built |
| **Wake word** | openWakeWord custom "Hey Jero" | CPU (always-on) | ✅ built, provisional gate-5 pass |
| **VAD** | Silero (MIT) | CPU | ✅ built |
| **Glue** | `jero-speech` SpeechService | CPU | ✅ built, runs end-to-end (Mac) |

Everything runs end-to-end today via `python -m brain.speech --print` (real STT + intent + wake + VAD + TTS).

---

## Key results

### STT on HTP — ✅ WORKS
- Zipformer encoder + decoder + joiner all execute **w8a16 on the V68 HTP**.
- Verified transcription of `samples/sample.wav` on the board: **"CHANGE LANGUAGE TO HINDI"** — matches the float sherpa-onnx reference **exactly** (token ids `[804,1529,37,23,217,1949]`).
- True on-HTP encoder compute ≈ **16.4 ms/chunk**, encoder RTF ≈ **0.05** (context load ~67 ms one-time). Only fbank + the greedy loop on CPU.
- Runner: pure-Python, no app/compilation — drives the ctx bins via `qnn-net-run`. On board at `~/zipformer_run/htp_decode.py` (`python3 htp_decode.py`).
- ctx bins: `~/zipformer_test/{encoder,decoder,joiner}_w8a16_ctx.bin.bin`.
- Non-obvious detail solved: quantized encoder's input states are transposed vs output states (unify shares encodings pairwise) — transpose between chunks.

### Intent on HTP — ❌ fidelity ceiling → running on CPU
- ModernBERT-base w8a16 **builds and runs cleanly on the HTP** (context binary, spill_bytes=0, loads, executes 120 utts) but **on-device top-1 agreement vs fp32 ≈ 41%** — fails the ≥99% gate.
- Two serious attempts both landed ~41–42%:
  1. Agent's direct export (input_ids + heads in-graph) → 42.5%.
  2. **Laya-recipe split** (embeddings + mask bias on host, encoder body only on NPU, mask surgery −3.4e38→−1e4, min-max w8a16) → **40.83%**.
- **Root cause: quantization-fidelity ceiling.** Laya's own on-device number is hidden-state Pearson ≈ **0.88** at w8a16 on this exact chip. That's enough for Laya's *coarse typed-decision* (reads a couple of marker tokens), but a **12-way intent classifier on mean-pooled hidden states is far more sensitive** — r≈0.88 flips many argmaxes → ~41%. NOT a recipe bug.
- Sim lies: host AIMET/ORT sim showed 98–100% (runs attn matmuls in float); real HTP int8 execution is the true ~41%. **Always verify on-device.**
- **Decision (2026-10-08): run intent on CPU** — fp32, full accuracy, ~13–18 ms (< 100 ms gate), spec-sanctioned fallback ("ModernBERT on CPU with ONNX Runtime if it meets 100 ms"). HTP-intent deferred (levers left: SQNR calibration, or QAT).
- Intent accuracy (synthetic test, CPU): intent 100% / slot F1 1.00 / none-leak 0% — **but saturated** (synthetic in-distribution); real numbers need real recordings.

### Wake word — ✅ provisional pass
- openWakeWord "Hey Jero" (205 KB) + Silero VAD. Trained on ~4k synthetic positives + adversarial negatives + RIR/noise aug (A100).
- Provisional gate-5 (synthetic + public AudioSet noise, with proper context padding): **miss 2.2%, 0 false-accepts/hr** — passes (≤10% miss, ≤1 FA/hr). NOT the real-hall gate.
- Push-to-talk is PRIMARY at the event; wake word is the gated 2nd mode.

### TTS / VAD — ✅
- Piper `en_US-amy-medium` via sherpa-onnx; canned lines pre-rendered (<50 ms). ~RTF 0.12–0.25 on RB3 big cores.
- Silero VAD: 400 ms end-silence, 0.3–6 s utterance.

---

## Key technical lessons (reusable)

1. **ModernBERT w8a16 mask surgery:** the attention mask fills masked positions with FP32 `finfo.min` (−3.4e38); int16 min-max quant blows the activation range apart → collapse. Fix: replace with −1e4 (softmax ≈ 0 either way, FP32-lossless). `mask_surgery.py`.
2. **V68 split recipe (from Laya):** V68 silently returns wrong rows for on-chip integer `Gather` → do the embedding lookup on the HOST (feed `inputs_embeds`); also move the attention-mask→additive-bias to the host. Only the encoder body runs on the NPU; heads stay host-float.
3. **Sim ≠ device:** AIMET/ORT sim runs attention activation×activation matmuls in float and over-reports fidelity; verify on real hardware.
4. **w8a16 fidelity ceiling (~r0.88) for ModernBERT on v68:** fine for coarse decisions, insufficient for strict multi-class top-1.
5. **Flaky board transfers:** use chunked `scp` / `scp -r` (stream file-by-file), never a big tar.gz extracted on the 5 GB board (OOM/crash). Retry with backoff; board auth flakes under load.
6. **STT-on-HTP without the C++ app:** pure-Python runner driving ctx bins via `qnn-net-run` + numpy quant/dequant works; no QNN headers/bindings needed on the board.

---

## Artifact locations

- **Repo / service:** `/Users/nameissakthi/Desktop/personal/jero-speech/` (brain/speech, models/, config.yaml, setup_rb3.sh, jero-speech.service).
- **A100** (`raemox@136.112.189.105`): intent fine-tune + ONNX `~/jero/intent/export/intent.onnx`; w8a16 HTP work `~/jero/intent/qnn/laya_split/` (encoder split + DLC); wake word `~/jero/wakeword/` (hey_jero.onnx). QAIRT 2.37 `~/qairt/2.37.1.250807`, AIMET `~/qhub45`, converter venv `~/qenv`.
- **RB3** (`ubuntu@192.168.100.33` pw <REDACTED_BOARD_PW>): STT HTP runner `~/zipformer_run/`, ctx bins `~/zipformer_test/`, intent HTP binary `~/intent_htp/intent_encoder_w8a16.bin` (built, ~41% — not deployed).
- **Zipformer repo:** `/Users/nameissakthi/Desktop/personal/zipformer-qcs6490/zipformer-enin-qairt-aimet/`.
- **Laya recipe reference:** `/Users/nameissakthi/Desktop/personal/laya-qcs6490/` (RECIPE.md, make256.py, pb256.py).

---

## Gates

| Gate | Status |
|---|---|
| STT decode on HTP | ✅ verified (exact match to float) |
| Intent offline (synthetic) | ✅ 100% but SATURATED (synthetic) |
| Intent offline (real) | ⬜ needs ≥200 real recordings |
| On-device intent latency / ≥99% int8 | N/A on CPU (full accuracy); HTP deferred |
| Wake word gate-5 (synthetic) | ✅ provisional (2.2% miss / 0 FA) |
| Wake word gate-5 (real hall) | ⬜ needs real mic + hall noise |
| End-to-end on RB3 (all models) | ⬜ pending (see Next) |

---

## Pending / Next

1. **Run the full `jero-speech` on the RB3** — STT on HTP (python runner) + intent/wake/VAD/TTS on CPU. Deploy via `setup_rb3.sh`, run `python -m brain.speech --print` on the board.
2. **Real recordings** (≥200, 5+ speakers, event mic, hall noise) → real gate-3 (intent) + real wake-word gate-5. Public data can cover noise + a proxy test (MUSAN/SLURP/Speech Commands/MASSIVE en-IN), but "Hey Jero" positives + Jero-command takes need a short human recording session.
3. **Slot head** on host (only intent head extracted so far; slot MatMul weight is an `onnx::MatMul_*` initializer).
4. **Integration** with Dev's skill executor + MuJoCo twin (20 scripted commands, 3 speakers).
5. **Open decisions:** language = English v1 (confirm); wake-word-at-event gated on real gate-5.
6. **(Deferred) Intent on HTP:** SQNR calibration round, or QAT.
