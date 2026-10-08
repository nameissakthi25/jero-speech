# jero-speech

> **Status (2026-10-08):** STT, TTS, VAD, intent + wake word all built and benchmarked on-device.
> STT (Zipformer) and TTS (Supertonic-3) run **w8a16 on the Hexagon V68 HTP**; **intent (ModernBERT
> "Laya") now runs w8a16 on HTP at 100% via QAT** (PTQ was capped at 42%); VAD + wake word are
> **CPU by design**. Every model has `cpu/` + `htp/` folders under `models/`. Measured numbers +
> HTP-vs-CPU correlations: **see `BENCHMARKS.md`**.


Speech stack for the Jero robot: **wake word -> VAD -> STT -> intent -> TTS**, with
a safety-aware `say()` and a JSON event stream for the rest of the brain.

Target device: RB3 Gen2 (QCS6490, Ubuntu 24.04 aarch64, ~5.2 GB RAM). Always-on
parts run on CPU; ASR/intent can move to the Hexagon V68 NPU later. Developed and
verified on macOS (arm64); all deps have aarch64 wheels.

## What's REAL vs MOCKED (today)

| Component        | Status | Notes |
|------------------|--------|-------|
| **TTS** (Piper en_US-amy-medium, sherpa-onnx OfflineTts) | ✅ REAL | voice + espeak-ng-data downloaded; 8 canned lines pre-rendered; CPU num_threads=4 |
| **STT** (Zipformer, sherpa-onnx OnlineRecognizer) | ✅ REAL | our `zipformer-enin-qairt-aimet` model; greedy, CPU |
| **SpeechService** glue (queue, events, safety, fast-path, self-hearing) | ✅ REAL | full implementation, measured latencies below |
| **Intent** classifier (ModernBERT "Laya") | ✅ REAL | `intent.onnx` wired via `loaders.IntentModel`; fp32 CPU 100%/28 ms, **QAT w8a16 = 100% on V68 HTP**. Mock keyword fallback remains only if the file is absent. |
| **Wake word** "hey jero" (openWakeWord) | ✅ REAL | `hey_jero.onnx` present at `models/wakeword/cpu/`; loaded by `loaders.WakeWordDetector`. gate-5 (synthetic): 2.2% miss, 0 FA/hr. |
| **VAD** (Silero v4) | ✅ REAL | `silero_vad.onnx` present at `models/vad/cpu/`; loaded via sherpa-onnx `VoiceActivityDetector`. **Benchmarked: 94.5% acc, AUC 0.94, 0.078 ms/frame.** |
| **asr_conf** value | 🟡 PLACEHOLDER | 0.9 for any transcript (build has no token probs); field/shape final |

Each loader loads the **real** model when its file is present (all three are now) and sets
`.is_real`; a transparent mock/keyword fallback only kicks in if a file is missing, so the
end-to-end flow still runs on text in a bare checkout. See `BENCHMARKS.md` for measured numbers.

## Quick start (Mac dev)

```bash
pip install -r requirements.txt            # or reuse the asrenv venv
python -m brain.speech.tts --text "Hi, I'm Jero!" --out samples/hi.wav   # TTS smoke test
python -m brain.speech.prerender           # render canned lines to cache/canned/
python -m brain.speech --print             # end-to-end demo, prints JSON event lines
```

`--print` feeds, through the **real Zipformer STT**: the Zipformer `sample.wav`,
then three TTS-rendered command phrases ("Jero stop", "walk forward", "play the
xylophone") so you see every event type (`state`, `stop`, `intent`, `unclear`).

On device: `bash scripts/setup_rb3.sh`, then `systemctl enable --now jero-speech`.

## Public API (Devansh's spec)

```python
from brain.speech import SpeechService
svc = SpeechService("config.yaml")

for ev in svc.events():        # blocking generator of event dicts (JSON-serializable)
    handle(ev)

svc.say("Walking forward.")                 # queued, returns immediately
svc.say("Stopping.", priority="safety")     # preempts playback (<100 ms)
svc.stop_speaking()                         # clear queue + cut audio
```

### Event schema (JSON lines)
```
intent  {type,id,text,intent,slots,conf,asr_conf,t_end_of_speech,latency_ms:{asr,intent}}
stop    {type,id,text,source}
unclear {type,id,text,conf}
state   {type,listening,speaking}
```

## Spec requirement -> where it lives

| Requirement | Implementation |
|---|---|
| `SpeechService(config_path)` | `brain/speech/service.py` `SpeechService.__init__` |
| `events()` blocking generator of dicts | `service.py` `SpeechService.events` |
| `say(text, priority="normal")`, returns at once, queued | `service.py` `SpeechService.say` + `_play_worker` |
| `say(priority="safety")` cuts off playback | `service.py` `say()` safety branch -> `player.stop()`; **measured ~13 ms** |
| `stop_speaking()` | `service.py` `SpeechService.stop_speaking` |
| Priority queue, interrupt <100 ms, safety preempts | `service.py` `_pq`/`_cv` + `player.Player.stop` (SIGKILL on player proc) |
| "Stopping." safety line preempts | `config.yaml` `stop_line` + `say_stop_line_on_stop`; `_handle_asr` |
| "stop"/"Jero stop" fast-path, skip intent, <300 ms | `service.py` `_is_stop` + `_handle_asr` fast-path; **measured ~0.5 ms** |
| intent conf < 0.7 -> say "Sorry, say that again?" + emit `unclear` | `service.py` `_handle_asr` (`conf_threshold`, `unclear_line`) |
| Self-hearing: ignore mic while speaking + 200 ms | `service.py` `mic_muted` / `_set_speaking` + `self_hearing_tail_ms`; enforced in `run_mic_loop` + `feed_wav(respect_self_hearing=True)` |
| Event schemas (intent/stop/unclear/state) | `service.py` `_emit_intent/_emit_stop/_emit_unclear/_emit_state` |
| Clean loaders: wake / VAD / STT / intent | `brain/speech/loaders.py` (each exposes `.is_real` + a MOCK fallback) |
| `python -m brain.speech --print` | `brain/speech/__main__.py` |
| TTS setup (Piper medium voice) | `brain/speech/tts.py` + `models/tts/vits-piper-en_US-amy-medium/` |
| Pre-render canned lines | `brain/speech/prerender.py` -> `cache/canned/` |
| `config.yaml` / `models.json` / `jero-speech.service` / `setup_rb3.sh` | repo root + `scripts/` |

## Where the A100 models plug in (exact seams)

Drop the exported files and flip the config flag — no code change needed.

**Intent** (`loaders.IntentModel`):
- file: `models/intent/intent.onnx` (+ optional `models/intent/post_process.py` exposing `post_process(logits) -> (intent, slots, conf)`)
- config: `intent.model_path`, `intent.post_process_path`, set `intent.use_mock: false`
- then implement tokenization in `IntentModel.infer()` (marked `TODO(A100)`); output contract already fixed as `IntentResult(intent, slots, conf)`.

**Wake word** (`loaders.WakeWordDetector`):
- file: `models/wakeword/hey_jero.onnx` (openWakeWord)
- config: `wakeword.model_path`, `wakeword.threshold`
- interface stays `.process(frame) -> score` / `.fired(frame) -> bool`.

**VAD** (`loaders.VoiceActivityDetector`):
- file: `models/wakeword/silero_vad.onnx`
- config: `vad.model_path` + `vad.{threshold,min_silence_ms,min_speech_ms,max_speech_s}` (400 ms / 0.3-6 s)
- interface stays `.is_speech(frame) -> bool`.

All four loaders print `REAL` vs `MOCK` at startup and set `.is_real`, so the
service logs honestly what is live.

## Layout
```
jero-speech/
  brain/speech/        service.py loaders.py tts.py prerender.py player.py __main__.py
  models/tts/          Piper en_US-amy-medium (real)
  models/wakeword/     hey_jero.onnx, silero_vad.onnx  (land from A100/device)
  models/intent/       intent.onnx, post_process.py    (land from A100)
  cache/canned/        pre-rendered canned WAVs
  config.yaml  models.json  requirements.txt
  jero-speech.service  scripts/setup_rb3.sh
```
