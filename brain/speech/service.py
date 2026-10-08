"""
SpeechService — the jero-speech public API (Devansh's spec).

Public surface
--------------
    svc = SpeechService(config_path)
    for ev in svc.events():            # blocking generator of event dicts
        ...
    svc.say("Walking forward.")                      # queued, returns at once
    svc.say("Stopping.", priority="safety")          # preempts playback <100 ms
    svc.stop_speaking()                              # clear queue + cut audio

Event schema (also what `python -m brain.speech --print` emits as JSON lines)
    intent  : {type,id,text,intent,slots,conf,asr_conf,t_end_of_speech,
               latency_ms:{asr,intent}}
    stop    : {type,id,text,source}
    unclear : {type,id,text,conf}
    state   : {type,listening,speaking}

Pipeline rules implemented here
    * priority queue; "safety" preempts normal playback within 100 ms
    * "stop"/"Jero stop" fast-path: STOP emitted straight from ASR text,
      intent model skipped, inside 300 ms
    * intent conf < threshold (0.7) -> service says "Sorry, say that again?"
      itself and emits `unclear`
    * self-hearing guard: mic ignored while speaking + 200 ms after

Audio input seams
    * feed_wav(path)   -> real Zipformer STT on a file (used by the demo today)
    * run_mic_loop()   -> live wake-word + VAD + STT loop (wired; wake/VAD mocked
                          until hey_jero.onnx / silero_vad.onnx land on device)
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .loaders import ASRResult, IntentModel, STTEngine, VoiceActivityDetector, WakeWordDetector
from .player import Player
from .prerender import DEFAULT_CANNED, cache_path, prerender
from .tts import PiperTTS, ROOT

log = logging.getLogger("jero.speech.service")

_EV_SENTINEL = object()


@dataclass(order=False)
class _Utt:
    text: str
    priority: str  # "safety" | "normal"


def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


class SpeechService:
    SAFETY = "safety"
    NORMAL = "normal"

    def __init__(self, config_path: str = None, config: dict = None, prerender_on_start: bool = True):
        if config is None:
            config_path = config_path or str(ROOT / "config.yaml")
            config = _load_yaml(config_path)
        self.cfg = config

        # --- thresholds / params ---
        self.conf_threshold = float(config.get("intent", {}).get("conf_threshold", 0.7))
        self.self_hearing_tail_ms = int(config.get("self_hearing_tail_ms", 200))
        self.stop_words = [w.lower() for w in config.get("stop_words",
                           ["stop", "jero stop", "halt", "freeze"])]
        self.unclear_line = config.get("unclear_line", "Sorry, say that again?")
        self.stop_line = config.get("stop_line", "Stopping.")
        self.say_stop_line_on_stop = bool(config.get("say_stop_line_on_stop", True))

        # --- events out ---
        self._events: "queue.Queue" = queue.Queue()

        # --- playback queue (custom: supports safety preemption + drain) ---
        self._pq: list = []
        self._cv = threading.Condition()
        self._seq = 0
        self._shutdown = threading.Event()

        # --- speaking / listening state ---
        self._speaking = False
        self._listening = False
        self._mute_until = 0.0
        self._state_lock = threading.Lock()

        # --- models (loaders decide real vs mock) ---
        stt_cfg = config.get("stt", {})
        stt_dir = stt_cfg.get("model_dir")
        if stt_dir and not os.path.isabs(stt_dir):
            stt_dir = str(ROOT / stt_dir)
        self.stt = STTEngine(stt_dir or "", num_threads=stt_cfg.get("num_threads", 4),
                             provider=stt_cfg.get("provider", "cpu"),
                             decoding_method=stt_cfg.get("decoding_method", "greedy_search"))

        intent_cfg = config.get("intent", {})
        ip = intent_cfg.get("model_path")
        if ip and not os.path.isabs(ip):
            ip = str(ROOT / ip)
        pp = intent_cfg.get("post_process_path")
        if pp and not os.path.isabs(pp):
            pp = str(ROOT / pp)
        self.intent = IntentModel(ip, pp, use_mock=intent_cfg.get("use_mock", True))

        ww_cfg = config.get("wakeword", {})
        wp = ww_cfg.get("model_path")
        if wp and not os.path.isabs(wp):
            wp = str(ROOT / wp)
        self.wake = WakeWordDetector(wp, threshold=ww_cfg.get("threshold", 0.5))
        # trigger mode: "vad" (default, no wake word), "ptt", or "wakeword"
        self.trigger = str(config.get("trigger", "vad")).lower()
        self._ptt = threading.Event()   # set/cleared by a PTT button handler

        vad_cfg = config.get("vad", {})
        vp = vad_cfg.get("model_path")
        if vp and not os.path.isabs(vp):
            vp = str(ROOT / vp)
        self.vad = VoiceActivityDetector(vp, params=vad_cfg)

        # --- TTS + canned cache ---
        self.tts = None
        self._canned: dict = {}
        cache_dir = config.get("tts", {}).get("cache_dir", str(ROOT / "cache" / "canned"))
        if not os.path.isabs(cache_dir):
            cache_dir = str(ROOT / cache_dir)
        self._cache_dir = cache_dir
        try:
            self.tts = PiperTTS.from_config(config)
            if prerender_on_start:
                lines = config.get("canned_lines", DEFAULT_CANNED)
                self._canned = prerender(self.tts, lines, cache_dir)
        except Exception as e:
            log.warning("TTS init/prerender failed (%s); say() will be text-only", e)

        # --- player + worker thread ---
        self.player = Player()
        self._worker = threading.Thread(target=self._play_worker, name="jero-tts", daemon=True)
        self._worker.start()
        log.info("SpeechService ready (STT real=%s, intent real=%s, wake real=%s, vad real=%s)",
                 self.stt.is_real, self.intent.is_real, self.wake.is_real, self.vad.is_real)
        self._emit_state()

    # ===================================================================== #
    # public API
    # ===================================================================== #
    def events(self) -> Iterator[dict]:
        """Blocking generator of event dicts. Ends when close() is called."""
        while True:
            ev = self._events.get()
            if ev is _EV_SENTINEL:
                return
            yield ev

    def say(self, text: str, priority: str = "normal") -> None:
        """Queue a line for playback. Returns immediately. safety preempts."""
        utt = _Utt(text=text, priority=priority)
        with self._cv:
            self._seq += 1
            if priority == self.SAFETY:
                # Drop pending normal chatter and cut current playback.
                self._pq = [it for it in self._pq if it[0] == 0]
                self.player.stop()          # kill current audio (<100 ms)
                self._pq.append((0, self._seq, utt))
            else:
                self._pq.append((1, self._seq, utt))
            self._pq.sort(key=lambda it: (it[0], it[1]))
            self._cv.notify()

    def stop_speaking(self) -> None:
        """Clear the queue and cut current audio."""
        with self._cv:
            self._pq.clear()
            self.player.stop()
            self._cv.notify()

    def close(self) -> None:
        self._shutdown.set()
        with self._cv:
            self._cv.notify_all()
        self.player.stop()
        self._events.put(_EV_SENTINEL)

    # ===================================================================== #
    # audio input seams
    # ===================================================================== #
    def feed_wav(self, wav_path: str, respect_self_hearing: bool = False) -> Optional[dict]:
        """
        Run real Zipformer STT on a WAV file, then process the transcript through
        the pipeline. Returns the primary event emitted (also pushed to events()).
        Used by the demo; on-device the same path runs on a VAD-segmented buffer.
        """
        if respect_self_hearing and self.mic_muted():
            log.info("feed_wav ignored: self-hearing guard active")
            return None
        import soundfile as sf
        import numpy as np
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav[:, 0]
        if sr != 16000:
            wav = _resample_to_16k(wav, sr)
        asr = self.stt.transcribe_samples(np.asarray(wav, dtype="float32"), 16000)
        return self._handle_asr(asr)

    def feed_samples(self, samples, sample_rate: int = 16000) -> Optional[dict]:
        asr = self.stt.transcribe_samples(samples, sample_rate)
        return self._handle_asr(asr)

    def run_mic_loop(self):  # pragma: no cover - device-only
        """
        Live always-on loop (RB3). Wired end-to-end; wake + VAD are MOCK until
        hey_jero.onnx / silero_vad.onnx are on device. See loaders.py TODOs.

        Loop: wait for wake word -> open VAD window (400 ms silence, 0.3-6 s) ->
        collect speech -> STT -> pipeline. Mic frames dropped while mic_muted().
        """
        try:
            import sounddevice as sd
            import numpy as np
        except Exception as e:
            raise RuntimeError(f"mic loop needs sounddevice+numpy: {e}")
        sr = 16000
        frame = int(0.03 * sr)  # 30 ms
        self._set_listening(True)
        with sd.InputStream(channels=1, samplerate=sr, dtype="float32") as mic:
            buf = []
            in_speech = False
            silence = 0.0
            while not self._shutdown.is_set():
                data, _ = mic.read(frame)
                chunk = data[:, 0]
                if self.mic_muted():          # self-hearing guard
                    buf, in_speech, silence = [], False, 0.0
                    continue
                # start-of-utterance gate (trigger mode)
                if not in_speech:
                    if self.trigger == "wakeword":
                        if not self.wake.fired(chunk):
                            continue
                    elif self.trigger == "ptt":
                        if not self._ptt.is_set():
                            continue
                    else:  # "vad" (default): start capture on speech onset
                        if not self.vad.is_speech(chunk):
                            continue
                in_speech = True
                buf.append(chunk)
                if self.vad.is_speech(chunk):
                    silence = 0.0
                else:
                    silence += 0.03
                dur = len(buf) * 0.03
                if (silence >= self.cfg.get("vad", {}).get("min_silence_ms", 400) / 1000.0
                        and dur >= self.cfg.get("vad", {}).get("min_speech_ms", 300) / 1000.0) \
                        or dur >= self.cfg.get("vad", {}).get("max_speech_s", 6):
                    self.feed_samples(np.concatenate(buf), sr)
                    buf, in_speech, silence = [], False, 0.0

    # --- push-to-talk control (for trigger: "ptt") -------------------------
    def start_push_to_talk(self):
        """Call on PTT button press (ptt trigger mode)."""
        self._ptt.set()

    def stop_push_to_talk(self):
        """Call on PTT button release."""
        self._ptt.clear()

    # ===================================================================== #
    # pipeline core
    # ===================================================================== #
    def _handle_asr(self, asr: ASRResult) -> Optional[dict]:
        text = (asr.text or "").strip()
        if not text:
            log.info("empty transcript; nothing to do")
            return None

        # --- fast-path STOP: skip intent model, emit straight from ASR text ---
        if self._is_stop(text):
            ev = self._emit_stop(text, source="asr_fastpath")
            if self.say_stop_line_on_stop:
                self.say(self.stop_line, priority=self.SAFETY)  # preempts playback
            return ev

        # --- intent model (mock today) ---
        t0 = time.time()
        ir = self.intent.infer(text, asr.conf)
        intent_latency = (time.time() - t0) * 1000.0

        if ir.conf < self.conf_threshold:
            self.say(self.unclear_line, priority=self.NORMAL)   # service speaks itself
            return self._emit_unclear(text, ir.conf)

        return self._emit_intent(asr, ir, intent_latency)

    def _is_stop(self, text: str) -> bool:
        t = text.lower().strip().rstrip(".!?,")
        for w in self.stop_words:
            if t == w or t.endswith(" " + w) or t.startswith(w + " ") or w in t.split():
                return True
        return False

    # ===================================================================== #
    # event emitters
    # ===================================================================== #
    def _emit(self, ev: dict) -> dict:
        self._events.put(ev)
        return ev

    @staticmethod
    def _id() -> str:
        return "evt-" + uuid.uuid4().hex[:12]

    def _emit_intent(self, asr: ASRResult, ir, intent_latency_ms: float) -> dict:
        return self._emit({
            "type": "intent",
            "id": self._id(),
            "text": asr.text,
            "intent": ir.intent,
            "slots": ir.slots,
            "conf": round(ir.conf, 3),
            "asr_conf": round(asr.conf, 3),
            "t_end_of_speech": round(asr.t_end_of_speech, 3),
            "latency_ms": {"asr": round(asr.latency_ms, 1),
                           "intent": round(intent_latency_ms, 1)},
        })

    def _emit_stop(self, text: str, source: str) -> dict:
        return self._emit({"type": "stop", "id": self._id(), "text": text, "source": source})

    def _emit_unclear(self, text: str, conf: float) -> dict:
        return self._emit({"type": "unclear", "id": self._id(), "text": text,
                           "conf": round(conf, 3)})

    def _emit_state(self) -> dict:
        with self._state_lock:
            return self._emit({"type": "state", "listening": self._listening,
                               "speaking": self._speaking})

    # ===================================================================== #
    # state helpers (speaking/listening + self-hearing mute)
    # ===================================================================== #
    def _set_speaking(self, flag: bool):
        changed = False
        with self._state_lock:
            if self._speaking != flag:
                self._speaking = flag
                changed = True
                if not flag:
                    # start the 200 ms self-hearing tail now
                    self._mute_until = time.time() + self.self_hearing_tail_ms / 1000.0
        if changed:
            self._emit_state()

    def _set_listening(self, flag: bool):
        changed = False
        with self._state_lock:
            if self._listening != flag:
                self._listening = flag
                changed = True
        if changed:
            self._emit_state()

    def mic_muted(self) -> bool:
        with self._state_lock:
            return self._speaking or time.time() < self._mute_until

    # ===================================================================== #
    # playback worker
    # ===================================================================== #
    def _resolve_wav(self, text: str) -> Optional[str]:
        # cached canned line?
        p = cache_path(self._cache_dir, text)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return str(p)
        # synth on the fly (and cache)
        if self.tts is not None:
            try:
                self.tts.synthesize_to_wav(text, p)
                return str(p)
            except Exception as e:
                log.warning("on-the-fly TTS failed for %r (%s)", text, e)
        return None

    def _play_worker(self):
        while not self._shutdown.is_set():
            with self._cv:
                while not self._pq and not self._shutdown.is_set():
                    self._cv.wait(timeout=0.2)
                if self._shutdown.is_set():
                    return
                _, _, utt = self._pq.pop(0)
            wav = self._resolve_wav(utt.text)
            self._set_speaking(True)
            if wav:
                self.player.play_blocking(wav)
            else:
                log.info("[say/%s] %s (no audio backend/voice; text-only)", utt.priority, utt.text)
            self._set_speaking(False)


def _resample_to_16k(wav, sr):
    import numpy as np
    if sr == 16000:
        return wav
    ratio = 16000 / float(sr)
    n = int(len(wav) * ratio)
    x_old = np.linspace(0, 1, num=len(wav), endpoint=False)
    x_new = np.linspace(0, 1, num=n, endpoint=False)
    return np.interp(x_new, x_old, wav).astype("float32")
