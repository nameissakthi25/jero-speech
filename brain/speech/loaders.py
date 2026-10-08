"""
Clean loader interfaces for every model the speech pipeline touches.

Design goal (per spec): the intent model and wake-word model are built by other
agents on the A100 and exported as ONNX. This module defines the *exact* seams
where those files plug in. Until a real file is present on disk, each loader
falls back to a Mock implementation with a clear TODO, so the whole pipeline
runs end-to-end on text today.

Four seams:
  1. WakeWordDetector  -> openWakeWord ONNX   (~/jero/wakeword/hey_jero.onnx)   [MOCK]
  2. VoiceActivityDetector -> Silero VAD ONNX (~/jero/wakeword/silero_vad.onnx) [MOCK]
  3. STTEngine        -> Zipformer sherpa-onnx OnlineRecognizer                 [REAL]
  4. IntentModel      -> intent ONNX + post_process (~/jero/intent/)            [MOCK]

Every loader exposes `.is_real` so the service can log honestly what is live.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

log = logging.getLogger("jero.speech.loaders")


# --------------------------------------------------------------------------- #
# Result dataclasses (stable across mock/real so callers never special-case)
# --------------------------------------------------------------------------- #
@dataclass
class ASRResult:
    text: str
    conf: float                 # 0..1 mean token confidence (heuristic for now)
    t_end_of_speech: float      # wall-clock epoch seconds when speech ended
    latency_ms: float           # decode wall time


@dataclass
class IntentResult:
    intent: str                 # e.g. "move", "turn", "dance", "stop", "none"
    slots: dict = field(default_factory=dict)
    conf: float = 0.0


# --------------------------------------------------------------------------- #
# 1. Wake word — openWakeWord ("hey jero") — MOCK until A100 exports hey_jero.onnx
# --------------------------------------------------------------------------- #
class WakeWordDetector:
    """
    Real interface (openWakeWord): feed 16 kHz float32 frames via `.process(frame)`,
    returns a score 0..1; fire when score > threshold.

    TODO(A100): drop hey_jero.onnx at config.wakeword.model_path and set
    `use_mock: false`. openWakeWord runtime: `pip install openwakeword`, then
    `openwakeword.Model(wakeword_models=[path])`. Keep the same `.process()` sig.
    """

    def __init__(self, model_path: Optional[str], threshold: float = 0.5):
        self.threshold = threshold
        self.model_path = model_path
        self._model = None
        self.is_real = False
        if model_path and os.path.exists(model_path):
            try:
                import openwakeword
                from openwakeword.model import Model
                # openWakeWord loads the shared melspec/embedding extractors from
                # its package resources dir; our bundle ships them next to
                # hey_jero.onnx, so copy them there if the lib didn't download them.
                try:
                    import shutil
                    res = os.path.join(os.path.dirname(openwakeword.__file__),
                                       "resources", "models")
                    os.makedirs(res, exist_ok=True)
                    bundle = os.path.dirname(model_path)
                    for fm in ("melspectrogram.onnx", "embedding_model.onnx"):
                        src, dst = os.path.join(bundle, fm), os.path.join(res, fm)
                        if os.path.exists(src) and not os.path.exists(dst):
                            shutil.copy(src, dst)
                except Exception:
                    pass
                self._model = Model(wakeword_models=[model_path],
                                    inference_framework="onnx")
                self.is_real = True
                log.info("WakeWordDetector: loaded REAL openWakeWord model %s", model_path)
            except Exception as e:  # pragma: no cover - depends on device deps
                log.warning("WakeWordDetector: real load failed (%s) -> MOCK", e)
        if not self.is_real:
            log.info("WakeWordDetector: MOCK (no %s yet)", model_path)

    def process(self, frame) -> float:
        """Return wake score for one audio frame."""
        if self.is_real:
            scores = self._model.predict(frame)
            return max(scores.values()) if scores else 0.0
        # MOCK: never fires on audio; wake is forced by text path in dev.
        return 0.0

    def fired(self, frame) -> bool:
        return self.process(frame) >= self.threshold


# --------------------------------------------------------------------------- #
# 2. VAD — Silero — MOCK until silero_vad.onnx present
# --------------------------------------------------------------------------- #
class VoiceActivityDetector:
    """
    Real interface (Silero VAD ONNX via sherpa-onnx VoiceActivityDetector or the
    standalone silero package): `.is_speech(frame)` -> bool over 16 kHz frames,
    with min_silence/min_speech from config (400 ms / 0.3-6 s).

    TODO(A100/device): place silero_vad.onnx at config.vad.model_path. sherpa-onnx
    ships a VAD helper (sherpa_onnx.VoiceActivityDetector) that loads this file.
    """

    def __init__(self, model_path: Optional[str], params: dict):
        self.params = params
        self.model_path = model_path
        self._vad = None
        self.is_real = False
        if model_path and os.path.exists(model_path):
            try:
                import sherpa_onnx
                cfg = sherpa_onnx.VadModelConfig()
                cfg.silero_vad.model = model_path
                cfg.silero_vad.threshold = params.get("threshold", 0.5)
                cfg.silero_vad.min_silence_duration = params.get("min_silence_ms", 400) / 1000.0
                cfg.silero_vad.min_speech_duration = params.get("min_speech_ms", 300) / 1000.0
                cfg.sample_rate = 16000
                self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=params.get("max_speech_s", 6))
                self.is_real = True
                log.info("VAD: loaded REAL Silero %s", model_path)
            except Exception as e:  # pragma: no cover
                log.warning("VAD: real load failed (%s) -> MOCK", e)
        if not self.is_real:
            log.info("VAD: MOCK (no %s yet)", model_path)

    def is_speech(self, frame) -> bool:
        if self.is_real:
            self._vad.accept_waveform(frame)
            return not self._vad.empty()
        # MOCK: treat any non-trivial-energy frame as speech (dev only).
        try:
            import numpy as np
            return float(np.abs(frame).mean()) > 1e-3
        except Exception:
            return True


# --------------------------------------------------------------------------- #
# 3. STT — Zipformer sherpa-onnx OnlineRecognizer — REAL
# --------------------------------------------------------------------------- #
class STTEngine:
    """
    Real wrapper around our streaming Zipformer (sherpa-onnx OnlineRecognizer).
    model_dir must hold encoder.onnx / decoder.onnx / joiner.onnx / tokens.txt.
    """

    def __init__(self, model_dir: str, num_threads: int = 4, provider: str = "cpu",
                 decoding_method: str = "greedy_search"):
        self.model_dir = model_dir
        self.is_real = False
        self._rec = None
        required = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
        missing = [f for f in required if not os.path.exists(os.path.join(model_dir, f))]
        if missing:
            log.warning("STT: model files missing in %s: %s -> MOCK", model_dir, missing)
            return
        try:
            import sherpa_onnx
            self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=os.path.join(model_dir, "tokens.txt"),
                encoder=os.path.join(model_dir, "encoder.onnx"),
                decoder=os.path.join(model_dir, "decoder.onnx"),
                joiner=os.path.join(model_dir, "joiner.onnx"),
                num_threads=num_threads,
                sample_rate=16000,
                feature_dim=80,
                decoding_method=decoding_method,
                provider=provider,
            )
            self.is_real = True
            log.info("STT: loaded REAL Zipformer from %s (threads=%d, %s)",
                     model_dir, num_threads, provider)
        except Exception as e:  # pragma: no cover
            log.error("STT: real load failed (%s) -> MOCK", e)

    def create_stream(self):
        return self._rec.create_stream() if self.is_real else None

    def transcribe_samples(self, samples, sample_rate: int = 16000) -> ASRResult:
        """Decode a complete float32 waveform (mono). Used for WAV / VAD segments."""
        t0 = time.time()
        if not self.is_real:
            # MOCK STT should never run in the demo (we require the real model),
            # but keep a safe fallback.
            return ASRResult(text="", conf=0.0, t_end_of_speech=t0, latency_ms=0.0)
        import numpy as np
        s = self._rec.create_stream()
        s.accept_waveform(sample_rate, samples)
        # flush with trailing silence so the last tokens emit
        s.accept_waveform(sample_rate, np.zeros(int(0.3 * sample_rate), dtype=np.float32))
        s.input_finished()
        while self._rec.is_ready(s):
            self._rec.decode_stream(s)
        res = self._rec.get_result(s)
        t_end = time.time()
        # Build-dependent: get_result() returns a plain str on our sherpa-onnx
        # 1.13.8; on other builds it's an object with .text / .ys_probs.
        text = (res if isinstance(res, str) else getattr(res, "text", str(res))).strip()
        conf = self._mean_conf(res, text)
        return ASRResult(text=text, conf=conf, t_end_of_speech=t_end,
                         latency_ms=(t_end - t0) * 1000.0)

    @staticmethod
    def _mean_conf(res, text: str) -> float:
        """
        Confidence of the ASR result.

        Preferred: average per-token acoustic probs (res.ys_probs) when the build
        exposes them. Our sherpa-onnx 1.13.8 greedy build returns a bare string
        with no probs, so we fall back to a PLACEHOLDER: 0.9 for any non-empty
        transcript, 0.0 for silence.
        TODO: switch to real token probs (or a CTC/RNN-T posterior) when we move
        to a build that exposes them; the field name `asr_conf` stays the same.
        """
        probs = getattr(res, "ys_probs", None)
        try:
            if probs:
                import math
                vals = [math.exp(p) if p <= 0 else p for p in probs]  # logprob -> prob
                vals = [min(max(v, 0.0), 1.0) for v in vals]
                if vals:
                    return sum(vals) / len(vals)
        except Exception:
            pass
        return 0.9 if text.strip() else 0.0


# --------------------------------------------------------------------------- #
# 4. Intent — intent ONNX + post_process — MOCK until A100 exports intent.onnx
# --------------------------------------------------------------------------- #
class IntentModel:
    """
    Real interface (A100 export): a small text/intent classifier exported to ONNX
    plus a `post_process(logits, slots)` step living under ~/jero/intent/.
    `.infer(text, asr_conf)` -> IntentResult.

    TODO(A100): place intent.onnx at config.intent.model_path and a sibling
    post_process.py exposing `post_process(logits) -> (intent, slots, conf)`.
    Set config.intent.use_mock=false. The MOCK below is a transparent keyword
    classifier so the end-to-end event flow is demonstrable today — it is NOT a
    stand-in for the trained model's accuracy.
    """

    # keyword -> (intent, slot-extractor). Order matters (first hit wins).
    _RULES: Sequence = (
        (re.compile(r"\b(dance|let'?s dance|boogie)\b", re.I), "dance", {}),
        (re.compile(r"\b(forward|ahead|straight)\b", re.I), "move", {"direction": "forward"}),
        (re.compile(r"\b(back|backward|reverse)\b", re.I), "move", {"direction": "backward"}),
        (re.compile(r"\b(left)\b", re.I), "turn", {"direction": "left"}),
        (re.compile(r"\b(right)\b", re.I), "turn", {"direction": "right"}),
        (re.compile(r"\b(walk|go|move)\b", re.I), "move", {"direction": "forward"}),
        (re.compile(r"\b(hello|hi|hey)\b", re.I), "greet", {}),
    )

    def __init__(self, model_path: Optional[str], post_process_path: Optional[str] = None,
                 use_mock: bool = True):
        self.model_path = model_path
        self._session = None
        self._post = None
        self.is_real = False
        self._tokenizer = None
        self._temperature = 1.0
        if (not use_mock) and model_path and os.path.exists(model_path):
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(model_path,
                                                      providers=["CPUExecutionProvider"])
                model_dir = os.path.dirname(model_path)
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
                # self-contained post_process module (exposes run/decode)
                import importlib.util
                pp = post_process_path or os.path.join(model_dir, "post_process.py")
                spec = importlib.util.spec_from_file_location("intent_post", pp)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._post = mod
                # calibration temperature from gate3 metrics (default 1.0)
                gm = os.path.join(model_dir, "gate3_metrics.json")
                if os.path.exists(gm):
                    import json
                    self._temperature = float(json.load(open(gm)).get("temperature", 1.0))
                self.is_real = True
                log.info("Intent: loaded REAL ONNX %s (T=%.3f)", model_path, self._temperature)
            except Exception as e:  # pragma: no cover
                log.warning("Intent: real load failed (%s) -> MOCK", e)
        if not self.is_real:
            log.info("Intent: MOCK keyword classifier (no %s yet)", model_path)

    def infer(self, text: str, asr_conf: float = 1.0) -> IntentResult:
        if self.is_real:
            # ASR emits ALL-CAPS; the model was trained on normal case. Lowercase
            # for inference (length-preserving, so slot char-spans stay aligned).
            res = self._post.run(text.lower(), self._session, self._tokenizer,
                                 max_len=32, temperature=self._temperature,
                                 threshold=0.7)
            slots = {s["value_key"]: s["value"] for s in res.get("slots", [])
                     if s.get("value") is not None}
            return IntentResult(intent=res["raw_intent"], slots=slots,
                                conf=float(res["confidence"]))
        return self._mock_infer(text, asr_conf)

    def _mock_infer(self, text: str, asr_conf: float) -> IntentResult:
        t = (text or "").strip()
        if not t:
            return IntentResult(intent="none", slots={}, conf=0.0)
        for pat, intent, slots in self._RULES:
            if pat.search(t):
                # Mock confidence: anchored high on a clear keyword, scaled by ASR conf,
                # so sub-0.7 cases are still reachable for low-conf / unknown input.
                conf = min(0.97, 0.85 + 0.12 * min(asr_conf, 1.0))
                return IntentResult(intent=intent, slots=dict(slots), conf=round(conf, 3))
        # No rule matched -> deliberately low confidence to exercise the
        # "Sorry, say that again?" unclear path.
        return IntentResult(intent="none", slots={}, conf=0.4)
