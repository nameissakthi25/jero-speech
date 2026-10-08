#!/usr/bin/env python3
"""
runtime.py -- Always-on CPU wake-word + VAD endpointing pipeline for "Hey Jero".

Target device : Ubuntu aarch64 (Qualcomm RB3 Gen2), CPU only, onnxruntime.
Models        : openWakeWord shared feature extractors (melspectrogram.onnx +
                embedding_model.onnx) + our trained classifier head hey_jero.onnx,
                plus Silero VAD (silero_vad.onnx) for endpointing.

This module is designed to be *imported* by the `jero-speech` service (built by
another agent). It has NO hard dependency on any microphone library: the caller
pushes 16 kHz mono int16 frames of 512 samples (32 ms) into `process_frame()`,
and the pipeline invokes callbacks. A tiny `__main__` is provided that can read
from a WAV file (or a mic via `sounddevice`, if installed) for local testing.

Activation modes
----------------
  * push-to-talk (PRIMARY)  : call `start_push_to_talk()` -> capture begins
                              immediately, wake word is bypassed.
  * wake word  (SECOND mode): N-consecutive-frame debounce on the hey_jero score
                              above `wake_threshold` triggers capture.

On activation the Silero VAD endpointer runs until end-of-speech
(min_silence_duration_ms trailing silence) or `max_speech_duration_s`, then the
captured utterance PCM (with a short pre-roll and trailing pad) is emitted via
`on_utterance(pcm_int16)`.

Self-hearing mute
-----------------
While the assistant's TTS is playing, and for `mute_tail_ms` after it stops,
incoming mic frames are dropped so the assistant never wakes on its own voice.
Call `notify_playback_start()` / `notify_playback_stop()` around TTS playback.

CPU budget: < 15% of one core on RB3 Gen2 -- see the COST note at the bottom.
"""
from __future__ import annotations

import os
import collections
from dataclasses import dataclass, field
from typing import Callable, Optional, Deque

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512          # 32 ms at 16 kHz (Silero VAD native window)
FRAME_MS = 1000 * FRAME_SAMPLES // SAMPLE_RATE


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class VADConfig:
    """Silero VAD config -- matches configs/silero_vad.json (the shipped gate)."""
    threshold: float = 0.5
    min_silence_duration_ms: int = 400     # trailing silence that ends an utterance
    min_speech_duration_ms: int = 300      # ignore speech blips shorter than this
    max_speech_duration_s: float = 6.0     # hard cap on a single utterance
    speech_pad_ms: int = 200               # pad kept on each side of speech
    window_size_samples: int = 512         # 16 kHz / 512-sample windows


@dataclass
class RuntimeConfig:
    model_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    wakeword_model: str = "hey_jero.onnx"          # our trained classifier head
    silero_model: str = "silero_vad.onnx"
    wake_threshold: float = 0.5                      # tune on real mic audio for gate 5
    wake_debounce_frames: int = 3                    # N consecutive frames above thr
    preroll_ms: int = 300                            # audio kept before activation
    capture_start_timeout_ms: int = 2500             # give up if no speech after wake
    mute_tail_ms: int = 200                          # keep muting this long after TTS
    vad: VADConfig = field(default_factory=VADConfig)


# --------------------------------------------------------------------------- #
# Silero VAD streaming wrapper (onnxruntime, CPU)
# --------------------------------------------------------------------------- #
class SileroVAD:
    """Streaming Silero VAD. Feed one 512-sample float32 frame -> speech prob."""

    def __init__(self, model_path: str, sample_rate: int = SAMPLE_RATE):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1          # keep CPU budget tiny
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, frame_f32: np.ndarray) -> float:
        x = frame_f32.reshape(1, -1).astype(np.float32)
        out, self._h, self._c = self.sess.run(
            None, {"input": x, "sr": self._sr, "h": self._h, "c": self._c}
        )
        return float(out[0, 0])


# --------------------------------------------------------------------------- #
# VAD endpointer -- turns a stream of frames into one bounded utterance
# --------------------------------------------------------------------------- #
class VADEndpointer:
    """
    Stateful endpointer. After the pipeline activates, each 512-sample frame is
    pushed via `update()`; it returns one of:
        "continue"  -- still capturing
        "endpoint"  -- end of speech reached (finalize utterance)
        "timeout"   -- hit max_speech_duration_s (finalize, truncated)
        "no_speech" -- capture_start window elapsed with no speech (abort)
    """

    def __init__(self, vad: SileroVAD, cfg: VADConfig, start_timeout_ms: int):
        self.vad = vad
        self.cfg = cfg
        self._min_sil = cfg.min_silence_duration_ms
        self._min_speech = cfg.min_speech_duration_ms
        self._max_ms = int(cfg.max_speech_duration_s * 1000)
        self._start_timeout = start_timeout_ms
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self._elapsed_ms = 0
        self._speech_ms = 0
        self._silence_ms = 0
        self._triggered = False     # have we seen confirmed speech yet?

    def update(self, frame_f32: np.ndarray) -> str:
        prob = self.vad(frame_f32)
        self._elapsed_ms += FRAME_MS

        if prob >= self.cfg.threshold:
            self._speech_ms += FRAME_MS
            self._silence_ms = 0
            if self._speech_ms >= self._min_speech:
                self._triggered = True
        else:
            if self._triggered:
                self._silence_ms += FRAME_MS

        if not self._triggered and self._elapsed_ms >= self._start_timeout:
            return "no_speech"
        if self._triggered and self._silence_ms >= self._min_sil:
            return "endpoint"
        if self._elapsed_ms >= self._max_ms:
            return "timeout"
        return "continue"


# --------------------------------------------------------------------------- #
# Wake word detector (openWakeWord, onnx CPU) with N-frame debounce
# --------------------------------------------------------------------------- #
class WakeWordDetector:
    def __init__(self, cfg: RuntimeConfig):
        from openwakeword.model import Model
        wpath = os.path.join(cfg.model_dir, cfg.wakeword_model)
        # melspectrogram.onnx + embedding_model.onnx are resolved by openWakeWord
        # from its package resources (or can be pointed at cfg.model_dir copies).
        self.model = Model(
            wakeword_models=[wpath],
            inference_framework="onnx",
            # VAD inside oww is disabled here -- we run our own Silero endpointer.
            vad_threshold=0.0,
        )
        self.key = os.path.splitext(os.path.basename(wpath))[0]  # "hey_jero"
        self.threshold = cfg.wake_threshold
        self.debounce_frames = cfg.wake_debounce_frames
        self._above = 0
        self.last_score = 0.0

    def reset(self) -> None:
        self._above = 0
        try:
            self.model.reset()
        except Exception:
            pass

    def update(self, frame_int16: np.ndarray) -> bool:
        """Feed a 512-sample int16 frame. Returns True on a debounced trigger."""
        scores = self.model.predict(frame_int16)
        self.last_score = float(scores.get(self.key, 0.0))
        if self.last_score >= self.threshold:
            self._above += 1
        else:
            self._above = 0
        if self._above >= self.debounce_frames:
            self._above = 0
            return True
        return False


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
class JeroWakeWord:
    """
    Always-on wake-word + endpointing pipeline.

    Usage (from jero-speech service)::

        ww = JeroWakeWord(on_utterance=handle_pcm)
        ...                                  # audio thread, 512-sample int16 frames
        for frame in mic_frames():           # 16 kHz mono int16, len 512
            ww.process_frame(frame)

        # push-to-talk button:
        ww.start_push_to_talk()

        # around TTS playback:
        ww.notify_playback_start(); ...; ww.notify_playback_stop()
    """

    IDLE = "idle"
    CAPTURING = "capturing"

    def __init__(
        self,
        on_utterance: Optional[Callable[[np.ndarray], None]] = None,
        on_wake: Optional[Callable[[float], None]] = None,
        config: Optional[RuntimeConfig] = None,
        enable_wakeword: bool = True,
    ):
        self.cfg = config or RuntimeConfig()
        self.on_utterance = on_utterance
        self.on_wake = on_wake
        self.enable_wakeword = enable_wakeword

        self.vad = SileroVAD(os.path.join(self.cfg.model_dir, self.cfg.silero_model))
        self.endpointer = VADEndpointer(self.vad, self.cfg.vad, self.cfg.capture_start_timeout_ms)
        self.detector = WakeWordDetector(self.cfg) if enable_wakeword else None

        preroll_frames = max(1, self.cfg.preroll_ms // FRAME_MS)
        self._preroll: Deque[np.ndarray] = collections.deque(maxlen=preroll_frames)
        self._captured: list[np.ndarray] = []
        self.state = self.IDLE

        self._muted_until_ms = 0.0
        self._playing = False
        self._now_ms = 0.0

    # --- self-hearing mute -------------------------------------------------- #
    def notify_playback_start(self) -> None:
        self._playing = True

    def notify_playback_stop(self) -> None:
        self._playing = False
        self._muted_until_ms = self._now_ms + self.cfg.mute_tail_ms

    def _muted(self) -> bool:
        return self._playing or self._now_ms < self._muted_until_ms

    # --- activation --------------------------------------------------------- #
    def start_push_to_talk(self) -> None:
        """PRIMARY path: begin capturing an utterance immediately."""
        self._begin_capture(source="ptt")

    def _begin_capture(self, source: str) -> None:
        self.state = self.CAPTURING
        self.endpointer.reset()
        self._captured = list(self._preroll)     # include pre-roll audio
        if self.detector is not None:
            self.detector.reset()

    def _finalize(self, reason: str) -> None:
        pad_frames = max(0, self.cfg.vad.speech_pad_ms // FRAME_MS)
        tail = [np.zeros(FRAME_SAMPLES, dtype=np.int16) for _ in range(pad_frames)]
        pcm = np.concatenate(self._captured + tail) if self._captured else np.zeros(0, np.int16)
        self.state = self.IDLE
        self._captured = []
        if reason in ("endpoint", "timeout") and self.on_utterance is not None:
            self.on_utterance(pcm)

    # --- main entry: feed one 512-sample int16 frame ------------------------ #
    def process_frame(self, frame_int16: np.ndarray) -> None:
        frame_int16 = np.asarray(frame_int16, dtype=np.int16).reshape(-1)
        if frame_int16.shape[0] != FRAME_SAMPLES:
            raise ValueError(f"expected {FRAME_SAMPLES} samples, got {frame_int16.shape[0]}")
        self._now_ms += FRAME_MS

        if self._muted():
            # Keep pre-roll fresh-ish but do not detect or capture while muted.
            return

        self._preroll.append(frame_int16.copy())
        frame_f32 = frame_int16.astype(np.float32) / 32768.0

        if self.state == self.IDLE:
            if self.enable_wakeword and self.detector is not None:
                if self.detector.update(frame_int16):
                    if self.on_wake is not None:
                        self.on_wake(self.detector.last_score)
                    self._begin_capture(source="wake")
            return

        # CAPTURING
        self._captured.append(frame_int16.copy())
        result = self.endpointer.update(frame_f32)
        if result == "continue":
            return
        self._finalize(result)


# --------------------------------------------------------------------------- #
# Local test harness (optional; not used by the service)
# --------------------------------------------------------------------------- #
def _iter_wav_frames(path: str):
    import soundfile as sf
    data, sr = sf.read(path, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    assert sr == SAMPLE_RATE, f"expected 16 kHz wav, got {sr}"
    for i in range(0, len(data) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        yield data[i:i + FRAME_SAMPLES]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Local test of the Hey Jero runtime pipeline")
    ap.add_argument("--wav", help="16 kHz mono wav to stream through the pipeline")
    ap.add_argument("--model-dir", default=RuntimeConfig.model_dir)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--ptt", action="store_true", help="use push-to-talk (bypass wake word)")
    args = ap.parse_args()

    cfg = RuntimeConfig(model_dir=args.model_dir, wake_threshold=args.threshold)
    n_utt = {"n": 0}

    def _on_utt(pcm):
        n_utt["n"] += 1
        print(f"[utterance #{n_utt['n']}] {len(pcm)} samples ({len(pcm)/SAMPLE_RATE:.2f}s)")

    def _on_wake(score):
        print(f"[wake] score={score:.3f}")

    ww = JeroWakeWord(on_utterance=_on_utt, on_wake=_on_wake, config=cfg,
                      enable_wakeword=not args.ptt)
    if args.wav:
        if args.ptt:
            ww.start_push_to_talk()
        for fr in _iter_wav_frames(args.wav):
            ww.process_frame(fr)
        ww._finalize("endpoint")  # flush any tail for the test
        print("done.")
    else:
        print("no --wav given; import this module from the jero-speech service instead.")

# --------------------------------------------------------------------------- #
# COST NOTE (expected CPU on RB3 Gen2, one Kryo/Cortex core @ ~1.8 GHz):
#   per 32 ms frame we run:
#     - Silero VAD (512-sample)         ~0.3-0.6 ms   (only while NOT idle, or
#                                                       always if you also gate
#                                                       wake with VAD)
#     - oww melspec+embed+classifier    ~2-4 ms per 80 ms of audio => amortized
#                                        ~1-1.6 ms per 32 ms frame
#   Total steady-state ~2-3 ms of compute per 32 ms of wall clock  => ~6-10% of
#   one core, within the < 15% budget. The classifier head (hey_jero.onnx) is a
#   few hundred KB and its own inference is < 0.1 ms; cost is dominated by the
#   shared melspectrogram+embedding extractors. Use intra_op_num_threads=1 (set
#   above for VAD; set the same for oww via ORT env) to avoid thread oversubscription.
# --------------------------------------------------------------------------- #
