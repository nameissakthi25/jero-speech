"""
Jero TTS — Piper (VITS) via sherpa-onnx OfflineTts.

On-device (RB3 Gen2, QCS6490) this runs on CPU with num_threads=4. The voice is
en_US-amy-medium (Piper). The sherpa-onnx bundle ships model.onnx + tokens.txt +
espeak-ng-data/, which is exactly what OfflineTts needs.

Usage:
    from brain.speech.tts import PiperTTS
    tts = PiperTTS.from_config(config)
    tts.synthesize_to_wav("Hi, I'm Jero!", "out.wav")

CLI:
    python -m brain.speech.tts --text "Hi, I'm Jero!" --out out.wav
"""
from __future__ import annotations

import argparse
import logging
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("jero.speech.tts")

# Repo root = jero-speech/  (this file is brain/speech/tts.py)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TTS_DIR = ROOT / "models" / "tts" / "vits-piper-en_US-amy-medium"


@dataclass
class TTSConfig:
    model: str
    tokens: str
    data_dir: str
    num_threads: int = 4
    provider: str = "cpu"
    speed: float = 1.0
    sid: int = 0  # speaker id (amy is single-speaker -> 0)


class PiperTTS:
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.sample_rate = 22050  # overwritten from the engine after load
        self._tts = None
        self._load()

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_dir(cls, tts_dir: os.PathLike | str = DEFAULT_TTS_DIR,
                 num_threads: int = 4, provider: str = "cpu",
                 speed: float = 1.0) -> "PiperTTS":
        d = Path(tts_dir)
        return cls(TTSConfig(
            model=str(d / "en_US-amy-medium.onnx"),
            tokens=str(d / "tokens.txt"),
            data_dir=str(d / "espeak-ng-data"),
            num_threads=num_threads, provider=provider, speed=speed,
        ))

    @classmethod
    def from_config(cls, config: dict) -> "PiperTTS":
        t = config.get("tts", {})
        tts_dir = t.get("model_dir")
        if tts_dir:
            tts_dir = (ROOT / tts_dir) if not os.path.isabs(tts_dir) else Path(tts_dir)
        else:
            tts_dir = DEFAULT_TTS_DIR
        return cls.from_dir(tts_dir,
                            num_threads=t.get("num_threads", 4),
                            provider=t.get("provider", "cpu"),
                            speed=t.get("speed", 1.0))

    def _load(self):
        for p in (self.cfg.model, self.cfg.tokens, self.cfg.data_dir):
            if not os.path.exists(p):
                raise FileNotFoundError(f"TTS asset missing: {p}")
        import sherpa_onnx
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=self.cfg.model,
                    tokens=self.cfg.tokens,
                    data_dir=self.cfg.data_dir,
                ),
                num_threads=self.cfg.num_threads,
                provider=self.cfg.provider,
            ),
        )
        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self.sample_rate = self._tts.sample_rate
        log.info("PiperTTS loaded (%s, sr=%d, threads=%d, %s)",
                 Path(self.cfg.model).name, self.sample_rate,
                 self.cfg.num_threads, self.cfg.provider)

    # -- synthesis ---------------------------------------------------------- #
    def synthesize(self, text: str):
        """Return (samples: list[float], sample_rate: int)."""
        audio = self._tts.generate(text, sid=self.cfg.sid, speed=self.cfg.speed)
        return audio.samples, audio.sample_rate

    def synthesize_to_wav(self, text: str, out_path: os.PathLike | str) -> str:
        samples, sr = self.synthesize(text)
        _write_wav(out_path, samples, sr)
        return str(out_path)


def _write_wav(out_path, samples, sr) -> None:
    import numpy as np
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(samples, dtype=np.float32)
    pcm = np.clip(arr, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _main(argv=None):
    ap = argparse.ArgumentParser(description="Jero Piper TTS")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="tts_out.wav")
    ap.add_argument("--tts-dir", default=str(DEFAULT_TTS_DIR))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    tts = PiperTTS.from_dir(args.tts_dir, num_threads=args.threads)
    path = tts.synthesize_to_wav(args.text, args.out)
    import os as _os
    print(f"wrote {path} ({_os.path.getsize(path)} bytes, sr={tts.sample_rate})")


if __name__ == "__main__":
    _main()
