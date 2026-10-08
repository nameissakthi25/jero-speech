"""
Interruptible WAV player.

Playback is a killable subprocess so a safety preemption can cut off audio well
within the 100 ms budget (SIGKILL on the player process stops the DAC feed
immediately). Backend is picked per platform:

  * macOS (dev):   afplay
  * Linux / RB3:   aplay  (ALSA; installed by setup_rb3.sh via alsa-utils)

A pure-python sounddevice backend is used as a fallback when neither CLI player
is on PATH.
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("jero.speech.player")


def _pick_backend() -> Optional[str]:
    if platform.system() == "Darwin" and shutil.which("afplay"):
        return "afplay"
    if shutil.which("aplay"):
        return "aplay"
    if shutil.which("afplay"):
        return "afplay"
    try:
        import sounddevice  # noqa: F401
        import soundfile    # noqa: F401
        return "sounddevice"
    except Exception:
        return None


class Player:
    def __init__(self):
        self.backend = _pick_backend()
        self._proc: Optional[subprocess.Popen] = None
        self._sd_stop = threading.Event()
        self._lock = threading.Lock()
        if not self.backend:
            log.warning("No audio backend found (afplay/aplay/sounddevice). "
                        "Playback is a no-op; timing is still simulated.")
        else:
            log.info("Player backend: %s", self.backend)

    def play_blocking(self, wav_path: str) -> None:
        """Play a WAV to completion unless stop() is called."""
        wav_path = str(wav_path)
        if not Path(wav_path).exists():
            log.error("WAV not found: %s", wav_path)
            return
        if self.backend in ("afplay", "aplay"):
            cmd = [self.backend, wav_path]
            with self._lock:
                self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL)
            self._proc.wait()
            with self._lock:
                self._proc = None
        elif self.backend == "sounddevice":
            self._play_sd(wav_path)
        else:
            # No backend: simulate duration so state/timing logic still exercises.
            self._simulate(wav_path)

    def stop(self) -> None:
        """Preempt current playback immediately (<100 ms)."""
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.kill()
        self._sd_stop.set()

    # -- backends ----------------------------------------------------------- #
    def _play_sd(self, wav_path: str):
        import sounddevice as sd
        import soundfile as sf
        self._sd_stop.clear()
        data, sr = sf.read(wav_path, dtype="float32")
        blk = int(0.02 * sr)  # 20 ms blocks -> fast stop response
        with sd.OutputStream(samplerate=sr, channels=(data.shape[1] if data.ndim > 1 else 1)) as out:
            for i in range(0, len(data), blk):
                if self._sd_stop.is_set():
                    break
                out.write(data[i:i + blk])

    def _simulate(self, wav_path: str):
        import wave
        try:
            with wave.open(wav_path, "rb") as wf:
                dur = wf.getnframes() / float(wf.getframerate())
        except Exception:
            dur = 0.5
        self._sd_stop.clear()
        end = time.time() + dur
        while time.time() < end and not self._sd_stop.is_set():
            time.sleep(0.01)
