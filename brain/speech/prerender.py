"""
Pre-render Jero's canned lines to cached WAVs at startup.

Why: these lines are spoken constantly and must play with zero synthesis latency
(especially the safety line "Stopping."). We render once into cache/canned/ and
the SpeechService plays the cached file directly.

The canned line set and cache dir come from config.yaml; a built-in default list
matches the spec so this runs standalone too.

CLI:
    python -m brain.speech.prerender            # uses config.yaml defaults
    python -m brain.speech.prerender --force    # re-render even if cached
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path

from .tts import PiperTTS, ROOT

log = logging.getLogger("jero.speech.prerender")

# Spec canned lines (also mirrored in config.yaml -> canned_lines).
DEFAULT_CANNED = [
    "Hi, I'm Jero!",
    "Let's dance!",
    "Ta-da!",
    "Oops!",
    "Sorry, say that again?",
    "Walking forward.",
    "Turning left.",
    "Stopping.",
]

DEFAULT_CACHE_DIR = ROOT / "cache" / "canned"


def cache_key(text: str) -> str:
    """Stable filename for a line (content-addressed so edits re-render)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def cache_path(cache_dir, text: str) -> Path:
    return Path(cache_dir) / f"{cache_key(text)}.wav"


def prerender(tts: PiperTTS, lines=None, cache_dir=DEFAULT_CACHE_DIR, force=False) -> dict:
    """Render each line to cache. Returns {text: wav_path}."""
    lines = lines or DEFAULT_CANNED
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for line in lines:
        wav = cache_path(cache_dir, line)
        if force or not wav.exists() or os.path.getsize(wav) == 0:
            tts.synthesize_to_wav(line, wav)
            log.info("rendered %-28s -> %s (%d bytes)", repr(line), wav.name,
                     os.path.getsize(wav))
        else:
            log.info("cached   %-28s -> %s", repr(line), wav.name)
        out[line] = str(wav)
    return out


def _main(argv=None):
    ap = argparse.ArgumentParser(description="Pre-render Jero canned lines")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = {}
    try:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        log.warning("config load failed (%s); using built-in defaults", e)

    lines = cfg.get("canned_lines", DEFAULT_CANNED)
    cache_dir = cfg.get("tts", {}).get("cache_dir", str(DEFAULT_CACHE_DIR))
    if not os.path.isabs(cache_dir):
        cache_dir = ROOT / cache_dir
    tts = PiperTTS.from_config(cfg)
    mapping = prerender(tts, lines, cache_dir, force=args.force)
    print(f"pre-rendered {len(mapping)} lines into {cache_dir}")


if __name__ == "__main__":
    _main()
