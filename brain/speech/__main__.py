"""
python -m brain.speech --print

Prints every speech event as a JSON line — test the whole pipeline without the
robot. Today this runs end-to-end on the REAL Zipformer STT over WAV input and a
MOCK intent classifier, so the event flow is demonstrable now.

Modes:
    --print                 run the demo WAVs and print JSON event lines (default)
    --wav PATH [PATH ...]   STT over your own WAV(s) instead of the demo set
    --mic                   live mic loop (wake+VAD mocked until models land)
    --say "text"            queue a line and exit (TTS smoke test)

The default demo feeds, through real STT:
    1. the Zipformer sample.wav (real human speech)
    2. TTS-rendered "Jero stop"        -> exercises the STOP fast-path
    3. TTS-rendered "walk forward"     -> exercises an intent event
    4. TTS-rendered "play the xylophone" -> exercises the unclear (<0.7) path
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

from .service import SpeechService
from .tts import ROOT

ZIPFORMER_SAMPLE = Path("/Users/nameissakthi/Desktop/personal/zipformer-qcs6490/"
                        "zipformer-enin-qairt-aimet/onnx_test/samples/sample.wav")


def _printer(svc, stop_evt):
    for ev in svc.events():
        if stop_evt.is_set():
            break
        print(json.dumps(ev), flush=True)


def _render_demo_phrases(svc, workdir: Path):
    """Render a few command phrases with our TTS so we can push them back
    through the REAL STT. Returns list of (label, wav_path)."""
    workdir.mkdir(parents=True, exist_ok=True)
    phrases = [
        ("stop_fastpath", "Jero stop"),
        ("intent_move", "walk forward"),
        ("unclear", "play the xylophone"),
    ]
    out = []
    if svc.tts is None:
        return out
    for label, text in phrases:
        p = workdir / f"demo_{label}.wav"
        try:
            svc.tts.synthesize_to_wav(text, p)
            out.append((label, str(p)))
        except Exception as e:
            logging.warning("could not render demo phrase %r: %s", text, e)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m brain.speech")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="print every event as a JSON line (default demo)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--wav", nargs="+", help="run STT on these WAV(s)")
    ap.add_argument("--mic", action="store_true", help="live mic loop")
    ap.add_argument("--say", help="queue a line then exit (TTS smoke test)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    svc = SpeechService(config_path=args.config)
    stop_evt = threading.Event()
    tp = threading.Thread(target=_printer, args=(svc, stop_evt), daemon=True)
    tp.start()

    try:
        if args.say:
            svc.say(args.say)
            time.sleep(3.0)
            return

        if args.mic:
            svc.run_mic_loop()
            return

        wavs = []
        if args.wav:
            wavs = [("user", w) for w in args.wav]
        else:
            # default demo set
            if ZIPFORMER_SAMPLE.exists():
                wavs.append(("zipformer_sample", str(ZIPFORMER_SAMPLE)))
            wavs += _render_demo_phrases(svc, ROOT / "cache" / "demo")

        for label, wav in wavs:
            print(json.dumps({"type": "_demo_input", "label": label, "wav": wav}), flush=True)
            svc.feed_wav(wav)
            time.sleep(0.4)  # let any triggered say()/state events flush in order

        # allow safety "Stopping." playback + state events to flush
        time.sleep(2.0)
    finally:
        stop_evt.set()
        svc.close()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
