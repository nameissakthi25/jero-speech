# TTS — CPU

Two CPU options:
- **Piper `en_US-amy-medium`** (sherpa-onnx OfflineTts) — the lightweight deployed CPU fallback (`config.yaml::tts.model_dir`). The 60 MB `.onnx` is out-of-band (see `.gitignore`).
- **Supertonic-3 fp32** — the reference pipeline that the HTP w8a16 build is quantized from and measured against (`cpu_cf_M1.wav`, seed 1000).
