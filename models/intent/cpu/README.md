# Intent — CPU (ModernBERT-base, fp32) — DEPLOYED

The intent classifier ("Laya", 12 intents + BIO slots) runs **fp32 on CPU** — full accuracy,
**~28 ms** (gate <100 ms). `intent.onnx` (~569 MB) is out-of-band (see `.gitignore`); tokenizer,
`label_maps.json`, `post_process.py`, `gate3_metrics.json` are here. Wired via `brain/speech/loaders.py::IntentModel`.
