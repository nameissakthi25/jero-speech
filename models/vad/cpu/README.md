# VAD — CPU (Silero v4)

Silero VAD (`silero_vad.onnx`, 1.7 MB) via sherpa-onnx. 400 ms end-silence, 0.3–6 s utterance.

**Benchmark:** acc 94.5%, precision 1.00, ROC-AUC 0.94, **0.078 ms/frame**, RTF 0.0024 (~410× RT). See `bench/bench_vad_cpu.json`.
