# STT — CPU (Zipformer, Indian-English)

Runtime: sherpa-onnx `OnlineRecognizer`, greedy, 4 threads.
Big binaries (encoder/decoder/joiner .onnx, ~287 MB) are synced **out-of-band** from
`zipformer-qcs6490/zipformer-enin-qairt-aimet/onnx_test/model/` (see `config.yaml::stt.model_dir`).
`tokens.txt` is kept here as the lightweight marker.

**Benchmark:** WER 34.8% (cross-accent proxy, US LibriSpeech), RTF 0.037 (~27× RT). See `../../../BENCHMARKS.md`.
