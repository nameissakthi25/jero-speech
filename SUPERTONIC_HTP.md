# Supertonic-3 on-device TTS → QCS6490 HTP (Hexagon v68), w8a16

Converting the 4 Supertonic-3 ONNX models to **int8-weight / int16-activation (w8a16)** and
running the full text-to-speech pipeline on the Qualcomm RB3 Gen2 HTP (soc_id 498, Hexagon v68).
Tooling: **QAIRT / QNN 2.37.1.250807**. SDK: `supertonic==1.3.1` (ONNX Runtime pipeline).

This log is honest about what works and what the quantization costs. w8a16 on a flow-matching
generative vocoder is the quality risk; results and verdict are in §6.

---

## 0. Pipeline (what runs where)

The supertonic `core.Supertonic.__call__` pipeline, verbatim:

```
text ─ host: unicode-indexer → text_ids[1,T] + text_mask[1,1,T]
     ─ host: char-embedding lookup  (V68 can't Gather integer indices on-chip)
duration_predictor(inputs_embeds, style_dp, text_mask) ─► duration (sec)      [NPU]
text_encoder(inputs_embeds, style_ttl, text_mask)      ─► text_emb[1,256,T]   [NPU]
host: sample noisy_latent[1,144,L] from N(0,1)*latent_mask  (L from duration)
for step in 0..7:   # DEFAULT_TOTAL_STEPS = 8, Euler step is BAKED INTO the graph
    vector_estimator(noisy_latent, text_emb, style_ttl, latent_mask, text_mask,
                     current_step, total_step) ─► noisy_latent (next)          [NPU]
vocoder(latent[1,144,L]) ─► wav                                                [NPU]
```

Key facts discovered by reading the SDK:
- The **flow-matching Euler update is inside `vector_estimator`** — it returns the next latent
  directly, so the host loop only feeds `current_step`. No host solver math.
- `text_ids` are **unicode-indexer indices** (`indexer[ord(char)]`), not raw codepoints; the
  language tag `<en>…</en>` is added as literal characters before indexing.
- Config: `sample_rate=44100`, latent frame = `base_chunk_size(512) * chunk_compress_factor(6) =
  3072` samples, `noisy_latent` channel dim = `latent_dim(24) * 6 = 144`.

## 1. Static buckets

Dynamic dims must be static for HTP. Measured over 30 short lines: text length T ∈ [23,34],
latent length L ∈ [21,37]. Chosen buckets: **T = 64, L = 64** (headroom for short robot lines).
Pad text_ids / noisy_latent to the bucket; `text_mask` / `latent_mask` zero the padding.

Padded-bucket vs unpadded reference (original fp32 models): text_emb corr **min 0.986 / mean 0.993**,
duration within ~2%. Bucketing is near-lossless on the valid region.

## 2. Graph surgery (P2) — each validated bit-exact vs original on CPU

Per model (`convert` scripts on A100 `~/stwork/`):

1. **Host-side char embedding.** `text_encoder` and `duration_predictor` each have exactly one
   token-embedding `Gather` on `text_ids` (tables `[8322,256]` and `[8322,64]`). Removed it and
   exposed `inputs_embeds` ([1,T,256] / [1,T,64]); the host does the lookup. All other `Gather`s are
   shape-gathers that fold under static shapes. **(V68 returns wrong rows for on-chip integer Gather.)**
2. **Mask fill surgery.** `vector_estimator` had one `-inf` attention mask-fill constant → replaced
   with `-1e4` (softmax-equivalent, keeps int16 range sane). `text_encoder`/`duration_predictor`
   take `text_mask` as a float input already — no internal `-inf`. `vocoder` is clean.
3. **Converter-compatibility fixes** (needed for qairt-converter 2.37, each exact):
   - `Erf` (exact GELU) has a translation but **no layout inferer** in 2.37 → replaced with the
     tanh approximation `erf(u) ≈ tanh(1.1283792·u + 0.1009·u³)` (Tanh is a supported Neuron op).
     Max approx error 3.6e-4; end-to-end effect: see below.
   - Relative-position-attention `Pad` nodes take a **constant `data` input** (`emb_rel_k/v`), which
     the converter's Pad translation can't fetch as a buffer → routed each through an `Identity`
     so it becomes a graph buffer. Then topologically re-sorted.
   - **Do NOT use onnx-simplifier** with static shapes here: it silently corrupted `text_encoder`
     (corr 0.88, duration 2.60→3.29s). A conservative numpy constant-fold was used instead.

CPU validation of surgered graphs vs originals:

| model | corr vs original (fp32) |
|---|---|
| text_encoder (host-embed) | **1.000000** (maxabs 0) |
| duration_predictor (host-embed) | exact (0.00e0) |
| vector_estimator (−inf→−1e4) | **1.000000** (maxabs 0) |
| full graph after erf-approx + fold (`_cf`) | text_emb **0.999999**; wav MCD **2.15 dB**, logmel_corr **1.000** |

The erf-approx + static graph is essentially inaudible (MCD 2.15 dB vs the original padded pipeline).

## 3. Convert + quantize to w8a16 (P3)

```bash
# convert (run converter under qenv so numpy/onnx are present)
~/qenv/bin/python $(which qairt-converter) --input_network <model>_cf_id.onnx \
    --output_path <m>.dlc  -s <input> <dims> ...        # one -s per input, static dims
# quantize  (weights default to 8-bit = w8a16)
~/qenv/bin/python $(which qairt-quantizer) --input_dlc <m>.dlc --output_dlc <m>_w8a16.dlc \
    --input_list calib_<m>.txt  --act_bitwidth 16 --bias_bitwidth 32
```

Calibration = **real intermediate tensors** dumped from the CPU pipeline on 30 varied short
lines, padded to the buckets (text_encoder/dp/vocoder: 30 samples each; vector_estimator: 72 =
24 lines × steps {0,3,7}). All 4 quantized min-max cleanly on the first pass.

**sqnr fallbacks pre-built** for the two generative-quality risks (vector_estimator, vocoder):
`ve_w8a16_sqnr.dlc`, `vo_w8a16_sqnr.dlc` (`--act_quantizer_calibration sqnr`). Staged + chunk-split
so they can be swapped in on-board immediately if min-max sounds buzzy/degraded.

| model | float DLC | w8a16 DLC |
|---|---|---|
| duration_predictor | 1.6 MB | 0.64 MB |
| text_encoder | 27.9 MB | 7.5 MB |
| vector_estimator | 257 MB | 65.6 MB |
| vocoder | 101.6 MB | 25.7 MB |

> Note: QAIRT's x86 CPU backend cannot compose these graphs (`OpConfig validation failed for
> Transpose`), so cheap on-host simulation was not possible — consistent with "sim != device".
> The real test is on-board audio (§6).

## 4. Context binaries on the board (P5 note)

Context binaries are generated **on the board** (`gen_ctx.sh`), because x86-host context-gen bakes
4 MB VTCM which fails on the board's 2 MB. One `qnn-context-binary-generator` per DLC, then
`qnn-net-run --retrieve_context` for inference.

## 5. Board driver

`board_driver.py` (board-side, pure Python + numpy + qnn-net-run): ports the unicode text
processor, does host char-embedding + mask building + noisy-latent sampling (seed 1000, matched to
the CPU reference) + the 8-step estimator loop, calling each HTP context binary via qnn-net-run.
Produces a WAV on the board.

## 6. On-device audio + honest verdict

**DONE (2026-10-08, QDC RB3 Gen2 HTP).** All 4 Supertonic-3 models run **w8a16 on the V68 HTP** and
produce audio. Final board driver: `models/tts/htp/board_driver.py` — M1 voice, sentence/clause
**chunking** for long text, **numpy spectral de-noiser + RMS loudness** post-processing.

- **Fidelity vs CPU-fp32** (same seed, DTW-aligned per-mel-normalized log-mel correlation, since
  flow-matching is stochastic and durations differ): raw HTP **0.754**, shipped (de-noised) **0.772**
  (method sanity: cpu-vs-cpu = 1.000). The graph-surgery/approx cost alone is MCD 2.15 dB / corr 1.000
  (§2), so the gap is the w8a16 vocoder quantization.
- **Perceptual:** intelligible, playful robot voice; **small residual buzz** remains (the known w8a16
  flow-matching risk), loudness now natural (crest ~18 dB). 5 M1 demo lines in `voice_audition/M1_team_demo/`.
- **Latency:** ~1.3× RTF cold per call (context reload); one-time load in a persistent service.

Metrics defined (device output compared to the **same-noise-seed** CPU fp32 reference
`cpu_cf_M1.wav` — essential, because flow-matching is stochastic and a different seed gives a
different-but-equally-valid waveform):
- `logmel_corr` — 40-band log-mel frame correlation (1.0 = identical)
- `MCD` — mel-cepstral distortion in dB (lower better; <3 excellent, 3–6 good, 6–8 noticeable, >8 degraded)

## Status / reproduce

- A100 build tree: `~/stwork/` (onnx surgery, dlc/, calib/, raw/ references).
- Mac staging: scratchpad `stboard/` (quantized DLCs, driver, references, chunked ve, compare_audio.py).
- cpu references (seed 1000): `cpu_padded_M1.wav` (original models), `cpu_cf_M1.wav` (surgered fp32).
