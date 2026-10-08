# TTS — HTP (Supertonic-3 w8a16 on Hexagon V68)

Full 4-model Supertonic-3 pipeline (text-encoder / duration / vector-estimator / vocoder) runs
**w8a16 on the V68 HTP**. `board_driver.py` is the final board-side driver (M1 voice, host char-embed,
8-step flow-matching loop, **sentence/clause chunking** for long text, **numpy spectral de-noiser +
RMS loudness** post-proc). `board_driver_final.py` is the pre-chunking version kept for reference.

- `assets/` — host-side tables needed to run: `te_emb.npy`, `dp_emb.npy` (char-embedding), `canon_inputs.npy` (M1 style), `style_*.npy` (other presets), `xt_htp.npy`.
- `dur_lut.json` — exact CPU durations for Jero lines (w8a16 duration predictor collapses to a constant; we look up / heuristically floor instead).
- **Context binaries** (`ctx_te/ve/vo`, ~100 MB) are built **on-board** and are not in git.

**Fidelity:** DTW-aligned log-mel correlation vs CPU-fp32 = **0.77**; small residual buzz. See `../../../BENCHMARKS.md`.

Usage on device: `python3 board_driver.py "Hi, I am Jero!" out.wav`
