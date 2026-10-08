# Model conversion — float → w8a16 on Hexagon V68 (QCS6490)

The scripts that convert/quantize each model to run on the RB3 Gen2 HTP. These ran on the
**A100 build box** (`~/stwork/` for TTS, `~/jero/intent/qnn/` for intent) with
**QAIRT/QNN 2.37.1.250807** + AIMET. Paths inside the scripts are A100-relative; they are
committed here as the **record + reproducible recipe**, not as turnkey local scripts.

> Full narrative recipes with exact `qairt-converter` / `qairt-quantizer` commands:
> **`../SUPERTONIC_HTP.md`** (TTS) and the Laya repo `laya-qcs6490/docs/RECIPE.md` (intent).

## tts_supertonic/ — Supertonic-3 (4 models → w8a16)

Pipeline order:
1. `inspect_graphs.py`, `inspect2.py` — find the token-embedding `Gather`s + dynamic dims.
2. `surgery.py` — **static-shape surgery**: host-side char embedding (V68 mis-Gathers on-chip), bucket T=64/L=64, mask fill −3.4e38→−1e4.
3. `erf_fix.py`, `erf_and_fold.py` — replace exact `Erf` GELU with tanh approx (QAIRT 2.37 has no Erf layout inferer) + conservative constant-fold.
4. `fixpad.py`, `padfix2.py` — route rel-pos-attention `Pad` constant inputs through `Identity` so the converter can fetch them as buffers.
5. `simplify.py` — numpy constant-fold (do **not** use onnx-simplifier: it corrupted text_encoder).
6. `capture.py` — dump real intermediate tensors from the CPU pipeline for calibration → `calib_{dp,te,ve,vo}.txt`.
7. `gen_cf_ref.py`, `measure.py`, `validate_f.py`, `validate_srg.py`, `simrun.py`, `check_textproc.py` — fp32 reference + per-stage validation (surgered-vs-original MCD 2.15 dB).
8. **Quantize** (commands in `../SUPERTONIC_HTP.md §3`): `qairt-quantizer --act_bitwidth 16 --bias_bitwidth 32` (weights=8 → w8a16); vocoder uses per-channel + CLE (best buzz).
9. `board_driver.py` — the A100 dev copy of the board runtime (final shipped one is `../models/tts/htp/board_driver.py`).

## intent_laya/ — ModernBERT intent ("Laya", w8a16 encoder split)

- `encoder_split.py` — split the model: **embedding lookup on host** (V68 Gather bug), only the encoder body on NPU (feed `inputs_embeds`), heads host-float.
- `mask_surgery.py` / `mask_surgery100.py` / `intent_surgery.py` — attention mask −3.4e38→−1e4 (int16-safe, softmax-lossless).
- `gen_calib.py`, `intent_calib.py`, `gen_encoder_data.py`, `gen_exposed_data.py` — calibration + exposed-embedding data.
- `aimet_quantsim.py`, `aimet_quantsim_maskfix.py`, `enc_aimet_sim.py` — AIMET quant-sim (host; **over-reports** — see below).
- `verify_*.py`, `eval_ondevice.py` — fp32-vs-w8a16 agreement, host sim **and on-device**.
- `BOARD_CONTEXTGEN_README.md` — build the context binary **on-board** (2 MB VTCM).
- `deliver/RESULTS.md` — the honest result: **host sim 100%, real V68 ~41%** → intent ships on CPU (fp32).
- `diag_*.py`, `inspect_expand.py`, `board_prep.py` — diagnostics.

## wakeword/ — openWakeWord "Hey Jero"

- `runtime.py` — openWakeWord inference wrapper. `eval_gate5.py` — gate-5 (miss/false-accept) eval.

## stt_zipformer — NOT here

The Indian-English Zipformer STT (encoder/decoder/joiner → w8a16 + on-board ctx bins, token-exact
on HTP) was built in its **own repo**: `zipformer-qcs6490/zipformer-enin-qairt-aimet/`. See
`../models/stt/htp/README.md` for the pointer and verified result.

## Key cross-model lessons

- **V68 Gather returns wrong rows** for on-chip integer indexing → do all embedding lookups on the host.
- **Mask fill −3.4e38 breaks int16 quant** → replace with −1e4 (softmax-equivalent).
- **Sim ≠ device**: AIMET/ORT run attention matmuls in float and over-report; always verify on real HTP.
- **Context binaries must be built on-board** (2 MB VTCM; x86 host bakes 4 MB and fails to load).
- w8a16 is the **only** option on V68 (no fp16/fp32; 16-bit weights fail `ComposeGraphs`).
