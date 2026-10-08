# Jero intent — w8a16 QNN for QCS6490 / HTP v68 — RESULTS

Model: ModernBERT-base joint intent+slot. Static inputs input_ids[1,32] +
attention_mask[1,32] (int64 in ONNX -> int32 in DLC). Outputs intent_logits[1,12],
slot_logits[1,32,17]. opset17, SDPA.

Toolchain: QAIRT 2.37.1.250807 (A100 host) ; AIMET 2.20.0 (venv ~/qhub45) ;
board = QCS6490 / HTP v68, soc_id 498, QNN v2.46.0, aarch64.

## Env fix (step 1)
`qairt-converter` needs numpy 1.26.x + onnx ~1.12-1.16. The venv **~/qenv**
(numpy 1.26.4, onnx 1.16.1) runs it clean; `source ~/qenv/bin/activate` after
`source $QAIRT/bin/envsetup.sh`. QAIRT 2.37 uses `--enable_float_fallback`
(NOT `--float_fallback`); omitted for all-quantized (v68 has no FP16).

## The three root causes found (and the fixes)

1. **Attention mask = -inf collapses quantization.** ModernBERT SDPA fills masked
   attention positions with torch.finfo(fp32).min = -3.40282e38 (ONNX Constant_6
   and Constant_11). Quantizing `attn/Add_2` (scores+mask) to int16 min-max gives
   scale 5.14e33 -> every real attention score rounds to 0 -> uniform attention ->
   collapse. **Fix: graph surgery** (`mask_surgery.py`) replaces the two -3.4e38
   constants with -1.0e4. FP32 output is unchanged (softmax(-1e4)~0 == softmax(-inf)).
   Lossless check: FP32 fixed vs original = intent 120/120, slots 874/874.

2. **int32 calibration misread as float32.** `qairt-quantizer --input_list`
   defaults to reading input raws as float32. Our input_ids/attention_mask raws
   are int32, so the attention_mask was garbage during calibration -> the pooling
   mask tensor `/model/Cast_output_0` got a degenerate range [0, 0.0001] instead
   of [0, 1] -> the mean-pool (ReduceSum(hidden*mask)/ReduceSum(mask)) crushed to
   ~0 -> near-constant intent logits (~0.03) on device. **Fix:
   `--use_native_input_files`** on qairt-quantizer. Mask range becomes [0, 1.0],
   pooled Div range becomes [-13, 4.7]. Same flag is required on `qnn-net-run`.

3. **v68 forces the attention BMMs to 8x8 (hardware limit).** For activation x
   activation matmul (Q@K^T, attn@V), the HTP v68 backend inserts Convert-to-uFxp_8
   on both operands (qairt-quantizer, backend=HTP). AIMET's host sim keeps 16-bit
   on one operand (Q@K = 16bit Q x 8bit K, attn@V = 16x16) and so reaches ~100% in
   simulation, but the real v68 runs 8x8 and loses accuracy that the sim does not
   predict. This is the residual on-device gap (see below). Not fixable by PTQ
   calibration alone.

## AIMET host int-sim (ORT) — top-1 intent agreement vs FP32 (120 utts)

| config (w8a16)                          | intent agree | slot agree |
|-----------------------------------------|--------------|------------|
| naive (htp_v68, -inf mask)              | 45.83%       | 75.29%     |
| uniform a16 (default cfg, -inf mask)    | 45.00%       | 75.40%     |
| per-channel weights (-inf mask)         | 45.00%       | 75.40%     |
| W16A16 control (lossless, -inf mask)    | 44.17%       | —          |
| **mask-fixed (-1e4), htp_v68**          | **100.00%**  | **99.77%** |

W16A16 ~= W8A16 proved the collapse was an ACTIVATION quantizer (the -inf mask),
not the weights. Mask fix -> 100% in sim. FP32 is 100% on these 120 (own test set).

## On-device (QCS6490 / HTP v68) — measured

- Context binary built ON THE BOARD from the quantized DLC (host x86 bakes 4MB
  VTCM which fails on the board's 2MB). **spill_bytes=0** (fits 2MB VTCM),
  no FP16, RC=0. `qnn-net-run --retrieve_context` reaches **"Finished Executing
  Graphs"**, ~12 ms/graph.
- DLC built with QAIRT 2.37 loaded fine by the board's QNN 2.46 generator.

| on-device DLC (w8a16, mask-fixed)                | intent agree vs FP32 | slot agree |
|--------------------------------------------------|----------------------|------------|
| native, int32-calib-bug + -inf pooling collapse  | 25.00% (constant id11)| 0.11%     |
| **native + --use_native_input_files (corrected)**| **42.50%**           | 74.37%     |
| sqnr-cal + symmetric weights                      | FAILS to compose on board (ComposeGraphs err; sha256 verified, not corruption) - does NOT deploy on v68 |

The corrected DLC un-collapses (predictions now vary across classes) but on-device
agreement is **42.5%**, far below the >=99% gate. Cause: root cause #3 (forced 8x8
attention BMM on v68). The same graph reaches 100% in host sim where the attention
keeps higher precision on one matmul operand.

## Honest conclusion

- w8a16 **holds in host simulation (100%)** for the mask-fixed graph.
- w8a16 **does NOT hold on real v68 hardware (42.5%)** because HTP v68 forces the
  attention activation-x-activation matmuls to 8-bit, which this ModernBERT cannot
  absorb. Host AIMET/ORT sim does not model that constraint, so it overestimates.
- Deployable artifact loads and runs on v68 (context binary, spill=0, executes),
  but fails the accuracy gate on-device.
- Paths NOT available in this QAIRT build: injecting AIMET's exact encodings into
  the DLC — `qairt-converter --quantization_overrides` dies on ModernBERT's mask
  `/model/encoder/Expand` (INT32 vs FLOAT32 op-config validation); with
  `--disable_qnn_op_config_validation` the IrQuantizer HANGS (cpu 0, never writes).
  `qairt-quantizer --use_aimet_quantizer` exits silently with no output.
- To actually hit >=99% on v68 would require QAT (train the model to tolerate 8-bit
  attention) or a target with 16-bit activation-matmul support — beyond PTQ.

## Deliverable artifacts (this dir)
- `intent_maskfix.onnx` — the mask-fixed FP32 graph (the key, portable fix)
- `intent_maskfix.encodings` — AIMET w8a16 encodings (870 act / 137 param; sim=100%)
- `intent_maskfix_nativein_quantized.dlc` — deployable quantized w8a16 DLC
  (graph name `intent_maskfix`); loads+runs on v68, 42.5% on-device
- `intent_maskfix_sqnr.dlc` — alt calibration (sqnr acts / symmetric wts), untested on board
- `htp_config.json` + `htp_backend_ext.json` — board backend-ext (soc 498, v68, vtcm 2MB, burst)
- `mask_surgery.py` — the -inf -> -1e4 fix
- `fp32_ref_board.json` — FP32 reference for the 120-utt on-device check
- `SHA256SUMS.txt`

## Reproduce ON THE BOARD (ubuntu@192.168.100.33, pw <REDACTED_BOARD_PW>)
```sh
# stage DLC + configs + vraw/ + input_list_board.txt to /tmp/jero_intent (tmpfs)
cd /tmp/jero_intent
export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp LD_LIBRARY_PATH=/usr/lib
# 1) context binary (2MB VTCM, v68, burst)
qnn-context-binary-generator --backend /usr/lib/libQnnHtp.so --model /usr/lib/libQnnModelDlc.so \
  --dlc_path intent_maskfix_nativein_quantized.dlc --output_dir . --binary_file intent_w8a16 \
  --config_file htp_config.json        # -> intent_w8a16.bin, expect spill_bytes=0
# 2) run 120 utts (NOTE --use_native_input_files for int32 inputs)
qnn-net-run --retrieve_context intent_w8a16.bin --backend /usr/lib/libQnnHtp.so \
  --input_list input_list_board.txt --output_dir out --use_native_input_files --log_level warn
# 3) compare argmax(out/Result_N/intent_logits.raw [1,12 f32]) to fp32_ref_board.json
```
