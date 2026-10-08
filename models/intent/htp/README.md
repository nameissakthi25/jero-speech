# Intent — HTP (ModernBERT w8a16) — ✅ SOLVED via QAT (100% on real V68)

**Updated 2026-10-08.** The earlier "w8a16 PTQ hits a ~42% ceiling → run on CPU" conclusion is
**superseded**. Post-training quant *is* capped on V68, but **quantization-aware training (QAT)
closes the gap: intent w8a16 = 100% top-1 on the REAL QCS6490 V68 HTP** (120/120 vs fp32 and vs
gold, 0 missing, all 12 classes used — verified on-device, not simulated).

## The result (on-device, QDC RB3 Gen2, soc_id 498)

| | top-1 vs fp32 | top-1 vs gold |
|---|---|---|
| PTQ w8a16 (old) | 42.5% | ~42% |
| **QAT w8a16 (new)** | **100%** | **100%** |

Run: build context binary on-board (190 MB, spills to DDR) → `qnn-net-run --retrieve_context
--use_native_input_files` over 120 inputs in **~3 s**. Full-graph DLC (input_ids + attention_mask
→ intent_logits[12], heads in-graph) — no host-side heads needed.

## Why QAT worked (the device-faithful sim)

The V68's HTP graph-prepare converts the attention matmul operands to a wide-range per-tensor
uint8 whose **effective precision is ~3 bits**, not 8. A naive 8-bit sim (AIMET) shows 100% and
**lies**; a **3-bit-attention sim reproduces the real 42.5%**. Training QAT against that
device-faithful sim recovered accuracy: PTQ 45.8% → 96.7% @100 steps → 100% @300. The checkpoint
is robust across attention bitwidths (fp32/8/4/3-bit = 100/100/97.5/100), which is why it survives
the real board. Recipe + scripts: `../../../conversion/intent_laya/qat/` (`qat.py`, `export_qat.py`,
`RESULTS_QAT.md`). Benchmark: `../../../bench/intent_htp_qat_device.json`.

## Honest caveats

- **100% = the quantized model matches fp32 on a HELD-OUT SYNTHETIC test** where fp32 is itself
  ~100% (saturated). It proves QAT removed the quantization loss on-device; it is **not** proof of
  real-world accuracy. Real gate needs ≥200 real recordings (5+ speakers, hall noise) — still open.
- **Slot head not yet scored on-device** (only intent top-1 was). Open check.

## Artifacts (on A100 `~/jero/intent/qnn/qat/`, too large for git)

`intent_qat_w8a16.dlc` (deployable w8a16 DLC), `intent_qat_maskfix.onnx`, `ckpt_qat3_best.pt`.
The context binary is built on-board from the DLC.
