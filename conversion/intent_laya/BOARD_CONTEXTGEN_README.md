# On-board context-binary generation — Jero intent (QCS6490 / HTP v68)

The host x86 `qnn-context-binary-generator` bakes in 4 MB VTCM, which FAILS to
load on the board's real 2 MB VTCM (err 5005) on v68. Therefore the final
context binary MUST be generated **on the board** from the quantized DLC.

The A100 box could NOT reach the board (`192.168.100.33`) during this build
(100% packet loss, SSH timeout — flaky VPN). Run the steps below once the board
is reachable.

## Inputs to copy to the board
- `intent_quantized.dlc`  (the w8a16 quantized DLC built on the A100)
- `htp_config.json`       (outer backend-extensions config, in this dir)
- `htp_backend_ext.json`  (inner HTP config: soc_id 498, dsp_arch v68, vtcm_mb 2, perf burst)

The DLC's internal graph name is **`intent_aimet_float`** (see `qairt-dlc-info`);
it is referenced in `htp_backend_ext.json -> graphs[].graph_names`.

## Steps (on the board: ubuntu@192.168.100.33, pw <REDACTED_BOARD_PW>)
```sh
# 1) stage to tmpfs first (never a direct eMMC burst write -> watchdog reset)
#    from the A100:
scp intent_quantized.dlc htp_config.json htp_backend_ext.json ubuntu@192.168.100.33:/tmp/jero_intent/

# 2) on the board
mkdir -p /tmp/jero_intent && cd /tmp/jero_intent
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

qnn-context-binary-generator \
    --backend     /usr/lib/libQnnHtp.so \
    --model       /usr/lib/libQnnModelDlc.so \
    --dlc_path    /tmp/jero_intent/intent_quantized.dlc \
    --config_file /tmp/jero_intent/htp_config.json \
    --output_dir  /tmp/jero_intent \
    --binary_file intent_w8a16_v68
# -> /tmp/jero_intent/intent_w8a16_v68.bin
```

### Acceptance gates for the generation step
- RC == 0  (exit 134 == an op the HTP cannot run -> on v68 almost always FP16)
- grep the stderr for `error|fp16|spill|fail` — in particular **`spill_bytes = 0`**
  (non-zero => the graph does not fit 2 MB VTCM and spills to DDR -> slow)

## Run / validate on the board
```sh
# input_list.txt lines:  input_ids:=<ids.raw> attention_mask:=<mask.raw>
# raws are INT32 [1,32] (the DLC converts the int64 ONNX inputs to int32)
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
qnn-net-run \
    --retrieve_context /tmp/jero_intent/intent_w8a16_v68.bin \
    --backend          /usr/lib/libQnnHtp.so \
    --config_file      /tmp/jero_intent/htp_config.json \
    --input_list       /tmp/jero_intent/verify_list.txt \
    --output_dir       /tmp/jero_intent/out \
    --log_level        warn
# outputs: out/Result_N/intent_logits.raw [1,12] float32, slot_logits.raw [1,32,17] float32
# compare argmax(intent_logits) to the FP32 ONNX reference (verify/fp32_ref.json)
```

NOTE: see the accuracy blocker in the main report — on v68 the attention BMM
operands are forced to 8-bit by the HTP backend, so this "w8a16" build does NOT
meet the >=99% top-1 gate in host simulation (~46%). Resolve that before relying
on the on-device numbers.
