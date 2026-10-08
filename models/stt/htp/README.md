# STT — HTP (Zipformer w8a16 on Hexagon V68)

Encoder + decoder + joiner run **w8a16 on the V68 HTP**. Context binaries
(`{encoder,decoder,joiner}_w8a16_ctx.bin`) are built **on-board** (2 MB VTCM) and live on the
RB3 at `~/zipformer_test/` — too large / device-baked for git.

**Verified:** token-exact match to the float decode (correlation **1.000**) — "CHANGE LANGUAGE TO HINDI".
Encoder ~16.4 ms/chunk, RTF ~0.05. Driver: `~/zipformer_run/htp_decode.py` (pure-Python, qnn-net-run).
