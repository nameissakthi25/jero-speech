# Intent — HTP (ModernBERT w8a16) — DEFERRED

Builds and runs on V68 (context binary, spill=0) but **on-device top-1 vs fp32 ≈ 41%** — a w8a16
**fidelity ceiling** (hidden-state r≈0.88). Fine for coarse typed decisions, not a 12-way argmax.
**Decision: intent runs on CPU.** HTP DLC/binary is on the RB3 (`~/intent_htp/`), not deployed.
Levers if revisited: sqnr calibration, QAT. See `../../../BENCHMARKS.md`.
