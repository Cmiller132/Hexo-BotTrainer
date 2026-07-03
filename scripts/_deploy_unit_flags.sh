#!/usr/bin/env bash
# Add the serve-speedup env flags to hexfield-supervisor-6.service (idempotent),
# then daemon-reload and start.
set -eu
UNIT=/etc/systemd/system/hexfield-supervisor-6.service
if grep -q HEXFIELD_TRITON_CONV "$UNIT"; then
  echo "flags already present"
else
  # Insert after the HEXFIELD_ANCHOR_ROOTS line, with a provenance comment.
  sed -i '/^Environment=HEXFIELD_ANCHOR_ROOTS=/a \
# Serve-speedup stack (2026-07-02, docs/analysis/MAIN6_SERVE_SPEEDUP_2026-07-02.md):\
# fused gather+GEMM triton conv + precomputed-pair flex score_mod + fp16 serve\
# copy + Rust parallel pack. ~2.2x serve forward, +63% self-play pos/s at live\
# batching. serve-half value parity 4.7e-3 (vs the 3e-3 flex-era gate) accepted\
# by the operator 2026-07-02; SealBot multistage eval (every 5 epochs) guards\
# strength. Training path untouched by all four flags.\
Environment=HEXFIELD_TRITON_CONV=1\
Environment=HEXFIELD_FLEX_PAIR=1\
Environment=HEXFIELD_SERVE_HALF=1\
Environment=HEXFIELD_RUST_PACK=1' "$UNIT"
  echo "flags added"
fi
systemctl daemon-reload
systemctl start hexfield-supervisor-6
sleep 5
systemctl is-active hexfield-supervisor-6
grep -A2 -B1 HEXFIELD_TRITON_CONV "$UNIT" | head -12
