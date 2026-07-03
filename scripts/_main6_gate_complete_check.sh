#!/usr/bin/env bash
# Decision-identity check for HEXFIELD_GATE_COMPLETE: fixed-seed selfplay bench
# must produce identical decisions/flush stats with the gate on vs off.
# Requires the NEW .so (phase timing + gate) shimmed via PYTHONPATH (arg 1).
set -eu
SHIM=${1:?shim dir with hexfield/ package}
cd /mnt/e/Hexo-BotTrainer-hexgt
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4
export HEXFIELD_SERVE_FLEX=1 HEXFIELD_ASYNC_EVAL=1 HEXFIELD_DEFER_DECODE=1
export HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_SERVE_HALF=1 HEXFIELD_RUST_PACK=1
export PYTHONPATH="$SHIM":/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
PY=/root/.venvs/hexgt-build/bin/python
CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000072.pt

echo "=== gate OFF ==="
"$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 64 24 512 64 16 | tee /tmp/gate_off.txt
echo "=== gate ON ==="
HEXFIELD_GATE_COMPLETE=1 \
  "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 64 24 512 64 16 | tee /tmp/gate_on.txt
echo "=== diff (decisions/flush lines must match) ==="
diff <(grep -E 'decisions=|flushed_states|full/fast|early_stops' /tmp/gate_off.txt | sed 's/seconds=[0-9.]*//') \
     <(grep -E 'decisions=|flushed_states|full/fast|early_stops' /tmp/gate_on.txt | sed 's/seconds=[0-9.]*//') \
  && echo "IDENTICAL" || echo "DIVERGED"
