#!/usr/bin/env bash
# GPU dev/test env for hexgnn (owner-authorized after the live run stopped).
# Same PYTHONPATH as the CPU env but does NOT hide the GPU, and sets the
# production allocator config. Usage: bash _hexgnn_gpu_env.sh <cmd...>
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
VENV=/root/.venvs/hexgt-build
PP="$ROOT/packages/hexo_engine/python"
PP="$PP:$ROOT/packages/hexo_utils/python"
PP="$PP:$ROOT/packages/hexo_runner/python"
PP="$PP:$ROOT/packages/hexo_train/python"
PP="$PP:$ROOT/packages/hexo_models/python"
PP="$PP:$ROOT/packages/hexgnn/python"
export PYTHONPATH="$PP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
exec "$VENV/bin/python" "$@"
