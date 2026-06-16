#!/usr/bin/env bash
set -uo pipefail
source /root/.venvs/hexo-bottrainer-wsl/bin/activate
ROOT=/mnt/e/Hexo-BotTrainer
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
cd "$ROOT"
echo "=== config parse (64x4) + schedule helper ==="
python - <<'PY'
import tomllib
from hexo_models.dense_cnn.config import parse_model1_config
from hexo_models.dense_cnn.selfplay import _move_temperature
c = parse_model1_config(tomllib.load(open("configs/dense_cnn_model1_target_64x4.toml","rb"))["model"]["config"])
sp = c.selfplay
print("arch:", c.architecture.channels, "x", c.architecture.residual_blocks)
print("temp:", sp.temperature, "final:", sp.final_temperature, "decay_moves:", sp.temperature_decay_moves)
print("validation_fraction:", c.samples.validation_fraction)
sched = [round(_move_temperature(m, initial=sp.temperature, final=sp.final_temperature, decay_moves=sp.temperature_decay_moves),3) for m in (0,10,20,30,45,100)]
print("schedule @ plies 0,10,20,30,45,100:", sched)
assert sched[0]==1.0 and sched[3]==0.2 and sched[-1]==0.2, sched
print("OK schedule")
PY
echo ""
echo "=== pytest: bridge signature + selfplay loop ==="
python -m pytest -q \
  tests/test_dense_cnn_sample_generation.py::test_mcts_session_search_forwards_slim_signature \
  tests/test_dense_cnn_pipeline.py::test_selfplay_searches_every_playable_move_until_games_finish \
  2>&1 | tail -25
echo "=== pytest rc=${PIPESTATUS[0]} ==="
