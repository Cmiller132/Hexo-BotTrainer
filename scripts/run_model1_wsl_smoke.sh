#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/e/Hexo-BotTrainer}"
VENV="${VENV:-/root/.venvs/hexo-bottrainer-wsl}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/dense_cnn_model1_wsl_smoke}"
CHECKPOINT="${CHECKPOINT:-$ROOT/data/checkpoints/dense_cnn_model1_latest.txt}"

ACTIVE_GAMES="${ACTIVE_GAMES:-1024}"
GAMES_PER_EPOCH="${GAMES_PER_EPOCH:-256}"
EVAL_GAMES="${EVAL_GAMES:-0}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-6}"
SELFPLAY_PROBE_POSITIONS="${SELFPLAY_PROBE_POSITIONS:-65536}"
INFERENCE_BATCH_CANDIDATES="${INFERENCE_BATCH_CANDIDATES:-512, 1024}"
SELFPLAY_BATCH_CANDIDATES="${SELFPLAY_BATCH_CANDIDATES:-$ACTIVE_GAMES}"
MIN_WSL_MEM_AVAILABLE_GB="${MIN_WSL_MEM_AVAILABLE_GB:-4}"
MIN_WSL_GPU_FREE_GB="${MIN_WSL_GPU_FREE_GB:-1.0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/root/.cache/hexo-bottrainer-cargo-target}"

if [[ ! -d "$VENV" ]]; then
  echo "Missing WSL venv: $VENV" >&2
  exit 2
fi

source "$VENV/bin/activate"
cd "$ROOT"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
SESSION="$RUN_ROOT/sessions/$STAMP"
mkdir -p "$SESSION"
CONFIG_PATH="$SESSION/config.toml"
LOG_PATH="$SESSION/train.log"
MONITOR_PATH="$SESSION/resource_monitor.jsonl"
GUARD_STOP_PATH="$SESSION/resource_guard.stop.json"
SUMMARY_PATH="$SESSION/summary.json"

python - "$CHECKPOINT" <<'PY' > "$SESSION/checkpoint.json"
import json
from pathlib import Path
import sys

import torch

def coerce_path(value: str, *, base: Path | None = None) -> Path:
    text = value.strip()
    if len(text) >= 3 and text[1] == ":" and text[2] in {"\\", "/"}:
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    path = Path(text)
    if path.is_absolute():
        return path
    return (base / path) if base is not None else path

path = coerce_path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"checkpoint missing: {path}")
if path.is_file() and path.suffix == ".txt":
    path = coerce_path(path.read_text(encoding="utf-8"), base=path.parent)
if not path.exists():
    raise SystemExit(f"resolved checkpoint missing: {path}")
payload = torch.load(path, map_location="cpu", weights_only=False)
epoch = int(payload.get("epoch") or 0)
if epoch < 1:
    raise SystemExit(f"checkpoint has invalid epoch: {payload.get('epoch')!r}")
print(json.dumps({"checkpoint": str(path), "epoch": epoch, "next_epoch": epoch + 1}, sort_keys=True))
PY

NEXT_EPOCH="$(python - "$SESSION/checkpoint.json" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["next_epoch"])
PY
)"
TARGET_EPOCHS="${LOOP_EPOCHS:-$NEXT_EPOCH}"
RESOLVED_CHECKPOINT="$(python - "$SESSION/checkpoint.json" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["checkpoint"])
PY
)"

cat > "$CONFIG_PATH" <<TOML
[model]
name = "dense_cnn"
module = "hexo_models.dense_cnn.plugin"

[model.config]
device = "cuda"

[model.config.architecture]
input_channels = 13
channels = 64
residual_blocks = 4
dropout = 0.0
short_term_value_horizons = [1, 4, 8]

[model.config.training]
batch_size = 128
learning_rate = 0.001
weight_decay = 0.0001
policy_weight = 1.0
value_weight = 1.0
opp_policy_weight = 0.25
lookahead_weight = 0.25
amp = true
max_grad_norm = 1.0
train_samples_per_epoch = 100000
max_train_bucket_per_new_data = 8.0
max_train_bucket_size = 500000
no_repeat_files = true
max_validation_samples = 100000

[model.config.samples]
compression_level = 6
shuffle_min_rows = 100000
shuffle_keep_target_rows = 600000
shuffle_taper_window_exponent = 0.65
shuffle_expand_window_per_row = 0.4
shuffle_taper_window_scale = 50000.0
approx_rows_per_out_file = 70000
shuffle_worker_group_size = 80000
validation_fraction = 0.0

[model.config.selfplay]
search_visits = 128
active_games = $ACTIVE_GAMES
mcts_session_cache_max_states = 1048576
mcts_active_root_limit = 1024
max_actions = 1024
temperature = 1.0

[model.config.evaluation]
games_per_epoch = $EVAL_GAMES
sealbot_variant = "best"
sealbot_time_limit = 0.05
max_actions = 1024
require_sealbot = false

[model.config.performance]
calibrate = true
target_selfplay_positions_per_second = 128.0
inference_batch_candidates = [128, 256, 512, 1024]
selfplay_batch_candidates = [1024]
training_batch_candidates = [64, 128, 192, 256]
mcts_virtual_batch_candidates = [4]
selfplay_probe_positions = $SELFPLAY_PROBE_POSITIONS
probe_batches = 1

[run]
name = "dense_cnn_model1_wsl_smoke"
output_dir = "$RUN_ROOT"
seed = 1

[loop]
epochs = $TARGET_EPOCHS

[selfplay]
games_per_epoch = $GAMES_PER_EPOCH
update_checkpoint_pointer = false
checkpoint_pointer = "$RUN_ROOT/selfplay_checkpoint.txt"

[train]
passes_per_epoch = 4

[checkpoint]
resume_from = "$RESOLVED_CHECKPOINT"
save_name = "wsl_smoke_latest"
TOML

python - <<'PY' > "$SESSION/static_checks.json"
import json
import tomllib
from pathlib import Path

import torch

from hexo_models.dense_cnn import BOARD_SIZE, INPUT_CHANNELS, parse_model1_config
from hexo_models.dense_cnn.samples import CURRENT_TARGET_SCHEMA_VERSION
from hexo_models.dense_cnn.rust_bridge import capabilities

raw = tomllib.loads(Path("configs/dense_cnn_model1.toml").read_text(encoding="utf-8"))
cfg = parse_model1_config(raw["model"]["config"])
caps = dict(capabilities())

assert torch.cuda.is_available(), "CUDA is not available to PyTorch inside WSL"
assert BOARD_SIZE == 41 and INPUT_CHANNELS == 13
assert cfg.selfplay.max_actions == 1024
assert cfg.evaluation.max_actions == 1024
assert cfg.selfplay.search_visits == 128
assert caps["model1_mcts_policy_nucleus_widening"]
assert caps["model1_mcts_tree_reuse_session"]

print(json.dumps({
    "status": "ok",
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_device": torch.cuda.get_device_name(0),
    "cuda_mem_info": torch.cuda.mem_get_info(),
    "board_size": BOARD_SIZE,
    "input_channels": INPUT_CHANNELS,
    "target_schema_version": CURRENT_TARGET_SCHEMA_VERSION,
    "capabilities": caps,
}, indent=2, sort_keys=True))
PY

echo "$SESSION" > "$RUN_ROOT/latest_session.txt"

python -m hexo_train.cli.train_model "$CONFIG_PATH" > "$LOG_PATH" 2>&1 &
TRAIN_PID="$!"
echo "$TRAIN_PID" > "$SESSION/pid"
START_SECONDS="$(date +%s)"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  python - "$TRAIN_PID" "$GUARD_STOP_PATH" "$MIN_WSL_MEM_AVAILABLE_GB" "$MIN_WSL_GPU_FREE_GB" <<'PY' >> "$MONITOR_PATH" || true
import json
import os
import subprocess
import sys
import signal
import time

pid = sys.argv[1]
guard_stop_path = sys.argv[2]
min_mem_available_gb = float(sys.argv[3])
min_gpu_free_gb = float(sys.argv[4])

def gb_from_kb(value: int) -> float:
    return round(value / 1024 / 1024, 3)

def status_kb(name: str) -> int:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(name + ":"):
                    return int(line.split()[1])
    except FileNotFoundError:
        return 0
    return 0

mem = {}
with open("/proc/meminfo", "r", encoding="utf-8") as handle:
    for line in handle:
        key, rest = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            mem[key] = int(rest.split()[0])

try:
    gpu_line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    ).strip().splitlines()[0]
    used, free, util, temp = [part.strip() for part in gpu_line.split(",")]
    gpu = {
        "used_gb": round(int(used) / 1024, 3),
        "free_gb": round(int(free) / 1024, 3),
        "util_percent": int(util),
        "temperature_c": int(temp),
    }
except Exception as exc:
    gpu = {"error": repr(exc)}

row = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "pid": int(pid),
    "rss_gb": gb_from_kb(status_kb("VmRSS")),
    "virtual_gb": gb_from_kb(status_kb("VmSize")),
    "mem_total_gb": gb_from_kb(mem.get("MemTotal", 0)),
    "mem_available_gb": gb_from_kb(mem.get("MemAvailable", 0)),
    "swap_free_gb": gb_from_kb(mem.get("SwapFree", 0)),
    "gpu": gpu,
}
critical = []
if row["mem_available_gb"] < min_mem_available_gb:
    critical.append(f"mem_available_gb < {min_mem_available_gb}")
if isinstance(gpu, dict) and "free_gb" in gpu and float(gpu["free_gb"]) < min_gpu_free_gb:
    critical.append(f"gpu_free_gb < {min_gpu_free_gb}")
row["critical"] = critical
print(json.dumps(row, sort_keys=True))
if critical:
    stop = dict(row)
    stop["status"] = "stopping_trainer"
    with open(guard_stop_path, "w", encoding="utf-8") as handle:
        json.dump(stop, handle, indent=2, sort_keys=True)
    try:
        os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
PY
  sleep "$MONITOR_INTERVAL_SECONDS"
done

set +e
wait "$TRAIN_PID"
RETURN_CODE="$?"
set -e
END_SECONDS="$(date +%s)"

python - "$RETURN_CODE" "$START_SECONDS" "$END_SECONDS" "$SESSION" "$LOG_PATH" "$MONITOR_PATH" "$SUMMARY_PATH" <<'PY'
import json
from pathlib import Path
import sys

return_code = int(sys.argv[1])
start_seconds = int(sys.argv[2])
end_seconds = int(sys.argv[3])
session = Path(sys.argv[4])
log_path = Path(sys.argv[5])
monitor_path = Path(sys.argv[6])
summary_path = Path(sys.argv[7])

peaks = {
    "rss_gb": 0.0,
    "virtual_gb": 0.0,
    "gpu_used_gb": 0.0,
    "min_mem_available_gb": None,
    "min_gpu_free_gb": None,
}
if monitor_path.exists():
    for line in monitor_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        peaks["rss_gb"] = max(peaks["rss_gb"], float(row.get("rss_gb") or 0.0))
        peaks["virtual_gb"] = max(peaks["virtual_gb"], float(row.get("virtual_gb") or 0.0))
        available = row.get("mem_available_gb")
        if available is not None:
            peaks["min_mem_available_gb"] = (
                float(available)
                if peaks["min_mem_available_gb"] is None
                else min(peaks["min_mem_available_gb"], float(available))
            )
        gpu = row.get("gpu") or {}
        if "used_gb" in gpu:
            peaks["gpu_used_gb"] = max(peaks["gpu_used_gb"], float(gpu["used_gb"]))
        if "free_gb" in gpu:
            peaks["min_gpu_free_gb"] = (
                float(gpu["free_gb"])
                if peaks["min_gpu_free_gb"] is None
                else min(peaks["min_gpu_free_gb"], float(gpu["free_gb"]))
            )

payload = {
    "return_code": return_code,
    "elapsed_seconds": end_seconds - start_seconds,
    "session": str(session),
    "log_path": str(log_path),
    "monitor_path": str(monitor_path),
    "peaks": peaks,
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

exit "$RETURN_CODE"
