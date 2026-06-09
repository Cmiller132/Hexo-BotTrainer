#!/usr/bin/env bash
WT=/mnt/e/Hexo-BotTrainer-hexgt
MAIN=/mnt/e/Hexo-BotTrainer
echo "=== 1a. DRIVER argv ==="
p=$(pgrep -f _rl_train.py | head -1)
echo "driver pid=$p start=$(ps -o lstart= -p "$p" 2>/dev/null)  alive"
tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -oE -- "--max-actions [0-9]+|--eval-every [0-9]+|--visits [0-9]+|--active [0-9]+"
echo "=== 1b. LIVE CODE has the changes (worktree imported via PYTHONPATH) ==="
echo -n "recency window builder: "; grep -c "def build_replay_window\|rng.choices(shards, weights=weights" $WT/scripts/_rl_train.py
echo -n "recency-decay/pool-cap args: "; grep -c "replay-recency-decay\|replay-pool-cap" $WT/scripts/_rl_train.py
echo -n "per-round seed: "; grep -c "search_round\|base_seed + epoch" $WT/packages/hexo_models/hexgt/python/hexo_models/hexgt/selfplay.py
echo -n "full .hxr writer: "; grep -c "_write_game_record\|finish_completed" $WT/packages/hexo_models/hexgt/python/hexo_models/hexgt/selfplay.py
echo -n "8 features + schema v2: "; grep -c "F_CAND_OWN_WIN3 = 22\|FEATURE_SCHEMA_VERSION = 2" $WT/packages/hexo_models/hexgt/python/hexo_models/hexgt/constants.py
echo -n "widening default 96: "; grep -c "widening-max-children\", type=int, default=96" $WT/scripts/_rl_train.py
echo "=== 2a. TIMING: has ep9 (post-deploy) trained yet? ==="
grep -E "epoch 9 train|epoch 9 done" $WT/runs/hexgt_rl_main2/rl_train.log || echo "ep9 NOT trained yet (still self-play/train) -> no recency train-line yet"
echo "ep9 selfplay: $(ls $WT/runs/hexgt_rl_main2/selfplay/epoch_000009_game_*.npz 2>/dev/null | wc -l)/256"
echo "=== 2b. PIPELINE wired? (so it WILL render once ep9 trains) ==="
echo -n "bridge running pid: "; pgrep -f _dashboard_bridge.py | grep -v pgrep | head -1
echo -n "bridge has _buffer_stats: "; grep -c "_buffer_stats\|_BUF_RE" $WT/_dashboard_bridge.py
echo -n "8080 proc pid: "; pgrep -f 'hexo_frontend.web.*8080' | head -1
echo -n "8080 serves buffer-stats app.js: "; curl -s localhost:8080/ | grep -oE "app.js\?v=[0-9a-z-]+" | head -1
echo -n "web.py buffer passthrough present (MAIN, served): "; grep -c '"buffer": payload.get("buffer")' $MAIN/packages/hexo_frontend/python/hexo_frontend/web.py
echo "=== mirror epoch JSONs: any buffer field yet? ==="
for f in $MAIN/runs/hexgt_rl_main2/diagnostics/dense_cnn.selfplay.epoch_0000{09,10}.json; do
  [ -f "$f" ] && echo "$(basename $f): buffer=$(grep -o '"buffer"' "$f" | head -1 || echo none)"
done
echo "=== regex match test against the EXACT format _rl_train will emit ==="
python3 - <<'PY'
import re
BUF = re.compile(r"epoch (\d+) train: (\d+) steps \(step=\d+\) total=([\d.]+) policy=([\d.]+) value=([\d.]+).*?window=(\d+)ep\[([-\d]+)\] (\d+)sh pool~(\d+)k/(\d+)k decay=([\d.]+)")
sample = "epoch 9 train: 512 steps (step=4922) total=3.1234 policy=1.8001 value=0.7002 opp=3.3108 prune=0.0% 254ms/step window=10ep[0-9] 2560sh pool~150k/500k decay=0.9"
m = BUF.search(sample)
print("MATCH" if m else "NO MATCH", "->", m.groups() if m else None)
PY
