#!/usr/bin/env bash
D=/mnt/e/Hexo-BotTrainer
C="$D/configs/dense_cnn_model1_target_96x6.toml"
RD="$D/runs/dense_cnn_model1_target_96x6"
echo "NOW:        $(date -u +%FT%TZ)"
echo "CONFIG_MTIME: $(stat -c %y "$C")"
echo "SO_MTIME:     $(stat -c %y "$D/packages/hexo_models/python/hexo_models/_rust.cpython-312-x86_64-linux-gnu.so")"
echo "LIVE_MTIME:   $(stat -c %y "$RD/diagnostics/dense_cnn.selfplay.live.json")"
echo "--- key config values ---"
grep -nE 'forced_playout_k|virtual_loss|search_visits|active_games|games_per_epoch|mcts_virtual_batch_candidates|selfplay_batch_candidates|mcts_session_cache_max_states|inference_use_tensorrt|resume_from|widening_max_children|temperature =' "$C"
echo "--- epoch2 shard count ---"
ls "$RD/selfplay/" | grep -c 'epoch_000002_game_.*npz'
echo "--- source files edited in last 6h (tracked-ish dirs) ---"
find "$D/packages" "$D/configs" "$D/scripts" -type f \( -name '*.py' -o -name '*.rs' -o -name '*.toml' -o -name '*.sh' \) -mmin -360 -not -path '*/target/*' -not -path '*__pycache__*' 2>/dev/null | head -30
