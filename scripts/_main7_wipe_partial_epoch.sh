#!/usr/bin/env bash
# Wipe the newest (partial) main_7 epoch's selfplay outputs before a restart —
# same documented procedure as _wipe_partial_epoch.sh for main_6. Refuses to
# run if that epoch's diagnostic exists (i.e. the epoch is complete).
set -eu
RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_7
LAST=$(ls -d "$RUN"/samples/epoch_* | sort | tail -1)
EP=$(basename "$LAST")
if [ -f "$RUN/diagnostics/$EP.json" ]; then
  echo "$EP is COMPLETE (diagnostic exists) — nothing to wipe"; exit 0
fi
echo "wiping partial $EP"
rm -f "$LAST"/game_*.npz "$LAST"/game_*.json
rm -f "$RUN"/selfplay/"$EP".hxr "$RUN"/selfplay/"$EP"_resume*.hxr
rmdir "$LAST" 2>/dev/null || true
echo "hxr count now: $(ls "$RUN"/selfplay/*.hxr 2>/dev/null | wc -l)"
