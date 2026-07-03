#!/usr/bin/env bash
# Wipe the newest (partial) main_6 epoch's selfplay outputs before a restart,
# per the documented supervisor procedure. Prints what it removes.
set -eu
RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6
LAST=$(ls -d "$RUN"/samples/epoch_* | sort | tail -1)
EP=$(basename "$LAST")
echo "wiping partial $EP"
ls "$LAST" | head -3 || true
rm -f "$LAST"/game_*.npz "$LAST"/game_*.json
rm -f "$RUN"/selfplay/"$EP".hxr "$RUN"/selfplay/"$EP"_resume*.hxr
echo "remaining in $LAST: $(ls "$LAST" 2>/dev/null | wc -l) files"
echo "hxr count now: $(ls "$RUN"/selfplay/*.hxr | wc -l)"
