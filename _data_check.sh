#!/bin/bash
R=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main
echo "=== old-run data inventory (read-only source) ==="
echo "selfplay shards: $(ls $R/selfplay/*.npz 2>/dev/null | wc -l)"
echo "example trace files: $(ls $R/eval/epoch_*_examples.json 2>/dev/null | wc -l)"
echo "epochs of shards:"; ls $R/selfplay/*.npz 2>/dev/null | sed -E 's/.*epoch_0*([0-9]+)_game.*/\1/' | sort -un | tr '\n' ' '; echo
echo "=== one example trace move (fields) ==="
python3 -c "import json; g=json.load(open('$R/eval/epoch_000000_examples.json'))[1]; print('game keys:',list(g.keys())); print('move sample:', g['moves'][2])" 2>/dev/null
