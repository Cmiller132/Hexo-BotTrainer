RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
echo "=== self-play NPZ shards written? (where dense_cnn stores them) ==="
echo "selfplay/: $(ls "$RD/selfplay" 2>/dev/null | wc -l) entries | *.npz: $(ls "$RD/selfplay"/*.npz 2>/dev/null | wc -l) | *.hxr: $(ls "$RD/selfplay"/*.hxr 2>/dev/null | wc -l)"
ls "$RD/selfplay" 2>/dev/null | head -4
echo "=== samples/ + shuffleddata/ (the replay window) ==="
du -sh "$RD/samples" "$RD/shuffleddata" 2>/dev/null
ls -R "$RD/shuffleddata" 2>/dev/null | head -10
find "$RD" -name "*.npz" 2>/dev/null | head -5
echo "  total .npz anywhere in run dir: $(find "$RD" -name '*.npz' 2>/dev/null | wc -l)"
echo "=== epoch-0 selfplay diag: recorded/sample counts? ==="
/root/.venvs/hexgt-build/bin/python -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.epoch_000001.json'));print({k:d.get(k) for k in d if any(t in k.lower() for t in ('sample','record','row','npz','shard','written','effective','raw'))})" 2>/dev/null
echo "=== epoch 1 self-play live ==="
/root/.venvs/hexgt-build/bin/python -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print('epoch',d.get('epoch'),d.get('status'),'games',d.get('games_finished'),'pos/s',round(d.get('positions_per_second',0),1))" 2>/dev/null
