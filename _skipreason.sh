RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
$V -c "import json;d=json.load(open('$RD/diagnostics/epoch_000002.json'));r=d['metadata']['result']
t=r.get('training',{}); print('TRAIN:',{k:t.get(k) for k in ('status','reason')})
s=r.get('samples',{}); sel=s.get('selection',{}); sh=sel.get('metadata',{}).get('shuffle',{})
print('SELECTION:',{k:sel.get(k) for k in ('status','reason')})
print('SHUFFLE:',{k:sh.get(k) for k in ('status','reason','output_rows','total_num_data_rows','used_rows','window_start_data_row_idx','shuffle_dir')})
fin=s.get('finalize',{}); print('FINALIZE:',{k:fin.get(k) for k in ('status','reason')})" 2>/dev/null
echo "--- replay/window config ---"
grep -nE "replay_window|train_window|min.*rows|window|max_train_bucket|passes_per_epoch|samples_per" configs/dense_cnn_rl_main1.toml | head
echo "--- shuffleddata + selfplay shard totals ---"
echo "shuffleddata: $(find "$RD/shuffleddata" -name '*.npz' 2>/dev/null | wc -l) npz | selfplay: $(ls "$RD/selfplay"/*game*.npz 2>/dev/null | wc -l) npz across epochs"
ls "$RD/selfplay"/*game*.npz 2>/dev/null | sed -E 's/.*epoch_0*([0-9]+)_game.*/epoch \1/' | sort | uniq -c
