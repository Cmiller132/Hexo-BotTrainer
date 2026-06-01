#!/usr/bin/env bash
# Wait for epoch-1 selfplay (post-calibration), capture TRT-adopt + measure pos/s.
D=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
echo "=== waiting for epoch-1 selfplay to start (TRT build ~46s after calibration) ==="
for i in $(seq 1 30); do
  a=$(grep -h 'adopted TRT FP16' $D/diagnostics/trainer.*.out.log 2>/dev/null | tail -1)
  f=$(grep -h 'NOT ADOPTED\|TRTAdoptError\|FATAL' $D/diagnostics/trainer.*.out.log 2>/dev/null | tail -1)
  n=$(ls $D/selfplay/epoch_000001_game_*.npz 2>/dev/null | wc -l)
  if [ -n "$f" ]; then echo "TRT FAILED LOUD: $f"; exit 1; fi
  if [ -n "$a" ] || [ "$n" -gt 0 ]; then echo "selfplay active (t+$((i*15))s): shards=$n"; [ -n "$a" ] && echo "TRT: $a"; break; fi
  sleep 15
done
echo "=== TRT adopt line ==="; grep -h 'adopted TRT FP16' $D/diagnostics/trainer.*.out.log 2>/dev/null | tail -2
echo "=== pos/s over 60s (games completed x mean turns) ==="
n0=$(ls $D/selfplay/epoch_000001_game_*.npz 2>/dev/null | wc -l); t0=$(date +%s)
sleep 60
n1=$(ls $D/selfplay/epoch_000001_game_*.npz 2>/dev/null | wc -l); t1=$(date +%s)
# mean turns per game from completed shards' .hxr-free count: use npz 'turn_index' max+1 if present, else assume ~108
mt=$(python3 - "$D" <<'PY' 2>/dev/null
import glob,sys,numpy as np
fs=sorted(glob.glob(sys.argv[1]+"/selfplay/epoch_000001_game_*.npz"))[-3:]
ts=[]
for f in fs:
    try:
        z=np.load(f)
        k='turn_index' if 'turn_index' in z.files else None
        ts.append(int(z[k].max())+1 if k else len(z[z.files[0]]))
    except Exception: pass
print(int(sum(ts)/len(ts)) if ts else 108)
PY
)
mt=${mt:-108}
dg=$((n1-n0)); dt=$((t1-t0))
echo "shards $n0 -> $n1 (+$dg games) over ${dt}s; mean_turns~$mt -> pos/s ~ $(python3 -c "print(round($dg*$mt/$dt,1))" 2>/dev/null)"
echo "=== GPU sm duty (6s) ==="; nvidia-smi dmon -s u -c 6 2>/dev/null | tail -6
