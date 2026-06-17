#!/usr/bin/env bash
D=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
rows() { python3 - "$D" <<'PY' 2>/dev/null
import glob,sys,numpy as np
tot=0
for f in glob.glob(sys.argv[1]+"/selfplay/epoch_000001_game_*.npz"):
    try:
        z=np.load(f); tot+=len(z[z.files[0]])
    except Exception: pass
print(tot)
PY
}
r0=$(rows); g0=$(ls $D/selfplay/epoch_000001_game_*.npz 2>/dev/null|wc -l); t0=$(date +%s)
sleep 120
r1=$(rows); g1=$(ls $D/selfplay/epoch_000001_game_*.npz 2>/dev/null|wc -l); t1=$(date +%s)
dt=$((t1-t0))
echo "sample_rows $r0 -> $r1 (+$((r1-r0))) ; games $g0 -> $g1 (+$((g1-g0))) over ${dt}s"
echo "live pos/s (sample-rows/s, ~positions/s) ~ $(python3 -c "print(round(($r1-$r0)/$dt,1))" 2>/dev/null)"
grep -h 'adopted TRT FP16' $D/diagnostics/trainer.*.out.log 2>/dev/null | tail -1
