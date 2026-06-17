#!/usr/bin/env bash
set -uo pipefail
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_64x4
python3 - "$RD/diagnostics/watch_wsl.jsonl" <<'PY'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
fr=[r.get("free_ram_gb") for r in rows if "free_ram_gb" in r]
print(f"samples={len(fr)} span={rows[0].get('t')} .. {rows[-1].get('t')}")
if fr:
    print(f"first={fr[0]:.2f}GB last={fr[-1]:.2f}GB min={min(fr):.2f}GB max={max(fr):.2f}GB")
    # coarse trajectory: every ~10th sample
    step=max(1,len(fr)//12)
    print("trajectory:", " ".join(f"{x:.1f}" for x in fr[::step]))
    # is the recent half lower than the first half? (leak indicator)
    h=len(fr)//2
    print(f"first-half avg={sum(fr[:h])/max(h,1):.2f}  second-half avg={sum(fr[h:])/max(len(fr)-h,1):.2f}")
PY
echo "=== RSS now (sample twice ~20s apart to see growth) ==="
tp=$(pgrep -f 'train_model .*64x4' | head -1)
r1=$(grep VmRSS /proc/$tp/status | awk '{print $2}')
sleep 20
r2=$(grep VmRSS /proc/$tp/status | awk '{print $2}')
echo "VmRSS t0=$((r1/1024))MB  t+20s=$((r2/1024))MB  delta=$(((r2-r1)/1024))MB"
echo "=== live ==="
python3 -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print('finished=%s/%s active=%s elapsed=%.0fs pos/s=%.1f'%(d.get('games_finished'),d.get('requested_games'),d.get('active_games'),d.get('elapsed_seconds',0),d.get('search_positions_per_second',0)))"
grep -E 'MemAvailable|SwapFree' /proc/meminfo