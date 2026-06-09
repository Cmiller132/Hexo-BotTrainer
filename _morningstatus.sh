#!/usr/bin/env bash
WT=/mnt/e/Hexo-BotTrainer-hexgt; RUN=$WT/runs/hexgt_rl_main2
echo "=== processes ==="
echo "supervisor: $(pgrep -af _rl_supervise | grep -v pgrep | head -1 | cut -c1-40)"
echo "driver:     $(pgrep -f _rl_train.py | head -1)  start=$(ps -o lstart= -p "$(pgrep -f _rl_train.py|head -1)" 2>/dev/null)"
echo "bridge:     $(pgrep -fc _dashboard_bridge.py)   telemetry: $(pgrep -fc 'query-gpu=timestamp,temperature')"
echo "=== current epoch / phase ==="
tail -1 "$RUN/rl_train.log"
echo "ep43 sp: $(ls $RUN/selfplay/epoch_000043_game_*.npz 2>/dev/null|wc -l)/256  ep44 sp: $(ls $RUN/selfplay/epoch_000044_game_*.npz 2>/dev/null|wc -l)/256"
echo "=== crashes overnight (EXCEPTION/INCOMPLETE/device-not-ready since 23:00) + supervisor EXIT codes ==="
grep -cE "EXCEPTION|device not ready" "$RUN/rl_train.log"
grep -E "EXIT pid=.*code=" "$RUN/supervisor.log" | tail -4
echo "=== crash_diag bundles written? ==="
ls -dt "$RUN/crash_artifacts"/crash_diag_* 2>/dev/null | head -5 || echo "(none)"
echo "=== telemetry: peak temp + any NON-idle throttle flag overnight ==="
python3 - "$RUN/gpu_telemetry.csv" <<'PY'
import sys
mx=0; mxp=0; nidle=0; n=0; bad=[]
for ln in open(sys.argv[1]):
    p=[x.strip() for x in ln.split(',')]
    if len(p)<9 or p[0].startswith('timestamp'): continue
    try:
        t=float(p[1]); pw=float(p[2].split()[0]); thr=p[8]; n+=1
        mx=max(mx,t); mxp=max(mxp,pw)
        if thr not in ('0x0000000000000000','0x0'):
            nidle+=1
            if thr!='0x0000000000000001' and len(bad)<5: bad.append((p[0],thr,t))
    except: pass
print(f"samples={n}  peak_temp={mx:.0f}C  peak_power={mxp:.0f}W")
print(f"non-zero-throttle samples={nidle} (0x1=idle, benign)")
print(f"NON-IDLE throttle (thermal/power/HW) samples: {len(bad)}", bad if bad else "(none — no thermal/power throttle)")
PY
echo "=== GPU now + RAM ==="
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.active --format=csv,noheader
awk '/MemAvailable/{printf "RAM free %.1f GB\n", $2/1048576}' /proc/meminfo
