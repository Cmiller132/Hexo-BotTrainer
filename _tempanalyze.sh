#!/usr/bin/env bash
CSV=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/gpu_telemetry.csv
python3 - "$CSV" <<'PY'
import sys, statistics as st
rows=[]
for ln in open(sys.argv[1]):
    p=[x.strip() for x in ln.split(',')]
    if len(p)<9 or p[0].startswith('timestamp'): continue
    try:
        ts=p[0]; temp=float(p[1]); pw=float(p[2].split()[0]); sm=float(p[4].split()[0])
        util=float(p[6].split()[0]); mem=float(p[7].split()[0]); thr=p[8]
        rows.append((ts,temp,pw,sm,util,mem,thr))
    except Exception: pass
n=len(rows)
print(f"samples={n}  (~{n} s at 1Hz)")
if not n: sys.exit()
last=rows[-1]
print(f"window: {rows[0][0]}  ->  {last[0]}")
# last ~600s
w=rows[-600:] if n>600 else rows
temps=[r[1] for r in w]; utils=[r[4] for r in w]; pws=[r[2] for r in w]
print(f"\n=== TEMP over last {len(w)}s ===")
print(f"current={last[1]:.0f}C  min={min(temps):.0f}  max={max(temps):.0f}  avg={st.mean(temps):.1f}  stdev={st.pstdev(temps):.1f}")
print(f"util:  min={min(utils):.0f}%  max={max(utils):.0f}%  avg={st.mean(utils):.1f}%")
print(f"power: min={min(pws):.0f}W  max={max(pws):.0f}W  avg={st.mean(pws):.0f}W")
# peak over WHOLE log
allt=[r[1] for r in rows]
print(f"\n=== PEAK over whole log ({n}s): max temp = {max(allt):.0f}C  min = {min(allt):.0f}C ===")
# throttle flags ever non-zero?
nz=[r for r in rows if r[6] not in ('0x0000000000000000','0x0')]
print(f"throttle (clocks_throttle_reasons.active) non-zero samples: {len(nz)}")
for r in nz[:5]: print("   ", r[0], "thr=", r[6], "temp=", r[1])
# swing characterization: count direction changes + amplitude in last 120s
w2=rows[-120:] if n>120 else rows
t2=[r[1] for r in w2]
ups=downs=0
for i in range(1,len(t2)):
    if t2[i]>t2[i-1]: ups+=1
    elif t2[i]<t2[i-1]: downs+=1
print(f"\n=== SWING (last {len(t2)}s) ===  range {min(t2):.0f}-{max(t2):.0f}C (span {max(t2)-min(t2):.0f}C), up-ticks={ups} down-ticks={downs}")
# temp-vs-util correlation (does temp track load?)
import math
if len(w)>10:
    tm=st.mean(temps); um=st.mean(utils)
    cov=sum((temps[i]-tm)*(utils[i]-um) for i in range(len(w)))/len(w)
    sdt=st.pstdev(temps); sdu=st.pstdev(utils)
    corr=cov/(sdt*sdu) if sdt>0 and sdu>0 else 0
    print(f"corr(temp,util) over {len(w)}s = {corr:.2f}  (|r|>0.5 => temp tracks load)")
# show a 20s strip so the pattern is visible: temp/util/power
print("\n=== last 20 samples (ts tail | temp util% powerW thr) ===")
for r in rows[-20:]:
    print(f"  {r[0][-12:]}  {r[1]:>3.0f}C  {r[4]:>3.0f}%  {r[2]:>3.0f}W  {r[6][-4:]}")
PY
echo "=== thermal limits (Tlimit/shutdown/slowdown) ==="
nvidia-smi -q -d TEMPERATURE | grep -iE "GPU Current Temp|Shutdown|Slowdown|GPU Max|GPU T.Limit|Target" | head -8
