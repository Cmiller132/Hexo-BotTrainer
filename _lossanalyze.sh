#!/usr/bin/env bash
LOG=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/rl_train.log
EVAL=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/eval
python3 - "$LOG" "$EVAL" <<'PY'
import sys, re, json, glob, os
log, evald = sys.argv[1], sys.argv[2]
txt=open(log,encoding='utf-8',errors='ignore').read()
# train lines: capture per-head losses + window/pool. stv terms are optional.
pat=re.compile(r"epoch (\d+) train: \d+ steps \(step=\d+\) total=([\d.]+) policy=([\d.]+) value=([\d.]+) opp=([\d.]+)(?P<stv>(?: stv\d+=[\d.]+)*) .*?window=(\d+)ep\[[^\]]*\] \d+sh pool~(\d+)k")
print(f"{'ep':>3} {'total':>7} {'policy':>7} {'value':>7} {'opp':>7} {'stv4':>7} {'stv12':>7} {'stv24':>7} {'win':>4} {'poolk':>6}")
rows={}
for m in pat.finditer(txt):
    ep=int(m.group(1))
    stv=dict(re.findall(r"stv(\d+)=([\d.]+)", m.group('stv')))
    rows[ep]=dict(total=float(m.group(2)),policy=float(m.group(3)),value=float(m.group(4)),
                  opp=float(m.group(5)),stv4=stv.get('4'),stv12=stv.get('12'),stv24=stv.get('24'),
                  win=int(m.group(7)),poolk=int(m.group(8)))
# eval win-rates vs dense_cnn (from log EVAL lines, robust)
ev={}
for m in re.finditer(r"epoch (\d+) EVAL vs dense_cnn e24: \d+W/\d+L/\d+D = ([\d.]+)%", txt):
    ev[int(m.group(1))]=float(m.group(2))
def f(x): return f"{x:7.4f}" if isinstance(x,float) else (f"{float(x):7.4f}" if x else "      -")
for ep in sorted(rows):
    r=rows[ep]; w=ev.get(ep)
    print(f"{ep:>3} {r['total']:7.4f} {r['policy']:7.4f} {r['value']:7.4f} {r['opp']:7.4f} "
          f"{f(r['stv4'])} {f(r['stv12'])} {f(r['stv24'])} {('%4.0f'%w) if w is not None else '   -':>4} {r['poolk']:>6}")
# deltas: policy/value pre vs post STV (STV came online when stv4 first non-None)
stv_start=min((ep for ep in rows if rows[ep]['stv4'] is not None), default=None)
print(f"\nSTV first logged at epoch {stv_start}")
if stv_start is not None and stv_start-1 in rows:
    a,b=rows[stv_start-1],rows[max(rows)]
    print(f"policy  ep{stv_start-1}={a['policy']:.4f} -> ep{max(rows)}={b['policy']:.4f}  (delta {b['policy']-a['policy']:+.4f})")
    print(f"value   ep{stv_start-1}={a['value']:.4f} -> ep{max(rows)}={b['value']:.4f}  (delta {b['value']-a['value']:+.4f})")
    print(f"opp     ep{stv_start-1}={a['opp']:.4f} -> ep{max(rows)}={b['opp']:.4f}  (delta {b['opp']-a['opp']:+.4f})")
    # stv slope
    s=[(ep,float(rows[ep]['stv4'])) for ep in sorted(rows) if rows[ep]['stv4']]
    if len(s)>=2:
        print(f"stv4    ep{s[0][0]}={s[0][1]:.4f} -> ep{s[-1][0]}={s[-1][1]:.4f}  (delta {s[-1][1]-s[0][1]:+.4f} over {s[-1][0]-s[0][0]} epochs)")
PY
