#!/usr/bin/env bash
LOG=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/rl_train.log
RUN=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2
python3 - "$LOG" <<'PY'
import sys, re
txt=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
# self-play Q metrics per epoch
sp=re.compile(r"epoch (\d+) selfplay: (\d+)C/(\d+)T games, (\d+) pos.*?cand=([\d.]+) \| Q1 decisive=([\d.]+)% len_med=([\d.]+) \| Q2 uniq_open=([\d.]+)% m2H=([\d.]+) \| Q3 visitH=([\d.]+) priorH=([\d.]+) \| Q4 \|val\|=([\d.]+) draw=([\d.]+)% \| Q5 forced=([\d.]+)%")
print(f"{'ep':>3} {'dec%':>5} {'lenmed':>6} {'uniqOpn%':>8} {'m2H':>5} {'visitH':>6} {'priorH':>6} {'|val|':>5} {'forced%':>7}")
sprows={}
for m in sp.finditer(txt):
    ep=int(m.group(1))
    sprows[ep]=dict(dec=float(m.group(6)),lenmed=float(m.group(7)),uniq=float(m.group(8)),m2H=float(m.group(9)),
                    visitH=float(m.group(10)),priorH=float(m.group(11)),absv=float(m.group(12)),forced=float(m.group(14)))
for ep in sorted(sprows):
    r=sprows[ep]
    print(f"{ep:>3} {r['dec']:>5.0f} {r['lenmed']:>6.0f} {r['uniq']:>8.1f} {r['m2H']:>5.2f} {r['visitH']:>6.2f} {r['priorH']:>6.2f} {r['absv']:>5.2f} {r['forced']:>7.1f}")
# eval trend + simple slope over last N
ev=[]
for m in re.finditer(r"epoch (\d+) EVAL vs dense_cnn e24: \d+W/\d+L/\d+D = ([\d.]+)%(?:.*?vs SealBot: \d+W/\d+L/\d+D = ([\d.]+)%)?", txt):
    ev.append((int(m.group(1)),float(m.group(2)),float(m.group(3)) if m.group(3) else None))
print("\n=== eval vs dense_cnn (and SealBot) ===")
print(" ".join(f"e{e}:{d:.0f}/{(s if s is not None else -1):.0f}" for e,d,s in ev))
if len(ev)>=4:
    import statistics
    xs=[e for e,_,_ in ev]; ys=[d for _,d,_ in ev]
    n=len(xs); mx=statistics.mean(xs); my=statistics.mean(ys)
    slope=sum((xs[i]-mx)*(ys[i]-my) for i in range(n))/sum((xs[i]-mx)**2 for i in range(n))
    print(f"dense win-rate slope = {slope:+.2f} %/epoch  (over e{xs[0]}-e{xs[-1]}; ±8% is the 40-game noise band)")
    last8=ys[-8:]; print(f"last 8 evals: mean={statistics.mean(last8):.1f}% min={min(last8):.0f} max={max(last8):.0f} stdev={statistics.pstdev(last8):.1f}")
    sl=[s for _,_,s in ev if s is not None]
    if sl: print(f"SealBot: mean={statistics.mean(sl):.1f}% range {min(sl):.0f}-{max(sl):.0f}")
PY
echo
echo "=== eval .hxr inventory (vs dense + vs sealbot dirs) ==="
ls -d "$RUN"/eval/epoch_*_vs_dense 2>/dev/null | tail -3
ls -d "$RUN"/eval/epoch_*_vs_sealbot 2>/dev/null | tail -3
echo "files in a recent vs_dense dir:"; ls "$RUN"/eval/epoch_000039_vs_dense/ 2>/dev/null | head;
echo "(count) vs_dense hxr in e39:"; ls "$RUN"/eval/epoch_000039_vs_dense/*.hxr 2>/dev/null | wc -l
