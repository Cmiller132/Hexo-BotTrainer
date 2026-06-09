#!/usr/bin/env bash
cd /mnt/e/Hexo-BotTrainer-hexgt && source /root/.venvs/hexgt-build/bin/activate
echo "=== self-play summary (RL epoch 0 = display epoch 1) ==="
cat runs/hexgt_rl_main2/eval/epoch_000000_selfplay.json 2>/dev/null | python3 -m json.tool 2>/dev/null | grep -iE "completed_games|truncated|searched_positions|mcts_sim|raw_samples|length|mean|max_actions" | head
python3 - <<'PY'
import json, statistics as st
from pathlib import Path
import hexo_engine.api as eng
from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexo_models.hexgt.expand import reconstruct_state
spd = Path("runs/hexgt_rl_main2/selfplay")
games = sorted(spd.glob("epoch_000000_game_*.npz"))[:120]
rows_per=[]; deepest=[]; term_at_deepest=0; legal_remaining=[]
for g in games:
    try: rs=read_compact_shard(g)
    except: continue
    if not rs: continue
    rows_per.append(len(rs))
    d=max(rs,key=lambda r:len(r.placement_history))
    deepest.append(len(d.placement_history))
    s=reconstruct_state(list(d.placement_history))
    t=eng.terminal(s)
    if t is not None: term_at_deepest+=1
    else: legal_remaining.append(len(list(eng.legal_actions(s))))
def stats(x):
    x=sorted(x); n=len(x)
    return f"n={n} min={x[0]} med={x[n//2]} mean={sum(x)/n:.1f} max={x[-1]}"
print("rows/game (sampled positions):", stats(rows_per))
print("deepest placement_history (=replay length):", stats(deepest))
print(f"deepest-sampled position is TERMINAL: {term_at_deepest}/{len(games)} games")
if legal_remaining:
    print(f"  for non-terminal-at-deepest: legal moves still available median={sorted(legal_remaining)[len(legal_remaining)//2]} (game continued past the last sampled position)")
# value decisiveness at deepest
dec=0
for g in games[:60]:
    try: rs=read_compact_shard(g)
    except: continue
    d=max(rs,key=lambda r:len(r.placement_history))
    if abs(float(d.value))>0.5: dec+=1
print(f"decisive (|value|>0.5) at deepest row: {dec}/60")
PY
