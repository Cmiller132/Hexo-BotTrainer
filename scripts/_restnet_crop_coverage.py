"""Gating check for a 29x29 crop: replay real HF human games and measure, at every
position, the max |dq|/|dr| offset of any legal move / stone / history / hot cell
from the dense crop center. A 29x29 crop (half=14) keeps facts with |dq|<=14 AND
|dr|<=14; 41x41 keeps |.|<=20. Reports the clip rates so we know whether shrinking
to 29 would drop legal moves in normal play."""
import sys, json
from pathlib import Path
ROOT = Path("/mnt/e/Hexo-BotTrainer-hexgt")
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models"):
    sys.path.insert(0, str(ROOT / "packages" / p / "python"))
sys.path.insert(0, str(ROOT / "packages" / "dense_cnn_restnet" / "python"))

import hexo_engine as engine
from hexo_engine.types import AxialCoord, PlacementAction
from dense_cnn_restnet.samples import sample_from_state
from dense_cnn_restnet.d6 import unpack_coord_pair

JSONL = ROOT / "runs/dense_cnn_restnet_main1_prefit/hf_corpus/hexo_human_corpus.jsonl"
MAX_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

def offsets(sample):
    cq, cr = sample.center
    m = 0
    # legal moves (the frontier — most spread out)
    for aid in sample.legal_action_ids:
        q, r = unpack_coord_pair(int(aid))
        m = max(m, abs(q - cq), abs(r - cr))
    # placed stones
    for q, r, _p in sample.stones:
        m = max(m, abs(int(q) - cq), abs(int(r) - cr))
    return m

games = []
with JSONL.open() as fh:
    for line in fh:
        if line.strip():
            games.append(json.loads(line))
        if len(games) >= MAX_GAMES:
            break

import collections
hist = collections.Counter()
n_pos = 0
clip29 = 0   # positions with a fact at offset >14 (dropped by 29, kept by 41)
clip41 = 0   # positions with a fact at offset >20 (dropped even by 41)
played_out29 = 0  # the PLAYED move (BC policy target) falls outside 29x29
played_out41 = 0  # the PLAYED move falls outside 41x41 (poisons even the 41 prefit)
worst = 0
for g in games:
    state = engine.new_game(seed=0)
    for ti, mv in enumerate(g["moves"]):
        x, y = int(mv[0]), int(mv[1])
        action = PlacementAction(AxialCoord(q=x, r=y))
        if not engine.is_legal_action(state, action):
            break
        s = sample_from_state(state, game_id=g.get("game_hash", "x"), turn_index=ti,
                              policy={int(engine.action_id(action)): 1.0},
                              root_prior_policy={int(engine.action_id(action)): 1.0})
        m = offsets(s)
        hist[m] += 1
        n_pos += 1
        worst = max(worst, m)
        if m > 14:
            clip29 += 1
        if m > 20:
            clip41 += 1
        cq, cr = s.center
        pmax = max(abs(x - cq), abs(y - cr))  # played-move offset from center
        if pmax > 14:
            played_out29 += 1
        if pmax > 20:
            played_out41 += 1
        engine.apply_action(state, action)

print(f"games={len(games)} positions={n_pos} worst_offset={worst}")
print(f"ANY-FACT clip@29 (offset>14): {clip29} = {100*clip29/max(1,n_pos):.2f}%")
print(f"ANY-FACT clip@41 (offset>20): {clip41} = {100*clip41/max(1,n_pos):.2f}%")
print(f"PLAYED-MOVE out@29 (target dropped): {played_out29} = {100*played_out29/max(1,n_pos):.2f}%")
print(f"PLAYED-MOVE out@41 (target dropped): {played_out41} = {100*played_out41/max(1,n_pos):.2f}%")
print("offset histogram (offset: count):")
for k in sorted(hist):
    bar = "#" * min(60, hist[k] * 60 // max(hist.values()))
    print(f"  {k:2d}: {hist[k]:7d} {bar}")
# percentiles
cum = 0
flat = sorted(hist.items())
for pct in (0.5, 0.9, 0.99, 0.999, 1.0):
    target = pct * n_pos
    c = 0
    for k, v in flat:
        c += v
        if c >= target:
            print(f"  p{pct*100:5.1f} offset = {k}")
            break
