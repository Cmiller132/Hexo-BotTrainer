"""Strength read with the FIXED (opening-diverse) eval: checkpoint A vs B over
N independent-opening games. Validates the eval fix AND answers "did RL move
the model off the BC anchor?".

Usage: python scripts/_hexfield_anchor_arena.py <ckptA> <ckptB> [games] [visits]
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import torch

from hexfield import _rust
from hexfield.config import parse_hexfield_config
from hexfield.evaluation import _play_pair
from hexfield.inference import HexfieldEvaluator
from hexfield.model import HexfieldNet

ckpt_a, ckpt_b = sys.argv[1], sys.argv[2]
games = int(sys.argv[3]) if len(sys.argv) > 3 else 30
visits = int(sys.argv[4]) if len(sys.argv) > 4 else 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(p):
    m = HexfieldNet()
    payload = torch.load(p, map_location="cpu", weights_only=False)
    m.load_state_dict(payload.get("model", payload), strict=True)
    return HexfieldEvaluator(m, device=device)


ea, eb = load(ckpt_a), load(ckpt_b)
sp = parse_hexfield_config({"selfplay": {"max_game_plies": 200}}).selfplay
wins = losses = draws = 0
lengths = []
for g in range(games):
    a_is_p0 = g % 2 == 0
    sa = _rust.HexfieldMctsSession(max_states=65536)
    sb = _rust.HexfieldMctsSession(max_states=65536)
    s0, e0, s1, e1 = (sa, ea, sb, eb) if a_is_p0 else (sb, eb, sa, ea)
    winner, plies = _play_pair(s0, e0, s1, e1, visits=visits, c_puct=1.5,
                               seed=4242 + g * 13, sp=sp, max_plies=sp.max_game_plies,
                               opening_plies=8, opening_temperature=1.0)
    lengths.append(plies)
    if winner is None:
        draws += 1
    elif (winner == 0) == a_is_p0:
        wins += 1
    else:
        losses += 1

print(f"A={ckpt_a.split('/')[-1]}  B={ckpt_b.split('/')[-1]}")
print(f"games={games} visits={visits} opening_plies=8 (independent)")
print(f"A wins {wins} / B wins {losses} / draws {draws}  -> A win_rate {wins/games:.3f}")
print(f"mean game length {sum(lengths)/len(lengths):.1f}")
