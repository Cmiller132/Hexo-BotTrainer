"""Fixed-baseline strength probe for hexfield: checkpoint A vs checkpoint B.

The production auto-eval only plays each epoch vs the IMMEDIATELY-PRIOR epoch
(every 5 epochs), which cannot distinguish "model is genuinely improving" from
"self-play targets are getting harder (longer games) so loss drifts up". This
runs head-to-head between two ARBITRARY checkpoints, reusing hexfield's tested
_play_pair protocol verbatim (same visits/c_puct/widening/opening-sampling as
production eval), so the win rate is an apples-to-apples strength comparison.

Usage:
  python _hexfield_fixed_baseline_arena.py CKPT_A CKPT_B N_GAMES OUT_JSON
A wins are reported as win_rate_a (>0.5 => A stronger than B).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
sys.path.insert(0, str(REPO / "packages" / "dense_cnn_restnet" / "python"))

import torch  # noqa: E402

torch.set_grad_enabled(False)

import tomllib  # noqa: E402

from hexfield import _rust  # noqa: E402
from hexfield.config import parse_hexfield_config  # noqa: E402
from hexfield.evaluation import _play_pair  # noqa: E402  (reuse tested protocol)
from hexfield.inference import HexfieldEvaluator  # noqa: E402
from hexfield.model import HexfieldNet  # noqa: E402

CKPT_A, CKPT_B = Path(sys.argv[1]), Path(sys.argv[2])
N_GAMES = int(sys.argv[3])
OUT = Path(sys.argv[4])
CONFIG = REPO / "configs" / "hexfield_main_1.toml"

cfg = parse_hexfield_config(tomllib.loads(CONFIG.read_text(encoding="utf-8"))["model"]["config"])
sp = cfg.selfplay


def load(path):
    model = HexfieldNet()
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return HexfieldEvaluator(model, device=cfg.device)


eval_a, eval_b = load(CKPT_A), load(CKPT_B)
print(f"A={CKPT_A.name}  B={CKPT_B.name}  games={N_GAMES}  "
      f"visits={cfg.evaluation.eval_visits}  c_puct={sp.c_puct}", flush=True)

wins_a = wins_b = draws = 0
lengths = []
for game in range(N_GAMES):
    s_a = _rust.HexfieldMctsSession(max_states=65536)
    s_b = _rust.HexfieldMctsSession(max_states=65536)
    a_is_p0 = game % 2 == 0
    # seat A (p0) gets whichever model is A_is_p0; mirror evaluate_epoch exactly.
    first, second = (s_a, s_b) if a_is_p0 else (s_b, s_a)
    efirst, esecond = (eval_a, eval_b) if a_is_p0 else (eval_b, eval_a)
    winner, plies = _play_pair(
        first, efirst, second, esecond,
        visits=cfg.evaluation.eval_visits, c_puct=sp.c_puct,
        seed=777_000 + game, sp=sp, max_plies=sp.max_game_plies,
    )
    lengths.append(plies)
    if winner is None:
        draws += 1
    elif (winner == 0) == a_is_p0:
        wins_a += 1
    else:
        wins_b += 1
    print(f"  game {game+1}/{N_GAMES}: winner={'A' if (winner==0)==a_is_p0 and winner is not None else ('draw' if winner is None else 'B')} "
          f"plies={plies}  running A={wins_a} B={wins_b} D={draws}", flush=True)

result = {
    "ckpt_a": CKPT_A.name, "ckpt_b": CKPT_B.name, "games": N_GAMES,
    "wins_a": wins_a, "wins_b": wins_b, "draws": draws,
    "win_rate_a": wins_a / max(N_GAMES, 1),
    "mean_game_length": sum(lengths) / max(len(lengths), 1),
    "visits": cfg.evaluation.eval_visits,
}
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("RESULT:", json.dumps(result), flush=True)
