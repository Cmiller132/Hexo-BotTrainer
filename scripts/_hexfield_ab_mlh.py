"""A/B the MLH search change on ONE checkpoint: NEW (two-sided bonus + final
pick, production) vs OLD (winning-side-only, no final pick = the prior live
behavior). Same model/evaluator on both seats — the ONLY difference is the
divergence_overrides — so this isolates the search change from weights/training.
Seats alternate for fairness. Reuses the tested _play_pair (per-seat overrides).

Usage: python _hexfield_ab_mlh.py CKPT.pt N_GAMES OUT_JSON
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
from hexfield.config import build_divergence_overrides, parse_hexfield_config  # noqa: E402
from hexfield.evaluation import _play_pair  # noqa: E402
from hexfield.inference import HexfieldEvaluator  # noqa: E402
from hexfield.model import HexfieldNet  # noqa: E402

CKPT = Path(sys.argv[1])
N = int(sys.argv[2])
OUT = Path(sys.argv[3])
CONFIG = REPO / "configs" / "hexfield_main_1.toml"

cfg = parse_hexfield_config(tomllib.loads(CONFIG.read_text(encoding="utf-8"))["model"]["config"])
sp = cfg.selfplay

NEW = build_divergence_overrides(sp)  # production: two-sided + final-pick ON
OLD = dict(NEW, ml_two_sided=False, ml_final_pick=False)  # prior live behavior
print("NEW:", NEW)
print("OLD:", OLD, flush=True)

model = HexfieldNet()
payload = torch.load(str(CKPT), map_location="cpu", weights_only=False)
model.load_state_dict(payload["model"], strict=True)
ev = HexfieldEvaluator(model, device=cfg.device)

new_wins = old_wins = draws = 0
lengths = []
for game in range(N):
    s1 = _rust.HexfieldMctsSession(max_states=65536)
    s2 = _rust.HexfieldMctsSession(max_states=65536)
    new_is_p0 = game % 2 == 0
    if new_is_p0:
        a, b, ov_a, ov_b = s1, s2, NEW, OLD
    else:
        a, b, ov_a, ov_b = s1, s2, OLD, NEW
    winner, plies = _play_pair(
        a, ev, b, ev, visits=cfg.evaluation.eval_visits, c_puct=sp.c_puct,
        seed=909_000 + game, sp=sp, max_plies=sp.max_game_plies,
        divergence_overrides=ov_a, divergence_overrides_b=ov_b,
    )
    lengths.append(plies)
    if winner is None:
        draws += 1
        tag = "draw"
    elif (winner == 0) == new_is_p0:
        new_wins += 1
        tag = "NEW"
    else:
        old_wins += 1
        tag = "OLD"
    print(f"  game {game+1}/{N}: {tag} plies={plies} | NEW={new_wins} OLD={old_wins} D={draws}", flush=True)

result = {
    "ckpt": CKPT.name, "games": N, "new_wins": new_wins, "old_wins": old_wins,
    "draws": draws, "new_win_rate": new_wins / max(N, 1),
    "mean_game_length": sum(lengths) / max(len(lengths), 1),
    "visits": cfg.evaluation.eval_visits,
}
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("RESULT:", json.dumps(result), flush=True)
