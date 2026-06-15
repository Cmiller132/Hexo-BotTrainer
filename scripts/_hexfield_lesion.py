"""M10 divergence lesion study (spec §10 M10): per-divergence arena A/B at
matched visits. Both players share ONE checkpoint/evaluator; they differ only
in search config (all §5.4 divergences on vs one lesioned off). A's win rate
> 50% => that divergence helps. For attribution + constant tuning, NOT a ship
gate. Also calibrates the moves-left auto-disable threshold.

Usage (WSL, GPU free): PYTHONPATH=packages/hexfield/python \
  /root/.venvs/hexgt-build/bin/python scripts/_hexfield_lesion.py <ckpt.pt> [games] [visits]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import torch

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield import _rust
from hexfield.geometry import unpack_action_id
from hexfield.inference import HexfieldEvaluator
from hexfield.model import HexfieldNet

DIVERGENCES = ["lcb_move_selection", "early_stop", "visit_scaled_c_puct", "moves_left_utility"]


def _search_kwargs(visits):
    return dict(
        visits=visits, c_puct=1.5, temperature=0.0, virtual_batch_size=8,
        widening_policy_mass=0.95, widening_max_children=96, widening_min_children=2,
        fpu_reduction=0.2, tss_enabled=True,
    )


def play_game(evaluator, visits, overrides_p0, overrides_p1, seed):
    """One game; player k uses overrides_pk (None => all divergences on)."""
    state = api.new_game()
    sess = [_rust.HexfieldMctsSession(max_states=65536),
            _rust.HexfieldMctsSession(max_states=65536)]
    overrides = [overrides_p0, overrides_p1]
    ply = 0
    while ply < 512:
        mover = 0 if ply == 0 else (1 if ((ply - 1) // 2) % 2 == 0 else 0)
        kw = dict(_search_kwargs(visits))
        # Opening diversity (first 8 plies temperature-sampled, symmetric) so
        # the games are independent lines, not one deterministic game repeated.
        kw["temperature"] = 1.0 if ply < 8 else 0.0
        if overrides[mover] is not None:
            kw["divergence_overrides"] = overrides[mover]
        r = sess[mover].search([0], (state,), seed=seed * 7919 + ply, evaluator=evaluator, **kw)[0]
        q, rr = unpack_action_id(int(r["action_id"]))
        out = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=rr)))
        ply += 1
        if out.terminal:
            return 0 if str(api.terminal(state).winner) == "player0" else 1, ply
    return None, ply


def arena(evaluator, visits, games, a_overrides, b_overrides, label):
    wins = losses = draws = 0
    lengths = []
    t0 = time.time()
    for g in range(games):
        a_is_p0 = g % 2 == 0
        o0, o1 = (a_overrides, b_overrides) if a_is_p0 else (b_overrides, a_overrides)
        winner, plies = play_game(evaluator, visits, o0, o1, seed=1000 + g)
        lengths.append(plies)
        if winner is None:
            draws += 1
        elif (winner == 0) == a_is_p0:
            wins += 1
        else:
            losses += 1
    return {
        "label": label, "games": games, "A_wins": wins, "B_wins": losses, "draws": draws,
        "A_win_rate": round(wins / max(games, 1), 3),
        "mean_len": round(sum(lengths) / max(len(lengths), 1), 1),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ckpt = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    visits = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexfieldNet()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model", payload), strict=True)
    evaluator = HexfieldEvaluator(model, device=device)
    print(f"lesion study: ckpt={ckpt} games={games} visits={visits} device={device}")

    # All-on baseline self-consistency (should be ~50%).
    print(arena(evaluator, visits, games, None, None, "all_on_vs_all_on(sanity~50%)"))
    # Each divergence: A=all on, B=that one off. A>50% => divergence helps.
    for div in DIVERGENCES:
        b = {div: False}
        print(arena(evaluator, visits, games, None, b, f"all_on_vs_{div}_off"))
    # Composite: all on vs all off (search_parity-equivalent divergence set).
    all_off = {d: False for d in DIVERGENCES}
    print(arena(evaluator, visits, games, None, all_off, "all_on_vs_all_off"))


if __name__ == "__main__":
    main()
