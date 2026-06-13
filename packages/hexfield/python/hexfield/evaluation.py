"""Per-epoch strength evaluation: head-to-head arena vs the previous epoch's
checkpoint (greedy lockstep searches; the §5.4 divergences run as in
production). Reports win rate + game lengths; written to diagnostics."""

from __future__ import annotations

import json
import time
from typing import Any

import torch

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from . import _rust
from .config import parse_hexfield_config
from .geometry import unpack_action_id
from .inference import HexfieldEvaluator
from .model import HexfieldNet


def _play_pair(session_a, eval_a, session_b, eval_b, *, visits, c_puct, seed, sp,
               max_plies, opening_plies=8, opening_temperature=1.0):
    """One game: A is player0. Returns (winner_int|None, plies).

    The first `opening_plies` moves are TEMPERATURE-SAMPLED (per-game seed) so
    the 16 eval games are independent lines rather than the same deterministic
    game repeated; after the opening, both sides play greedy (temperature 0) so
    the result measures pure strength. Sampling is symmetric (both models use it),
    so it does not bias the win rate."""

    state = api.new_game()
    sessions = (session_a, session_b)
    evaluators = (eval_a, eval_b)
    ply = 0
    while ply < max_plies:
        mover = 0 if ply == 0 else (1 if ((ply - 1) // 2) % 2 == 0 else 0)
        session = sessions[mover]
        evaluator = evaluators[mover]
        temperature = opening_temperature if ply < opening_plies else 0.0
        result = session.search(
            [seed], (state,), visits=visits, c_puct=c_puct, temperature=temperature,
            seed=seed * 5003 + ply, evaluator=evaluator,
            virtual_batch_size=8,
            widening_policy_mass=sp.widening_policy_mass,
            widening_max_children=sp.widening_max_children,
            widening_min_children=sp.widening_min_children,
            fpu_reduction=sp.fpu_reduction, tss_enabled=sp.tss_enabled,
            search_parity_mode=sp.search_parity_mode,
        )[0]
        q, r = unpack_action_id(int(result["action_id"]))
        outcome = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        ply += 1
        if outcome.terminal:
            terminal = api.terminal(state)
            return (0 if str(terminal.winner) == "player0" else 1), ply
    return None, ply


def evaluate_epoch(*, ctx, components, epoch: int) -> dict[str, Any]:
    cfg = parse_hexfield_config(ctx.config.model.config)
    games = cfg.evaluation.games_per_epoch
    every = max(int(cfg.evaluation.eval_every), 1)
    if games <= 0 or epoch < 2:
        return {"status": "skipped", "epoch": epoch, "reason": "no opponent yet"}
    if epoch % every != 0:
        return {"status": "skipped", "epoch": epoch, "reason": f"eval_every={every}"}
    # Compare against the most recent prior checkpoint that exists (eval_every
    # may mean it is several epochs back).
    prev_epoch = epoch - 1
    while prev_epoch >= 1 and not (ctx.checkpoint_dir / f"epoch_{prev_epoch:06d}.pt").is_file():
        prev_epoch -= 1
    prev_path = ctx.checkpoint_dir / f"epoch_{prev_epoch:06d}.pt"
    if prev_epoch < 1 or not prev_path.is_file():
        return {"status": "skipped", "epoch": epoch, "reason": "previous checkpoint missing"}

    current = components.model.model
    prev_model = HexfieldNet()
    payload = torch.load(prev_path, map_location="cpu", weights_only=False)
    prev_model.load_state_dict(payload["model"], strict=True)

    eval_cur = HexfieldEvaluator(current, device=cfg.device)
    eval_prev = HexfieldEvaluator(prev_model, device=cfg.device)
    sp = cfg.selfplay
    started = time.time()
    wins = losses = draws = 0
    lengths = []
    for game in range(games):
        s_cur = _rust.HexfieldMctsSession(max_states=65536)
        s_prev = _rust.HexfieldMctsSession(max_states=65536)
        cur_is_p0 = game % 2 == 0
        a, b = (s_cur, s_prev) if cur_is_p0 else (s_prev, s_cur)
        ea, eb = (eval_cur, eval_prev) if cur_is_p0 else (eval_prev, eval_cur)
        winner, plies = _play_pair(
            a, ea, b, eb, visits=cfg.evaluation.eval_visits, c_puct=sp.c_puct,
            seed=epoch * 100_000 + game, sp=sp, max_plies=sp.max_game_plies,
        )
        lengths.append(plies)
        if winner is None:
            draws += 1
        elif (winner == 0) == cur_is_p0:
            wins += 1
        else:
            losses += 1
    current.train()

    result = {
        "status": "completed",
        "epoch": epoch,
        "opponent": f"epoch_{prev_epoch:06d}",
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / max(games, 1),
        "mean_game_length": sum(lengths) / max(len(lengths), 1),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    diag_path = ctx.diagnostics_dir / f"hexfield.evaluation.epoch_{epoch:06d}.json"
    diag_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
