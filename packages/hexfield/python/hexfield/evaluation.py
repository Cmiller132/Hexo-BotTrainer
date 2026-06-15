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
from .config import ML_AUTO_DISABLED_FLAG, build_divergence_overrides, parse_hexfield_config
from .geometry import unpack_action_id
from .inference import HexfieldEvaluator
from .model import HexfieldNet


def _play_pair(session_a, eval_a, session_b, eval_b, *, visits, c_puct, seed, sp,
               max_plies, opening_plies=8, opening_temperature=1.0,
               divergence_overrides=None, divergence_overrides_b=None):
    """One game: A is player0. Returns (winner_int|None, plies).

    The first `opening_plies` moves are TEMPERATURE-SAMPLED (per-game seed) so
    the 16 eval games are independent lines rather than the same deterministic
    game repeated; after the opening, both sides play greedy (temperature 0) so
    the result measures pure strength. Sampling is symmetric (both models use it),
    so it does not bias the win rate."""

    state = api.new_game()
    sessions = (session_a, session_b)
    evaluators = (eval_a, eval_b)
    # MLH steered identically for both seats (symmetric -> unbiased win rate) by
    # default; an A/B can pass divergence_overrides_b to give seat B a different
    # search config (e.g. lever-off) for a head-to-head of the search change.
    ov_a = divergence_overrides if divergence_overrides is not None else build_divergence_overrides(sp)
    ov_b = divergence_overrides_b if divergence_overrides_b is not None else ov_a
    overrides_by_mover = (ov_a, ov_b)
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
            divergence_overrides=overrides_by_mover[mover],
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
    sp = cfg.selfplay
    mse_cfg = cfg.multi_stage_eval
    current = components.model.model
    started = time.time()
    result: dict[str, Any] = {"status": "completed", "epoch": epoch}

    # ===== Deep multistage strength eval (the COMPARABLE in-run eval) ==========
    # Replaces the old 16-game vs-immediately-prior arena (near-uninformative:
    # adjacent checkpoints are nearly identical). Plays games_budget (72) paired
    # games vs the FIXED roster (permanent anchors bc_prefit + ep5, the sliding
    # bracket, and the L-lag champion) at the LOCKED full_search_visits (512) and
    # eval_virtual_batch_size (16), pooling every edge into the persisted, append-
    # only Bradley-Terry pool so per-epoch games COMPOUND into an apples-to-apples
    # progress curve. PURE EVAL — it reports a PROMOTE/REGRESS/INCONCLUSIVE label
    # and writes ONLY its own diagnostics + the pool; it NEVER gates/halts/promotes
    # (eval_gating_enabled/eval_promotion_enabled stay False; the runner asserts it).
    # FAIL-SOFT: any error is recorded and NEVER breaks the training epoch (the
    # loop re-raises evaluate_epoch exceptions, which would kill the run).
    every = max(int(mse_cfg.every_n_epochs), 1)
    cand_path = ctx.checkpoint_dir / f"epoch_{epoch:06d}.pt"
    if not mse_cfg.enabled:
        result["multistage"] = {"status": "disabled"}
    elif epoch < 2:
        result["multistage"] = {"status": "skipped", "reason": "no opponent yet (epoch<2)"}
    elif epoch % every != 0:
        result["multistage"] = {"status": "skipped", "reason": f"every_n_epochs={every}"}
    elif not cand_path.is_file():
        result["multistage"] = {"status": "skipped", "reason": "candidate checkpoint missing"}
    else:
        try:
            from . import multistage_eval as mse  # lazy: pulls torch/GPU only here

            # CONCURRENT ONE-PASS eval: play SealBot + ALL checkpoint opponents in
            # ONE batched pass (one shared candidate forward across opponents), so
            # wall-clock is MAX over matches, not SUM (the old --parts path ran
            # opponents serially). Stage D statistics are UNCHANGED. PURE EVAL.
            report = mse.run_multistage_eval_concurrent(
                ctx.output_dir,
                cand_path,
                cfg,
                candidate_epoch=epoch,
                checkpoints_dir=ctx.checkpoint_dir,
                diagnostics_dir=ctx.diagnostics_dir,
                write_diagnostics=True,
            )
            meta = report.get("meta") or {}
            verdict = report.get("verdict") or {}
            result["multistage"] = {
                "status": "completed",
                "verdict": verdict.get("label"),
                "anchor": meta.get("anchor"),
                "elapsed_seconds": meta.get("elapsed_seconds"),
                "diagnostics_path": meta.get("diagnostics_path"),
            }
        except Exception as exc:  # FAIL-SOFT: the training epoch must survive.
            result["multistage"] = {"status": "error", "error": repr(exc)}
        finally:
            # The eval loaded opponent models in eval(); the live training model
            # (``current``) is untouched by the runner (candidate is reloaded from
            # disk), but restore train() defensively before the head audit.
            current.train()
    result["elapsed_seconds"] = round(time.time() - started, 1)
    # §5.4.4 head-health monitor + heal-gate: the MLH lever is only as safe as
    # the moves-left head. Audit the head on recent shards; if it would steer
    # search backward, drop the run-dir flag that forces the lever off next epoch
    # (build_divergence_overrides reads it); clear the flag once it heals.
    try:
        from .head_audit import audit_moves_left_head

        shards = sorted(
            ctx.samples_dir.glob("epoch_*/game_*.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        head = audit_moves_left_head(current, shards, device=cfg.device, max_games=40)
        flag = ctx.diagnostics_dir / ML_AUTO_DISABLED_FLAG
        if head.get("passed"):
            flag.unlink(missing_ok=True)
        elif sp.moves_left_utility:
            flag.write_text(json.dumps({"epoch": epoch, "audit": head}), encoding="utf-8")
        result["moves_left_head_audit"] = head
        result["ml_auto_disabled"] = bool(not head.get("passed") and sp.moves_left_utility)
        current.train()  # audit left the model in eval(); restore for next epoch
    except Exception as exc:  # a monitor failure must never break the eval epoch
        result["moves_left_head_audit"] = {"error": repr(exc)}
    diag_path = ctx.diagnostics_dir / f"hexfield.evaluation.epoch_{epoch:06d}.json"
    diag_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
