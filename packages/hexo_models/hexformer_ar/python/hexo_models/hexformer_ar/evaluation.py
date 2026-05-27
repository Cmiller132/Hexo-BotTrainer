"""Epoch evaluation against SealBot for Hexformer AR checkpoints."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer
from hexo_runner.modes.match import run_match
from hexo_runner.session import GameSpec

from .benchmarks import benchmark_plan
from .player import HexformerPlayer


def evaluate_epoch(*, ctx: Any, components: Any, epoch: int) -> dict[str, Any]:
    trainer = components.model.trainer
    config = trainer.config
    eval_config = config.evaluation
    output_dir = ctx.output_dir / "evaluation" / f"hexformer_ar_epoch_{epoch:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx.diagnostics.write_json(f"hexformer_ar.benchmarks.epoch_{epoch:06d}.json", benchmark_plan())

    sealbot_config = SealBotConfig(
        variant=eval_config.sealbot_variant,
        time_limit=eval_config.sealbot_time_limit,
    )
    try:
        sealbot_config.validate()
    except Exception as exc:
        payload = {
            "status": "unavailable",
            "epoch": epoch,
            "reason": str(exc),
            "requested_games": eval_config.games_per_epoch,
            "sealbot_variant": eval_config.sealbot_variant,
            "sealbot_time_limit": eval_config.sealbot_time_limit,
            "required": eval_config.require_sealbot,
        }
        path = ctx.diagnostics.write_json(f"hexformer_ar.evaluation.epoch_{epoch:06d}.json", payload)
        if eval_config.require_sealbot:
            raise RuntimeError(f"Required SealBot evaluation is unavailable: {exc}") from exc
        return {**payload, "diagnostics_path": str(path)}

    wins = 0
    losses = 0
    completed = 0
    turns: list[int] = []
    ema_scope = trainer.ema_weights() if hasattr(trainer, "ema_weights") else nullcontext()
    with ema_scope:
        for game_index in range(eval_config.games_per_epoch):
            hexformer_is_p0 = game_index % 2 == 0
            hexformer = HexformerPlayer(
                identity_id="hexformer-ar-eval",
                model=components.model.model,
                trainer=trainer,
            )
            sealbot = SealBotPlayer(sealbot_config, player_id="sealbot-best-50ms")
            players = (hexformer, sealbot) if hexformer_is_p0 else (sealbot, hexformer)
            result = run_match(
                GameSpec(
                    game_id=f"hexformer-eval-{epoch:06d}-{game_index:04d}",
                    seed=(ctx.config.run.seed or 0) + epoch * 100_000 + game_index,
                    is_evaluation=True,
                    max_actions=eval_config.max_actions,
                ),
                players,  # type: ignore[arg-type]
                output_dir,
            )
            if str(result.status) == "completed":
                completed += 1
            turns.append(int(result.turns))
            hexformer_role = "player0" if hexformer_is_p0 else "player1"
            if result.winner == hexformer_role:
                wins += 1
            elif result.winner is not None:
                losses += 1

    diagnostics = {
        "status": "completed",
        "epoch": epoch,
        "model": "hexo_models.hexformer_ar",
        "games": eval_config.games_per_epoch,
        "completed": completed,
        "wins": wins,
        "losses": losses,
        "mean_turns": sum(turns) / max(1, len(turns)),
        "output_dir": str(output_dir),
    }
    path = ctx.diagnostics.write_json(f"hexformer_ar.evaluation.epoch_{epoch:06d}.json", diagnostics)
    return {**diagnostics, "diagnostics_path": str(path)}
