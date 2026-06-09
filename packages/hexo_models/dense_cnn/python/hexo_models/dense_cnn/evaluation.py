"""Epoch evaluation adapter for dense CNN checkpoints.

The generic trainer asks the model plugin to evaluate an epoch. This module
turns the current dense CNN model into a `hexo_runner` player, pairs it against
SealBot when available, and returns runner/evaluation metadata. It does not
generate training samples and does not mutate the replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer, SealBotUnavailableError
from hexo_runner.modes.match import run_match
from hexo_runner.player import PlayerFactory
from hexo_runner.session import GameSpec

from .player import DenseCNNPlayer


@dataclass(frozen=True, slots=True)
class _DenseFactory:
    model: object
    trainer: object
    player_id: str

    def create_player(self) -> DenseCNNPlayer:
        return DenseCNNPlayer(
            identity_id=self.player_id,
            model=self.model,
            trainer=self.trainer,
            record_samples=False,
        )


@dataclass(frozen=True, slots=True)
class _SealBotFactory:
    config: SealBotConfig
    player_id: str

    def create_player(self) -> SealBotPlayer:
        return SealBotPlayer(self.config, player_id=self.player_id)


def evaluate_epoch(*, ctx: Any, components: Any, epoch: int) -> dict[str, Any]:
    """Run configured SealBot evaluation games for one checkpoint epoch."""

    trainer = components.model.trainer
    config = trainer.config
    eval_config = config.evaluation

    # Eval cadence (eval_every): when >1, only run the SealBot eval on epochs where
    # epoch % eval_every == 0; otherwise skip it (the eval is the dominant ~15-min
    # per-epoch cost). Behavior-preserving for eval_every in {0, 1} (every epoch).
    # No eval diagnostic is written on a skip, so the dashboard eval-trend (which
    # globs dense_cnn.evaluation.epoch_*.json) simply has a gap; the epoch result
    # carries this skipped status for the per-epoch row.
    eval_every = int(getattr(eval_config, "eval_every", 0) or 0)
    if eval_every > 1 and (epoch % eval_every) != 0:
        return {
            "status": "skipped",
            "epoch": epoch,
            "reason": f"eval cadence: every {eval_every} epochs (epoch % {eval_every} != 0)",
            "games": 0,
            "completed": 0,
            "wins": 0,
            "losses": 0,
            "mean_turns": None,
        }

    output_dir = ctx.output_dir / "evaluation" / f"epoch_{epoch:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        path = ctx.diagnostics.write_json(
            f"dense_cnn.evaluation.epoch_{epoch:06d}.json",
            payload,
        )
        if eval_config.require_sealbot:
            raise RuntimeError(f"Required SealBot evaluation is unavailable: {exc}") from exc
        return {
            "status": "unavailable",
            "epoch": epoch,
            "reason": str(exc),
            "required": eval_config.require_sealbot,
            "diagnostics_path": str(path),
        }

    wins = 0
    losses = 0
    completed = 0
    turns: list[int] = []
    for game_index in range(eval_config.games_per_epoch):
        # Alternate colors so the evaluation result is not tied to one fixed
        # first-player role.
        dense_is_p0 = game_index % 2 == 0
        game_seed = (ctx.config.run.seed or 0) + epoch * 100_000 + game_index
        dense = DenseCNNPlayer(
            identity_id="dense-cnn-eval",
            model=components.model.model,
            trainer=trainer,
            record_samples=False,
            eval_seed=game_seed,
            opening_temperature=eval_config.opening_temperature,
            opening_moves=eval_config.opening_moves,
        )
        sealbot = SealBotPlayer(sealbot_config, player_id="sealbot-best-50ms")
        players = (dense, sealbot) if dense_is_p0 else (sealbot, dense)
        result = run_match(
            GameSpec(
                game_id=f"eval-{epoch:06d}-{game_index:04d}",
                seed=game_seed,
                is_evaluation=True,
                max_actions=eval_config.max_actions,
            ),
            players,  # type: ignore[arg-type]
            output_dir,
        )
        if str(result.status) == "completed":
            completed += 1
        turns.append(int(result.turns))
        dense_role = "player0" if dense_is_p0 else "player1"
        if result.winner == dense_role:
            wins += 1
        elif result.winner is not None:
            losses += 1

    diagnostics = {
        "status": "completed",
        "epoch": epoch,
        "games": eval_config.games_per_epoch,
        "completed": completed,
        "wins": wins,
        "losses": losses,
        "mean_turns": sum(turns) / max(1, len(turns)),
        "output_dir": str(output_dir),
    }
    path = ctx.diagnostics.write_json(f"dense_cnn.evaluation.epoch_{epoch:06d}.json", diagnostics)
    return {**diagnostics, "diagnostics_path": str(path)}
