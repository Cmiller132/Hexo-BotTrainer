"""Epoch evaluation against SealBot for dense CNN checkpoints."""

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
    trainer = components.model.trainer
    config = trainer.config
    eval_config = config.evaluation
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
        dense_is_p0 = game_index % 2 == 0
        dense = DenseCNNPlayer(
            identity_id="dense-cnn-eval",
            model=components.model.model,
            trainer=trainer,
            record_samples=False,
        )
        sealbot = SealBotPlayer(sealbot_config, player_id="sealbot-best-50ms")
        players = (dense, sealbot) if dense_is_p0 else (sealbot, dense)
        result = run_match(
            GameSpec(
                game_id=f"eval-{epoch:06d}-{game_index:04d}",
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
