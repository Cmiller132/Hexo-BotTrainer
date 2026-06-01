"""Head-to-head evaluation for hexgt (Model 2).

Two pieces:

- `run_head_to_head` — the reusable, deterministic match driver. Pairs any two
  `hexo_runner` players over N games, ALTERNATING the first-player role so the
  result is not tied to one color, with fixed per-game seeds. Returns the score
  from player A's perspective. Used by the SealBot epoch eval below AND by the
  standalone hexgt-vs-dense_cnn / hexgt-vs-SealBot comparisons (matched-compute
  fairness per the rewrite plan).
- `evaluate_epoch` — the pipeline hook (mirrors `dense_cnn/evaluation.py`):
  turns the current hexgt checkpoint into a `HexgtPlayer`, pairs it against
  SealBot best-50ms when available, writes diagnostics, returns metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer, SealBotUnavailableError
from hexo_runner.modes.match import run_match
from hexo_runner.session import GameSpec

from .config import HexgtConfig
from .player import HexgtPlayer

# A player factory takes the per-game seed and returns a fresh runner player.
PlayerFactory = Callable[[int], Any]


@dataclass(frozen=True, slots=True)
class HeadToHeadResult:
    """Score from player A's perspective over a head-to-head match set."""

    games: int
    completed: int
    wins: int
    losses: int
    draws: int
    mean_turns: float

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "completed": self.completed,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "mean_turns": self.mean_turns,
        }


def run_head_to_head(
    make_a: PlayerFactory,
    make_b: PlayerFactory,
    *,
    games: int,
    output_dir: Any,
    base_seed: int = 0,
    max_actions: int = 1024,
    game_id_prefix: str = "h2h",
) -> HeadToHeadResult:
    """Play `games` matches A-vs-B, alternating colors, deterministic per seed.

    `make_a(seed)` / `make_b(seed)` build a FRESH player per game (a native MCTS
    session is per-game). Even games: A is player0; odd: A is player1. Wins are
    counted from A's perspective.
    """

    wins = losses = draws = completed = 0
    turns: list[int] = []
    for game_index in range(games):
        a_is_p0 = game_index % 2 == 0
        game_seed = base_seed + game_index
        player_a = make_a(game_seed)
        player_b = make_b(game_seed)
        players = (player_a, player_b) if a_is_p0 else (player_b, player_a)
        result = run_match(
            GameSpec(
                game_id=f"{game_id_prefix}-{game_index:04d}",
                seed=game_seed,
                is_evaluation=True,
                max_actions=max_actions,
            ),
            players,  # type: ignore[arg-type]
            output_dir,
        )
        if str(result.status) == "completed":
            completed += 1
        turns.append(int(result.turns))
        a_role = "player0" if a_is_p0 else "player1"
        if result.winner == a_role:
            wins += 1
        elif result.winner is not None:
            losses += 1
        else:
            draws += 1

    return HeadToHeadResult(
        games=games,
        completed=completed,
        wins=wins,
        losses=losses,
        draws=draws,
        mean_turns=sum(turns) / max(1, len(turns)),
    )


def make_hexgt_factory(
    model: Any,
    config: HexgtConfig,
    *,
    device: str,
    fp16: bool = True,
    identity_id: str = "hexgt",
    opening_temperature: float = 0.0,
    opening_moves: int = 0,
    virtual_batch_size: int = 0,
) -> PlayerFactory:
    """A `run_head_to_head` factory that builds a deterministic `HexgtPlayer`."""

    def factory(seed: int) -> HexgtPlayer:
        return HexgtPlayer(
            identity_id=identity_id,
            model=model,
            config=config,
            device=device,
            fp16=fp16,
            eval_seed=seed,
            opening_temperature=opening_temperature,
            opening_moves=opening_moves,
            virtual_batch_size=virtual_batch_size,
        )

    return factory


def evaluate_epoch(*, ctx: Any, components: Any, epoch: int) -> dict[str, Any]:
    """Run configured SealBot evaluation games for one checkpoint epoch."""

    trainer = components.model.trainer
    config: HexgtConfig = trainer.config
    model = components.model.model
    eval_config = config.evaluation
    device = str(next(model.parameters()).device)
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
        path = ctx.diagnostics.write_json(f"hexgt.evaluation.epoch_{epoch:06d}.json", payload)
        if eval_config.require_sealbot:
            raise RuntimeError(f"Required SealBot evaluation is unavailable: {exc}") from exc
        return {
            "status": "unavailable",
            "epoch": epoch,
            "reason": str(exc),
            "required": eval_config.require_sealbot,
            "diagnostics_path": str(path),
        }

    base_seed = (ctx.config.run.seed or 0) + epoch * 100_000
    make_hexgt = make_hexgt_factory(
        model,
        config,
        device=device,
        fp16=(device != "cpu"),
        identity_id="hexgt-eval",
        opening_temperature=eval_config.opening_temperature,
        opening_moves=eval_config.opening_moves,
        virtual_batch_size=eval_config.virtual_batch_size,
    )

    def make_sealbot(_seed: int) -> SealBotPlayer:
        return SealBotPlayer(sealbot_config, player_id="sealbot-best-50ms")

    try:
        result = run_head_to_head(
            make_hexgt,
            make_sealbot,
            games=eval_config.games_per_epoch,
            output_dir=output_dir,
            base_seed=base_seed,
            max_actions=eval_config.max_actions,
            game_id_prefix=f"eval-{epoch:06d}",
        )
    except SealBotUnavailableError as exc:
        payload = {
            "status": "unavailable",
            "epoch": epoch,
            "reason": str(exc),
            "required": eval_config.require_sealbot,
        }
        path = ctx.diagnostics.write_json(f"hexgt.evaluation.epoch_{epoch:06d}.json", payload)
        if eval_config.require_sealbot:
            raise RuntimeError(f"Required SealBot evaluation is unavailable: {exc}") from exc
        return {**payload, "diagnostics_path": str(path)}

    diagnostics = {
        "status": "completed",
        "epoch": epoch,
        "opponent": "sealbot-best-50ms",
        **result.as_dict(),
        "output_dir": str(output_dir),
    }
    path = ctx.diagnostics.write_json(f"hexgt.evaluation.epoch_{epoch:06d}.json", diagnostics)
    return {**diagnostics, "diagnostics_path": str(path)}
