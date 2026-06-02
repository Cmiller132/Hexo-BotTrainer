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
from typing import Any, Callable, Protocol, Sequence

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer, SealBotUnavailableError
from hexo_runner.modes.match import run_match
from hexo_runner.session import GameSpec

from .config import HexgtConfig
from .inference import HexgtInference
from .mcts import new_mcts_session
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


class BatchedSearcher(Protocol):
    """A deterministic side that searches a BATCH of live states at once.

    The whole point of the parallel eval: many games' current positions are
    searched together so the GPU sees one fat batched forward per round instead
    of one game at a time. `search_batch` returns the chosen action id per state
    (greedy / deterministic); `discard` drops a finished game's cached subtree;
    `reset` clears all state between matches.
    """

    def search_batch(self, keys: Sequence[int], states: Sequence[object], *, seed: int, plies: Sequence[int] | None = None) -> list[int]: ...
    def discard(self, key: int) -> None: ...
    def reset(self) -> None: ...


class HexgtBatchedSearcher:
    """Batched hexgt searcher backed by ONE persistent native session (subtree
    reuse across rounds). Greedy + NO Dirichlet by default, so a parallel match
    reproduces the sequential one (batch composition is invariant under a
    deterministic forward + greedy selection).

    OPENING VARIETY (optional): for the first `opening_moves` plies it samples the
    action at `opening_temperature` instead of greedy. This decorrelates the
    eval games (pure-greedy on fixed seeds replays near-identical lines, so the
    win rate is lumpy); the sampling is per-(round,game) seeded so it stays
    REPEATABLE across epochs (a valid paired comparison) and independent of GPU
    batch composition. Greedy resumes after the opening, so the bulk of each game
    still measures skill, not noise."""

    def __init__(self, model: Any, config: HexgtConfig, *, device: str, fp16: bool, visits: int,
                 opening_moves: int = 0, opening_temperature: float = 0.0) -> None:
        self.config = config
        self.visits = int(visits)
        self.opening_moves = int(opening_moves)
        self.opening_temperature = float(opening_temperature)
        self.inference = HexgtInference(model, device=device, fp16=fp16)
        self.session = new_mcts_session(
            max_states=config.selfplay.mcts_session_cache_max_states,
            n=config.architecture.candidate_radius,
        )

    def search_batch(self, keys: Sequence[int], states: Sequence[object], *, seed: int,
                     plies: Sequence[int] | None = None) -> list[int]:
        sp = self.config.selfplay
        move_temps: list[float] | None = None
        if self.opening_moves > 0 and self.opening_temperature > 0.0 and plies is not None:
            move_temps = [self.opening_temperature if int(p) < self.opening_moves else 0.0 for p in plies]
        results = self.session.run(
            list(keys), list(states), self.inference,
            visits=self.visits, c_puct=sp.c_puct, temperature=0.0, seed=seed,
            virtual_batch_size=None, active_root_limit=sp.mcts_active_root_limit,
            root_policy_temperature=sp.root_policy_temperature, fpu_reduction=sp.fpu_reduction,
            virtual_loss=sp.virtual_loss, widening_policy_mass=sp.widening_policy_mass,
            widening_max_children=sp.widening_max_children, widening_min_children=sp.widening_min_children,
            move_temperatures=move_temps,
        )
        return [int(r.action_id) for r in results]

    def discard(self, key: int) -> None:
        self.session.discard(int(key))

    def reset(self) -> None:
        self.session.clear()


def run_head_to_head_parallel(
    searcher_a: BatchedSearcher,
    searcher_b: BatchedSearcher,
    *,
    games: int,
    base_seed: int = 0,
    max_actions: int = 1024,
    active_limit: int | None = None,
) -> HeadToHeadResult:
    """Play `games` A-vs-B matches with MANY games in flight at once.

    Each round, the active games are partitioned by whose turn it is and each
    side's batch of positions is searched in ONE call (fat GPU forward), then the
    chosen actions are applied. Alternates colors (even games: A=player0), fixed
    per-game seeds, greedy/no-noise -> the per-game result is invariant to the
    async batch composition (verified against the sequential driver). `active_limit`
    caps concurrency (keep == the self-play active_games to size the GPU the same).
    """

    searcher_a.reset()
    searcher_b.reset()
    limit = int(active_limit or games)
    wins = losses = draws = completed = 0
    turns: list[int] = []
    next_game = 0
    a_keys: set[int] = set()
    b_keys: set[int] = set()
    active: list[dict[str, Any]] = []
    round_idx = 0  # per-round seed offset -> reproducible opening sampling

    while next_game < games or active:
        while len(active) < limit and next_game < games:
            seed = base_seed + next_game
            active.append({
                "i": next_game, "state": engine.new_game(seed=seed),
                "a_is_p0": next_game % 2 == 0, "plies": 0,
            })
            next_game += 1

        a_grp: list[dict[str, Any]] = []
        b_grp: list[dict[str, Any]] = []
        finished: list[dict[str, Any]] = []
        for g in active:
            term = engine.terminal(g["state"])
            if term is not None or g["plies"] >= max_actions:
                g["winner"] = (
                    str(getattr(term.winner, "value", term.winner))
                    if term is not None and term.winner is not None else None
                )
                g["truncated"] = term is None
                finished.append(g)
                continue
            mover_is_a = (str(engine.current_player(g["state"])) == "player0") == g["a_is_p0"]
            (a_grp if mover_is_a else b_grp).append(g)

        for g in finished:
            turns.append(g["plies"])
            a_role = "player0" if g["a_is_p0"] else "player1"
            if g["winner"] == a_role:
                wins += 1
            elif g["winner"] is not None:
                losses += 1
            else:
                draws += 1
            if not g.get("truncated"):
                completed += 1
            active.remove(g)
            if g["i"] in a_keys:
                searcher_a.discard(g["i"]); a_keys.discard(g["i"])
            if g["i"] in b_keys:
                searcher_b.discard(g["i"]); b_keys.discard(g["i"])

        for searcher, grp, keyset in ((searcher_a, a_grp, a_keys), (searcher_b, b_grp, b_keys)):
            if not grp:
                continue
            actions = searcher.search_batch(
                [g["i"] for g in grp], [g["state"] for g in grp],
                seed=base_seed + round_idx, plies=[g["plies"] for g in grp],
            )
            for g, act in zip(grp, actions):
                engine.apply_action(g["state"], engine.PlacementAction(unpack_coord_id(act)))
                g["plies"] += 1
                keyset.add(g["i"])
        round_idx += 1

    return HeadToHeadResult(
        games=games, completed=completed, wins=wins, losses=losses, draws=draws,
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
