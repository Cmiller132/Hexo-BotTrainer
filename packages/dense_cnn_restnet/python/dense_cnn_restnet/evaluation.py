"""Epoch evaluation adapter for dense CNN checkpoints.

The generic trainer asks the model plugin to evaluate an epoch. This module
plays the configured number of games against SealBot and returns runner/
evaluation metadata. It does not generate training samples and does not mutate
the replay buffer.

Eval games are run **concurrently with cross-game leaf batching**, mirroring how
self-play drives many games through one persistent native MCTS session
(`selfplay.py`). All `games_per_epoch` games are kept in flight at once; on every
round, the positions where the dense player is to move are searched together in a
single `mcts_session.run([...keys], [...states])` call, so the network forward
batches leaves across every game instead of one game at a time. SealBot's moves
(a fixed 50ms minimax per turn) are independent per game and are played serially,
each game keeping its own isolated worker exactly as the per-game runner did.

This is purely an orchestration change: per-game search settings (visits, vbatch,
widening, opening temperature) are identical to the old serial path, and SealBot
remains the same opponent. Batched and serial search are not bit-identical (see
``rust/src/mcts.rs``), but they are strength-equivalent — the same property
self-play already relies on — so the win/loss/turn distribution is statistically
equivalent, just produced far faster (the old path issued ~one tiny forward per
game per simulation chunk, serially).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer
from hexo_runner.records import AbortRecord, HexoRecordFile, HexoRecordPlayer

from .inference import DenseCNNInference
from .mcts import new_mcts_session

_DENSE_PLAYER_ID = "dense-cnn-eval"
_DENSE_LABEL = "Dense CNN"
_SEALBOT_PLAYER_ID = "sealbot-best-50ms"


def _build_opponent(sealbot_config: SealBotConfig) -> SealBotPlayer:
    """Construct the per-game opponent. Tests monkeypatch this seam.

    One opponent is created per game so each keeps its own isolated SealBot
    worker (and minimax transposition state) across that game's turns, exactly
    as the previous per-game runner did — the bot a game faces is unaffected by
    the other games in flight.
    """

    return SealBotPlayer(sealbot_config, player_id=_SEALBOT_PLAYER_ID)


@dataclass(slots=True)
class _EvalGame:
    """One in-flight evaluation game and its bookkeeping."""

    game_id: str
    search_key: int
    seed: int
    dense_is_p0: bool
    state: Any
    opponent: Any
    actions: list[Any] = field(default_factory=list)  # PlacementAction, in order
    dense_decisions: int = 0
    done: bool = False
    status: str = "aborted"
    winner: str | None = None

    @property
    def dense_role(self) -> engine.Player:
        return engine.Player.PLAYER_0 if self.dense_is_p0 else engine.Player.PLAYER_1

    @property
    def dense_role_label(self) -> str:
        return "player0" if self.dense_is_p0 else "player1"

    def dense_to_move(self) -> bool:
        return engine.current_player(self.state) == self.dense_role


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

    outcome = _run_games_concurrent(
        ctx=ctx,
        trainer=trainer,
        components=components,
        eval_config=eval_config,
        sealbot_config=sealbot_config,
        output_dir=output_dir,
        epoch=epoch,
    )

    diagnostics = {
        "status": "completed",
        "epoch": epoch,
        "games": eval_config.games_per_epoch,
        "completed": outcome["completed"],
        "wins": outcome["wins"],
        "losses": outcome["losses"],
        "mean_turns": outcome["mean_turns"],
        "output_dir": str(output_dir),
        # Timing split (additive; the dashboard ignores unknown keys). Answers
        # "where does eval time go": batched dense search vs SealBot's fixed
        # think time vs everything else.
        "elapsed_seconds": outcome["elapsed_seconds"],
        "mcts_search_elapsed_seconds": outcome["mcts_search_elapsed_seconds"],
        "opponent_elapsed_seconds": outcome["opponent_elapsed_seconds"],
        "rounds": outcome["rounds"],
        "dense_forward_batches": outcome["dense_forward_batches"],
        "dense_decisions": outcome["dense_decisions"],
    }
    path = ctx.diagnostics.write_json(f"dense_cnn.evaluation.epoch_{epoch:06d}.json", diagnostics)
    return {**diagnostics, "diagnostics_path": str(path)}


def _run_games_concurrent(
    *,
    ctx: Any,
    trainer: Any,
    components: Any,
    eval_config: Any,
    sealbot_config: SealBotConfig,
    output_dir: Any,
    epoch: int,
) -> dict[str, Any]:
    """Play all eval games concurrently, batching dense leaves across games."""

    selfplay = trainer.config.selfplay
    base_seed = ctx.config.run.seed or 0
    games_per_epoch = int(eval_config.games_per_epoch)

    # Eval is a torch-only benchmark: TRT is self-play-only (a per-epoch engine
    # build is not worth it for eval, and torch FP16 is strength-equivalent — see
    # player.py). Bucketing + a max batch keep the now-large cross-game forwards
    # equivalence-safely chunked.
    inference = DenseCNNInference(
        components.model.model,
        device=trainer.device,
        amp=trainer.config.training.amp,
        return_logits=False,
        max_batch_size=trainer.inference_batch_size,
        use_trt=False,
        bucket_pad_multiple=(trainer.config.performance.inference_bucket_pad_multiple or None),
        fp16_model=trainer.config.performance.inference_fp16_model,
        fp16_allow_fallback=trainer.config.performance.inference_fp16_allow_fallback,
        use_torch_compile=trainer.config.performance.inference_use_torch_compile,
        compile_allow_fallback=trainer.config.performance.inference_compile_allow_torch_fallback,
        attention_kv_gather=trainer.config.performance.attention_kv_gather,
    )
    # Same vbatch policy as the old single-game eval player: an eval-only override
    # if set, else the calibrated self-play value. This is per-game leaf
    # parallelism and is unchanged by cross-game batching.
    eval_vbatch = eval_config.virtual_batch_size
    virtual_batch_size = eval_vbatch if eval_vbatch > 0 else trainer.mcts_virtual_batch_size

    mcts_session = new_mcts_session(max_states=selfplay.mcts_session_cache_max_states)

    games: list[_EvalGame] = []
    for game_index in range(games_per_epoch):
        # Alternate colors so the result is not tied to one fixed first-player role.
        dense_is_p0 = game_index % 2 == 0
        game_seed = (base_seed or 0) + epoch * 100_000 + game_index
        games.append(
            _EvalGame(
                game_id=f"eval-{epoch:06d}-{game_index:04d}",
                search_key=game_index,
                seed=game_seed,
                dense_is_p0=dense_is_p0,
                state=engine.new_game(seed=game_seed),
                opponent=_build_opponent(sealbot_config),
            )
        )

    mcts_search_elapsed = 0.0
    opponent_elapsed = 0.0
    rounds = 0
    dense_forward_batches = 0
    dense_decisions = 0
    # Deterministic per-run search seed BASE; the native session derives a distinct
    # per-root RNG (seed + root index) so games diverge within one round, and the
    # per-round increment below gives each game a FRESH sampling draw at every
    # opening move (a constant seed reused the same selection quantile for all
    # `opening_moves` decisions of a game, collapsing the opening diversification
    # this temperature exists for).
    search_seed = (base_seed or 0) + epoch

    def _finalize(game: _EvalGame) -> None:
        terminal = engine.terminal(game.state)
        if terminal is not None:
            game.status = "completed"
            game.winner = str(terminal.winner) if terminal.winner is not None else None
        else:
            # Reached max_actions without a terminal: an aborted/truncated game,
            # matching the runner's runner.max_actions abort.
            game.status = "aborted"
            game.winner = None
        game.done = True
        mcts_session.discard(game.search_key)

    def _settle(game: _EvalGame) -> bool:
        if engine.terminal(game.state) is not None or len(game.actions) >= eval_config.max_actions:
            _finalize(game)
            return True
        return False

    try:
        while True:
            active = [game for game in games if not game.done]
            if not active:
                break
            rounds += 1
            plies_this_round = 0

            # --- Batched dense ply across every game where dense is to move. ---
            dense_games = [game for game in active if game.dense_to_move()]
            if dense_games:
                # Per-game move temperature: sample the opening, then play greedily,
                # exactly as the old eval player did per game (opening_moves /
                # opening_temperature), just vectorized over the batch.
                move_temperatures = [
                    eval_config.opening_temperature
                    if (game.dense_decisions < eval_config.opening_moves and eval_config.opening_temperature > 0.0)
                    else 0.0
                    for game in dense_games
                ]
                started = perf_counter()
                searches = mcts_session.run(
                    [game.search_key for game in dense_games],
                    [game.state for game in dense_games],
                    inference,
                    visits=selfplay.search_visits,
                    c_puct=selfplay.c_puct,
                    temperature=0.0,
                    seed=search_seed + rounds * 1_000_003,
                    virtual_batch_size=virtual_batch_size,
                    active_root_limit=selfplay.mcts_active_root_limit,
                    root_policy_temperature=selfplay.root_policy_temperature,
                    fpu_reduction=selfplay.fpu_reduction,
                    virtual_loss=selfplay.virtual_loss,
                    widening_policy_mass=selfplay.widening_policy_mass,
                    widening_max_children=selfplay.widening_max_children,
                    widening_min_children=selfplay.widening_min_children,
                    move_temperatures=move_temperatures,
                )
                mcts_search_elapsed += perf_counter() - started
                dense_forward_batches += 1
                if len(searches) != len(dense_games):
                    raise RuntimeError(
                        f"dense_cnn eval MCTS returned {len(searches)} results for {len(dense_games)} games"
                    )
                for game, search in zip(dense_games, searches):
                    action = engine.PlacementAction(unpack_coord_id(search.action_id))
                    engine.apply_action(game.state, action)
                    game.actions.append(action)
                    game.dense_decisions += 1
                    dense_decisions += 1
                    plies_this_round += 1
                    _settle(game)

            # --- SealBot turns, serially per game, fully drained per turn. ---
            # SealBot's 50ms minimax is a fixed wall per turn and is independent
            # across games; it is not the eval bottleneck (the batched dense
            # search above is). Draining a full turn keeps each game's own worker
            # buffer self-contained.
            for game in active:
                if game.done:
                    continue
                while not game.done and not game.dense_to_move():
                    started = perf_counter()
                    decision = game.opponent.decide(game.state)
                    opponent_elapsed += perf_counter() - started
                    engine.apply_action(game.state, decision.action)
                    game.actions.append(decision.action)
                    plies_this_round += 1
                    _settle(game)

            if plies_this_round == 0:
                # Defensive: every active game is either done or dense-to-move
                # after the SealBot drain, so the next round must advance unless
                # all games are done. Bail rather than spin forever.
                raise RuntimeError("dense_cnn eval made no progress in a round; aborting to avoid a hang")
    finally:
        for game in games:
            try:
                game.opponent.close()
            except Exception:
                pass

    _write_records(games, output_dir, eval_config, max_actions=eval_config.max_actions)

    wins = sum(1 for game in games if game.winner == game.dense_role_label)
    losses = sum(1 for game in games if game.winner is not None and game.winner != game.dense_role_label)
    completed = sum(1 for game in games if game.status == "completed")
    turns = [len(game.actions) for game in games]

    return {
        "wins": wins,
        "losses": losses,
        "completed": completed,
        "mean_turns": sum(turns) / max(1, len(turns)),
        "elapsed_seconds": mcts_search_elapsed + opponent_elapsed,
        "mcts_search_elapsed_seconds": mcts_search_elapsed,
        "opponent_elapsed_seconds": opponent_elapsed,
        "rounds": rounds,
        "dense_forward_batches": dense_forward_batches,
        "dense_decisions": dense_decisions,
    }


def _write_records(games: list[_EvalGame], output_dir: Any, eval_config: Any, *, max_actions: int) -> None:
    """Write one `.hxr` per game, preserving the prior per-game eval layout."""

    variant_label = f"SealBot {eval_config.sealbot_variant}"
    metadata = engine.engine_metadata()
    for game in games:
        dense_player = HexoRecordPlayer(_DENSE_PLAYER_ID, game.dense_role_label, _DENSE_LABEL)
        sealbot_role = "player1" if game.dense_is_p0 else "player0"
        sealbot_player = HexoRecordPlayer(_SEALBOT_PLAYER_ID, sealbot_role, variant_label)
        record_players = (dense_player, sealbot_player) if game.dense_is_p0 else (sealbot_player, dense_player)

        record_path = output_dir / f"{game.game_id}.hxr"
        with HexoRecordFile.create(record_path, metadata, record_players) as record_file:
            writer = record_file.begin_game(game.game_id, seed=game.seed)
            for action in game.actions:
                writer.record_action(action)
            if game.status == "completed":
                writer.finish_completed(game.winner, len(game.actions))
            else:
                writer.finish_aborted(
                    AbortRecord(
                        stage="runner.max_actions",
                        exception_type="MaxActionsExceeded",
                        message=f"dense_cnn eval reached max_actions={max_actions} before terminal state.",
                    )
                )
