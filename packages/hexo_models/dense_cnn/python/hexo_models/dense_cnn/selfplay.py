"""Sequential self-play sample generation for dense CNN Model 1.

The production self-play loop works from live `hexo_engine.HexoState` objects.
It batches active games through one persistent Rust MCTS session, records compact
pre-decision samples from the same live states, applies selected actions through
the generic engine, writes `.hxr` records, and finalizes targets once each game
has an outcome.

The loop is game-driven: every active nonterminal game decision is searched with
MCTS until the game reaches a terminal outcome or `max_actions`.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_runner.records import AbortRecord, HexoRecordFile, HexoRecordPlayer

from .inference import DenseCNNInference
from .mcts import BatchedMctsSession, SearchResult, new_mcts_session
from .performance import _extend_mcts_diagnostic_batches, _summarize_mcts_diagnostic_batches
from .samples import CompressedSample, Model1SampleData, finalize_game_samples, sample_from_state

def generate_selfplay_epoch(*, ctx: Any, components: Any, epoch: int, games_per_epoch: int) -> dict[str, Any]:
    """Generate one epoch of dense_cnn self-play samples.

    The function is called by `hexo_train`. It mutates only model-owned
    components: the replay buffer and dense-cnn diagnostics. Game truth remains
    in `hexo_engine`, and MCTS search state remains in the native dense-cnn
    session.
    """

    trainer = components.model.trainer
    config = trainer.config
    buffer = trainer.buffer
    inference = DenseCNNInference(
        components.model.model,
        device=trainer.device,
        amp=config.training.amp,
        return_logits=False,
        max_batch_size=getattr(trainer, "inference_batch_size", 1024),
    )
    requested_games = int(games_per_epoch or 0)
    if requested_games < 0:
        raise ValueError("games_per_epoch must be >= 0")

    record_dir = ctx.output_dir / "selfplay"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / f"epoch_{epoch:06d}.hxr"
    samples_added = 0
    searched_positions = 0
    mcts_simulations = 0
    games_started = 0
    completed_games = 0
    truncated_games = 0
    mcts_search_elapsed = 0.0
    mcts_diagnostic_batches: list[Mapping[str, Any]] = []
    started = perf_counter()

    players = (
        HexoRecordPlayer("dense-cnn-a", "player0", "Dense CNN A"),
        HexoRecordPlayer("dense-cnn-b", "player1", "Dense CNN B"),
    )
    configured_active_limit = int(getattr(trainer, "selfplay_batch_size", config.selfplay.active_games))
    if configured_active_limit <= 0:
        raise ValueError("selfplay active game count must be > 0")
    virtual_batch_size = getattr(trainer, "mcts_virtual_batch_size", None)
    progressive_widening_initial_actions = int(
        getattr(
            trainer,
            "mcts_progressive_widening_initial_actions",
            config.selfplay.progressive_widening_initial_actions,
        )
    )
    progressive_widening_child_initial_actions = int(
        getattr(
            trainer,
            "mcts_progressive_widening_child_initial_actions",
            config.selfplay.progressive_widening_child_initial_actions,
        )
    )
    progressive_widening_candidate_actions = int(
        getattr(
            trainer,
            "mcts_progressive_widening_candidate_actions",
            config.selfplay.progressive_widening_candidate_actions,
        )
    )
    progressive_widening_growth_interval = float(
        getattr(
            trainer,
            "mcts_progressive_widening_growth_interval",
            config.selfplay.progressive_widening_growth_interval,
        )
    )
    progressive_widening_growth_base = float(
        getattr(
            trainer,
            "mcts_progressive_widening_growth_base",
            config.selfplay.progressive_widening_growth_base,
        )
    )
    root_noise_enabled = bool(
        getattr(
            trainer,
            "mcts_root_dirichlet_noise_enabled",
            config.selfplay.root_dirichlet_noise_enabled,
        )
    )
    root_dirichlet_alpha = float(
        getattr(trainer, "mcts_root_dirichlet_alpha", config.selfplay.root_dirichlet_alpha)
    )
    root_dirichlet_noise_fraction = float(
        getattr(
            trainer,
            "mcts_root_dirichlet_noise_fraction",
            config.selfplay.root_dirichlet_noise_fraction,
        )
    )
    hidden_prior_mass = float(
        getattr(trainer, "mcts_hidden_prior_mass", config.selfplay.hidden_prior_mass)
    )
    fpu_reduction = float(getattr(trainer, "mcts_fpu_reduction", config.selfplay.fpu_reduction))
    virtual_loss = float(getattr(trainer, "mcts_virtual_loss", config.selfplay.virtual_loss))
    mcts_session = new_mcts_session(max_states=config.selfplay.mcts_session_cache_max_states)
    active_root_limit = int(getattr(trainer, "mcts_active_root_limit", config.selfplay.mcts_active_root_limit))
    if active_root_limit <= 0:
        raise ValueError("mcts_active_root_limit must be > 0")
    if configured_active_limit > active_root_limit:
        raise ValueError("selfplay active game count must be <= mcts_active_root_limit")
    active_limit = configured_active_limit
    next_game_index = 0
    active: list[dict[str, Any]] = []
    with HexoRecordFile.create(record_path, engine.engine_metadata(), players) as record_file:
        while next_game_index < requested_games or active:
            while len(active) < active_limit and next_game_index < requested_games:
                game_id = f"epoch-{epoch:06d}-selfplay-{next_game_index:06d}"
                seed = (ctx.config.run.seed or 0) + epoch * 1_000_000 + next_game_index
                active.append(
                    {
                        "game_id": game_id,
                        "search_key": next_game_index,
                        "seed": seed,
                        "state": engine.new_game(seed=seed),
                        "pending": [],
                        "actions": [],
                    }
                )
                next_game_index += 1
                games_started += 1

            playable = [
                game
                for game in active
                if engine.terminal(game["state"]) is None
                and len(game["actions"]) < config.selfplay.max_actions
            ]
            if playable:
                # Search live roots through the persistent native session. The
                # session keys are per-game and let Rust promote the selected
                # child subtree after each turn.
                search_started = perf_counter()
                searches = _search_playable_games(
                    playable,
                    inference=inference,
                    visits=getattr(trainer, "search_visits", config.selfplay.search_visits),
                    temperature=config.selfplay.temperature,
                    seed=(ctx.config.run.seed or 0) + epoch,
                    virtual_batch_size=virtual_batch_size,
                    progressive_widening_initial_actions=progressive_widening_initial_actions,
                    progressive_widening_child_initial_actions=progressive_widening_child_initial_actions,
                    progressive_widening_candidate_actions=progressive_widening_candidate_actions,
                    progressive_widening_growth_interval=progressive_widening_growth_interval,
                    progressive_widening_growth_base=progressive_widening_growth_base,
                    mcts_session=mcts_session,
                    root_dirichlet_alpha=root_dirichlet_alpha if root_noise_enabled else None,
                    root_dirichlet_noise_fraction=(
                        root_dirichlet_noise_fraction if root_noise_enabled else None
                    ),
                    hidden_prior_mass=hidden_prior_mass,
                    fpu_reduction=fpu_reduction,
                    virtual_loss=virtual_loss,
                    active_root_limit=active_root_limit,
                )
                if len(searches) != len(playable):
                    raise RuntimeError(
                        f"dense_cnn MCTS returned {len(searches)} results for {len(playable)} playable games"
                    )
                if searches:
                    _extend_mcts_diagnostic_batches(mcts_diagnostic_batches, searches)
                mcts_search_elapsed += perf_counter() - search_started
                for game, search in zip(playable, searches):
                    configured_visits = int(getattr(trainer, "search_visits", config.selfplay.search_visits))
                    if int(search.visits) != configured_visits:
                        raise RuntimeError(
                            f"dense_cnn MCTS returned {search.visits} visits; expected exactly {configured_visits}"
                        )
                    searched_positions += 1
                    mcts_simulations += int(search.visits)
                    state = game["state"]
                    # The sample is captured before applying the chosen action:
                    # policy/value targets describe the decision position, while
                    # final value/lookahead targets are filled once the game ends.
                    sample = sample_from_state(
                        state,
                        game_id=game["game_id"],
                        turn_index=len(game["actions"]),
                        policy=search.visit_policy,
                        value=search.root_value,
                        metadata={
                            "epoch": epoch,
                            "search_visits": search.visits,
                            "configured_search_visits": configured_visits,
                            "mcts_sims_exact": True,
                            "sample_source": "mcts",
                        },
                    )
                    compressed_sample = CompressedSample.from_data(
                        sample,
                        compression_level=config.samples.compression_level,
                    )
                    game["pending"].append((sample.current_player, compressed_sample, search.root_value))
                    action = engine.PlacementAction(unpack_coord_id(search.action_id))
                    engine.apply_action(state, action)
                    game["actions"].append(search.action_id)

            finished = [
                game
                for game in active
                if engine.terminal(game["state"]) is not None
                or len(game["actions"]) >= config.selfplay.max_actions
            ]
            for game in finished:
                terminal = engine.terminal(game["state"])
                truncated = terminal is None
                winner = _player_label(terminal.winner) if terminal is not None and terminal.winner is not None else None
                writer = record_file.begin_game(game["game_id"], seed=game["seed"])
                for action_id in game["actions"]:
                    writer.record_action(engine.PlacementAction(unpack_coord_id(action_id)))
                if truncated:
                    writer.finish_aborted(
                        AbortRecord(
                            stage="selfplay",
                            exception_type="MaxActionsReached",
                            message=f"dense_cnn self-play reached max_actions={config.selfplay.max_actions}",
                        )
                    )
                    truncated_games += 1
                else:
                    writer.finish_completed(winner, len(game["actions"]))
                    completed_games += 1

                finalized = _finalize_game_samples(
                    game["pending"],
                    winner,
                    config.architecture.lookahead_horizons,
                    truncated=truncated,
                )
                buffer.extend(finalized)
                samples_added += len(finalized)
                active.remove(game)
                # A finished game cannot legally reuse its old native root.
                mcts_session.discard(int(game["search_key"]))

    elapsed = perf_counter() - started
    search_positions_per_second = searched_positions / max(mcts_search_elapsed, 1.0e-9)
    end_to_end_positions_per_second = searched_positions / max(elapsed, 1.0e-9)
    mcts_diagnostics = _summarize_mcts_diagnostic_batches(mcts_diagnostic_batches)
    debug_path = ctx.diagnostics.write_json(
        f"dense_cnn.selfplay.epoch_{epoch:06d}.json",
        {
            "epoch": epoch,
            "record_path": str(record_path),
            "selfplay_mode": "game_driven_all_mcts",
            "requested_games": requested_games,
            "games_started": games_started,
            "games_finished": completed_games + truncated_games,
            "completed_games": completed_games,
            "truncated_games": truncated_games,
            "samples_added": samples_added,
            "searched_positions": searched_positions,
            "mcts_simulations": mcts_simulations,
            "mcts_search_elapsed_seconds": mcts_search_elapsed,
            "search_positions_per_second": search_positions_per_second,
            "end_to_end_positions_per_second": end_to_end_positions_per_second,
            "search_visits": int(getattr(trainer, "search_visits", config.selfplay.search_visits)),
            "mcts_sims_per_searched_position": (
                mcts_simulations / searched_positions if searched_positions else 0.0
            ),
            "no_policy_rollout_tails": True,
            "configured_active_games": configured_active_limit,
            "active_games": active_limit,
            "mcts_virtual_batch_size": virtual_batch_size,
            "mcts_tree_reuse_session": True,
            "mcts_progressive_widening_initial_actions": progressive_widening_initial_actions,
            "mcts_progressive_widening_child_initial_actions": progressive_widening_child_initial_actions,
            "mcts_progressive_widening_candidate_actions": progressive_widening_candidate_actions,
            "mcts_progressive_widening_growth_interval": progressive_widening_growth_interval,
            "mcts_progressive_widening_growth_base": progressive_widening_growth_base,
            "mcts_root_dirichlet_noise_enabled": root_noise_enabled,
            "mcts_root_dirichlet_alpha": root_dirichlet_alpha if root_noise_enabled else None,
            "mcts_root_dirichlet_noise_fraction": (
                root_dirichlet_noise_fraction if root_noise_enabled else None
            ),
            "mcts_hidden_prior_mass": hidden_prior_mass,
            "mcts_fpu_reduction": fpu_reduction,
            "mcts_virtual_loss": virtual_loss,
            "mcts_session_cache_max_states": config.selfplay.mcts_session_cache_max_states,
            "mcts_active_root_limit": active_root_limit,
            "mcts_diagnostics": mcts_diagnostics,
        },
    )
    return {
        "status": "completed",
        "epoch": epoch,
        "selfplay_mode": "game_driven_all_mcts",
        "requested_games": requested_games,
        "games_started": games_started,
        "games_finished": completed_games + truncated_games,
        "completed_games": completed_games,
        "games": completed_games,
        "truncated_games": truncated_games,
        "samples_added": samples_added,
        "searched_positions": searched_positions,
        "mcts_simulations": mcts_simulations,
        "buffer_count": buffer.sample_count,
        "record_path": str(record_path),
        "debug_path": str(debug_path),
        "elapsed_seconds": elapsed,
        "mcts_search_elapsed_seconds": mcts_search_elapsed,
        "positions_per_second": end_to_end_positions_per_second,
        "search_positions_per_second": search_positions_per_second,
        "end_to_end_positions_per_second": end_to_end_positions_per_second,
        "samples_per_second": samples_added / max(elapsed, 1.0e-9),
        "no_policy_rollout_tails": True,
        "configured_active_games": configured_active_limit,
        "active_games": active_limit,
        "mcts_tree_reuse_session": True,
        "mcts_diagnostics": mcts_diagnostics,
    }


def _finalize_game_samples(
    pending: list[tuple[str, Model1SampleData | CompressedSample, float]],
    winner: str | None,
    horizons: tuple[int, ...],
    *,
    truncated: bool = False,
) -> list[Model1SampleData]:
    return finalize_game_samples(pending, winner, horizons, truncated=truncated)


def _player_label(value: object) -> str:
    return str(getattr(value, "value", value))


def _search_playable_games(
    playable: list[dict[str, Any]],
    *,
    inference: DenseCNNInference,
    visits: int,
    temperature: float,
    seed: int,
    virtual_batch_size: int | None = None,
    progressive_widening_initial_actions: int | None = None,
    progressive_widening_child_initial_actions: int | None = None,
    progressive_widening_candidate_actions: int | None = None,
    progressive_widening_growth_interval: float | None = None,
    progressive_widening_growth_base: float | None = None,
    mcts_session: BatchedMctsSession | None = None,
    root_dirichlet_alpha: float | None = None,
    root_dirichlet_noise_fraction: float | None = None,
    hidden_prior_mass: float | None = None,
    fpu_reduction: float | None = None,
    virtual_loss: float | None = None,
    active_root_limit: int | None = None,
) -> list[SearchResult]:
    """Search currently playable live states with the required native session."""

    if mcts_session is None:
        raise RuntimeError("dense_cnn self-play requires a native MCTS session")
    return mcts_session.run(
        [int(game["search_key"]) for game in playable],
        [game["state"] for game in playable],
        inference,
        visits=visits,
        temperature=temperature,
        seed=seed,
        virtual_batch_size=virtual_batch_size,
        progressive_widening_initial_actions=progressive_widening_initial_actions,
        progressive_widening_child_initial_actions=progressive_widening_child_initial_actions,
        progressive_widening_candidate_actions=progressive_widening_candidate_actions,
        progressive_widening_growth_interval=progressive_widening_growth_interval,
        progressive_widening_growth_base=progressive_widening_growth_base,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_dirichlet_noise_fraction=root_dirichlet_noise_fraction,
        hidden_prior_mass=hidden_prior_mass,
        fpu_reduction=fpu_reduction,
        virtual_loss=virtual_loss,
        active_root_limit=active_root_limit,
    )
