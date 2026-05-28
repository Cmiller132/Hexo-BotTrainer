"""Sequential self-play sample generation for the dense CNN model."""

from __future__ import annotations

from random import Random
from time import perf_counter
from typing import Any, Mapping

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_runner.records import AbortRecord, HexoRecordFile, HexoRecordPlayer

from .debug_artifacts import render_preview_game_actions
from .inference import DenseCNNInference
from .mcts import SearchResult, new_mcts_evaluation_cache, run_batched_mcts
from .performance import _extend_mcts_diagnostic_batches, _summarize_mcts_diagnostic_batches
from .samples import Model1SampleData, finalize_game_samples, sample_from_state


def generate_selfplay_epoch(*, ctx: Any, components: Any, epoch: int, games_per_epoch: int) -> dict[str, Any]:
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
    target_samples = int(config.selfplay.samples_per_epoch)
    max_games = max(int(games_per_epoch or 0), 1)
    if target_samples > 0:
        max_games = max(max_games, target_samples)

    record_dir = ctx.output_dir / "selfplay"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / f"epoch_{epoch:06d}.hxr"
    debug_games: list[dict[str, Any]] = []
    samples_added = 0
    searched_positions = 0
    mcts_simulations = 0
    completed_games = 0
    truncated_games = 0
    mcts_search_elapsed = 0.0
    mcts_diagnostic_batches: list[Mapping[str, Any]] = []
    started = perf_counter()

    players = (
        HexoRecordPlayer("dense-cnn-a", "player0", "Dense CNN A"),
        HexoRecordPlayer("dense-cnn-b", "player1", "Dense CNN B"),
    )
    active_limit = max(1, int(getattr(trainer, "selfplay_batch_size", config.selfplay.active_games)))
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
    evaluation_cache = new_mcts_evaluation_cache(
        max_states=config.selfplay.mcts_evaluation_cache_max_states
    )
    active_root_limit = max(
        1,
        int(getattr(trainer, "mcts_active_root_limit", config.selfplay.mcts_active_root_limit)),
    )
    next_game_index = 0
    active: list[dict[str, Any]] = []
    with HexoRecordFile.create(record_path, engine.engine_metadata(), players) as record_file:
        while (samples_added < target_samples or active) and (next_game_index < max_games or active):
            while (
                len(active) < active_limit
                and next_game_index < max_games
                and len(active) < max(0, target_samples - samples_added - sum(len(game["pending"]) for game in active))
            ):
                game_id = f"epoch-{epoch:06d}-selfplay-{next_game_index:06d}"
                seed = (ctx.config.run.seed or 0) + epoch * 1_000_000 + next_game_index
                active.append(
                    {
                        "game_id": game_id,
                        "seed": seed,
                        "state": engine.new_game(seed=seed),
                        "pending": [],
                        "actions": [],
                    }
                )
                next_game_index += 1

            playable = [
                game
                for game in active
                if engine.terminal(game["state"]) is None
                and len(game["actions"]) < config.selfplay.max_actions
            ]
            if playable:
                pending_count = sum(len(item["pending"]) for item in active)
                remaining_samples = max(0, target_samples - samples_added - pending_count)
                search_games = playable[:remaining_samples]
                rollout_games = playable[remaining_samples:]
                if search_games:
                    search_started = perf_counter()
                    searches = _search_playable_games(
                        search_games,
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
                        evaluation_cache=evaluation_cache,
                        active_root_limit=active_root_limit,
                    )
                    if searches:
                        _extend_mcts_diagnostic_batches(mcts_diagnostic_batches, searches)
                    mcts_search_elapsed += perf_counter() - search_started
                else:
                    searches = []
                for game, search in zip(search_games, searches):
                    configured_visits = int(getattr(trainer, "search_visits", config.selfplay.search_visits))
                    if int(search.visits) != configured_visits:
                        raise RuntimeError(
                            f"dense_cnn MCTS returned {search.visits} visits; expected exactly {configured_visits}"
                        )
                    searched_positions += 1
                    mcts_simulations += int(search.visits)
                    state = game["state"]
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
                    game["pending"].append((sample.current_player, sample, search.root_value))
                    action = engine.PlacementAction(unpack_coord_id(search.action_id))
                    engine.apply_action(state, action)
                    game["actions"].append(search.action_id)
                if rollout_games:
                    rollout_actions = _policy_rollout_actions(
                        rollout_games,
                        inference=inference,
                        temperature=config.selfplay.temperature,
                        seed=(ctx.config.run.seed or 0) + epoch + searched_positions,
                    )
                    for game, action_id in zip(rollout_games, rollout_actions):
                        action = engine.PlacementAction(unpack_coord_id(action_id))
                        engine.apply_action(game["state"], action)
                        game["actions"].append(action_id)

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
                if len(debug_games) < config.debug.preview_games:
                    debug_games.append(
                        {
                            "game_id": game["game_id"],
                            "winner": winner,
                            "truncated": truncated,
                            "actions": game["actions"],
                            "samples": len(finalized),
                        }
                    )
                active.remove(game)

    elapsed = perf_counter() - started
    search_positions_per_second = searched_positions / max(mcts_search_elapsed, 1.0e-9)
    end_to_end_positions_per_second = searched_positions / max(elapsed, 1.0e-9)
    mcts_diagnostics = _summarize_mcts_diagnostic_batches(mcts_diagnostic_batches)
    preview_artifacts: list[dict[str, Any]] = []
    if config.debug.write_sample_previews and debug_games:
        preview_dir = ctx.output_dir / "diagnostics" / "dense_cnn_previews" / f"epoch_{epoch:06d}"
        for game in debug_games:
            preview_artifacts.append(
                render_preview_game_actions(
                    game["actions"],
                    preview_dir,
                    game_id=game["game_id"],
                    file_prefix=game["game_id"],
                    max_actions=config.selfplay.max_actions,
                    max_images=4,
                    actions_per_image=64,
                )
            )
    game_history_path = None
    if config.debug.write_game_history:
        game_history_path = ctx.diagnostics.write_json(
            f"dense_cnn.game_history.epoch_{epoch:06d}.json",
            {
                "epoch": epoch,
                "record_path": str(record_path),
                "games": debug_games,
                "preview_artifacts": preview_artifacts,
            },
        )
    debug_path = ctx.diagnostics.write_json(
        f"dense_cnn.selfplay.epoch_{epoch:06d}.json",
        {
            "epoch": epoch,
            "record_path": str(record_path),
            "game_history_path": str(game_history_path) if game_history_path is not None else None,
            "preview_games": debug_games,
            "preview_artifacts": preview_artifacts,
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
            "completion_rollout": "model_policy_after_sample_budget",
            "active_games": active_limit,
            "mcts_virtual_batch_size": virtual_batch_size,
            "mcts_progressive_widening_initial_actions": progressive_widening_initial_actions,
            "mcts_progressive_widening_child_initial_actions": progressive_widening_child_initial_actions,
            "mcts_progressive_widening_candidate_actions": progressive_widening_candidate_actions,
            "mcts_progressive_widening_growth_interval": progressive_widening_growth_interval,
            "mcts_progressive_widening_growth_base": progressive_widening_growth_base,
            "mcts_evaluation_cache_max_states": config.selfplay.mcts_evaluation_cache_max_states,
            "mcts_active_root_limit": active_root_limit,
            "mcts_diagnostics": mcts_diagnostics,
        },
    )
    return {
        "status": "completed",
        "epoch": epoch,
        "games": completed_games,
        "truncated_games": truncated_games,
        "samples_added": samples_added,
        "searched_positions": searched_positions,
        "mcts_simulations": mcts_simulations,
        "buffer_count": buffer.sample_count,
        "record_path": str(record_path),
        "game_history_path": str(game_history_path) if game_history_path is not None else None,
        "debug_path": str(debug_path),
        "elapsed_seconds": elapsed,
        "mcts_search_elapsed_seconds": mcts_search_elapsed,
        "positions_per_second": end_to_end_positions_per_second,
        "search_positions_per_second": search_positions_per_second,
        "end_to_end_positions_per_second": end_to_end_positions_per_second,
        "samples_per_second": samples_added / max(elapsed, 1.0e-9),
        "completion_rollout": "model_policy_after_sample_budget",
        "mcts_diagnostics": mcts_diagnostics,
    }


def _finalize_game_samples(
    pending: list[tuple[str, Model1SampleData, float]],
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
    evaluation_cache: object | None = None,
    active_root_limit: int | None = None,
) -> list[SearchResult]:
    return run_batched_mcts(
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
        evaluation_cache=evaluation_cache,
        active_root_limit=active_root_limit,
    )


def _sample_policy_action(policy: Any, *, temperature: float, seed: int) -> int:
    items = [(int(action_id), max(0.0, float(weight))) for action_id, weight in policy.items()]
    if not items:
        raise RuntimeError("cannot sample from an empty visit policy")
    if temperature <= 1.0e-6:
        return max(items, key=lambda item: (item[1], -item[0]))[0]
    inv_temperature = 1.0 / max(temperature, 1.0e-3)
    weights = [(weight or 1.0e-12) ** inv_temperature for _action, weight in items]
    total = sum(weights)
    threshold = Random(seed).random() * total
    for (action_id, _weight), weight in zip(items, weights):
        threshold -= weight
        if threshold <= 0:
            return action_id
    return items[-1][0]


def _policy_rollout_actions(
    playable: list[dict[str, Any]],
    *,
    inference: DenseCNNInference,
    temperature: float,
    seed: int,
) -> list[int]:
    evaluations = inference.infer_states([game["state"] for game in playable])
    return [
        _sample_policy_action(
            evaluation.legal_priors,
            temperature=temperature,
            seed=seed + index * 1_000_003,
        )
        for index, evaluation in enumerate(evaluations)
    ]
