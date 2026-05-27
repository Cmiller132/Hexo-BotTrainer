"""Sequential AlphaZero-style self-play for Hexformer AR."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from random import Random
from time import perf_counter
from typing import Any

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_runner.records import AbortRecord, HexoRecordFile, HexoRecordPlayer
from hexo_utils.samples import append_samples

from .input import build_selfplay_sample_payloads
from .inference import HexformerInference
from .mcts import SearchResult, run_mcts
from .samples import SAMPLE_NAMESPACE, HexformerSample, training_record_from_sample


@dataclass(frozen=True, slots=True)
class PendingDecision:
    state: object
    player: str
    turn_index: int
    search: SearchResult


def generate_selfplay_epoch(*, ctx: Any, components: Any, epoch: int, games_per_epoch: int) -> dict[str, Any]:
    trainer = components.model.trainer
    config = trainer.config
    curriculum_seeded = _seed_curriculum_before_selfplay(ctx=ctx, components=components, epoch=epoch, trainer=trainer)
    ema_scope = trainer.ema_weights() if hasattr(trainer, "ema_weights") else nullcontext()
    target_samples = int(config.selfplay.samples_per_epoch)
    max_games = max(int(games_per_epoch or config.selfplay.games_per_epoch), 1)
    if target_samples > 0:
        max_games = max(max_games, target_samples)
    rng = Random((ctx.config.run.seed or 0) + epoch)

    record_dir = ctx.output_dir / "selfplay"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / f"hexformer_ar_epoch_{epoch:06d}.hxr"
    players = (
        HexoRecordPlayer("hexformer-a", "player0", "Hexformer A"),
        HexoRecordPlayer("hexformer-b", "player1", "Hexformer B"),
    )
    started = perf_counter()

    with ema_scope:
        inference = HexformerInference(
            components.model.model,
            config=config,
            device=trainer.device,
            amp=config.training.amp,
        )
        result = _generate_with_inference(
            ctx=ctx,
            components=components,
            epoch=epoch,
            max_games=max_games,
            target_samples=target_samples,
            rng=rng,
            record_path=record_path,
            players=players,
            inference=inference,
            started=started,
            curriculum_seeded=curriculum_seeded,
        )
    return result


def _generate_with_inference(
    *,
    ctx: Any,
    components: Any,
    epoch: int,
    max_games: int,
    target_samples: int,
    rng: Random,
    record_path: Any,
    players: Any,
    inference: HexformerInference,
    started: float,
    curriculum_seeded: bool,
) -> dict[str, Any]:
    trainer = components.model.trainer
    config = trainer.config
    buffer = trainer.buffer
    samples_added = 0
    searched_positions = 0
    mcts_simulations = 0
    completed_games = 0
    truncated_games = 0
    debug_games: list[dict[str, Any]] = []

    with HexoRecordFile.create(record_path, engine.engine_metadata(), players) as record_file:
        for game_index in range(max_games):
            if samples_added >= target_samples:
                break
            game_id = f"epoch-{epoch:06d}-hexformer-{game_index:06d}"
            seed = (ctx.config.run.seed or 0) + epoch * 1_000_000 + game_index
            state = engine.new_game(seed=seed)
            pending: list[PendingDecision] = []
            actions: list[int] = []
            while engine.terminal(state) is None and len(actions) < config.selfplay.max_actions and samples_added + len(pending) < target_samples:
                visits = _visits_for_position(config, rng)
                search = run_mcts(
                    state,
                    inference,
                    visits=visits,
                    c_puct=config.selfplay.c_puct,
                    temperature=config.selfplay.temperature,
                    seed=seed + len(actions),
                )
                pending.append(
                    PendingDecision(
                        state=engine.clone_state(state),
                        player=_player_label(engine.current_player(state)),
                        turn_index=len(actions),
                        search=search,
                    )
                )
                engine.apply_action(state, engine.PlacementAction(unpack_coord_id(search.action_id)))
                actions.append(search.action_id)
                searched_positions += 1
                mcts_simulations += search.visits

            terminal = engine.terminal(state)
            truncated = terminal is None
            winner = _player_label(terminal.winner) if terminal is not None and terminal.winner is not None else None
            writer = record_file.begin_game(game_id, seed=seed)
            for action_id in actions:
                writer.record_action(engine.PlacementAction(unpack_coord_id(action_id)))
            if truncated:
                writer.finish_aborted(
                    AbortRecord(
                        stage="selfplay",
                        exception_type="MaxActionsReached",
                        message=f"hexformer_ar self-play reached max_actions={config.selfplay.max_actions}",
                    )
                )
                truncated_games += 1
            else:
                writer.finish_completed(winner, len(actions))
                completed_games += 1

            finalized = _finalize_pending(game_id, pending, winner, config)
            buffer.extend(finalized)
            sample_store = getattr(components.shared, "sample_store", None)
            if sample_store is not None and finalized:
                append_samples(
                    sample_store,
                    tuple(training_record_from_sample(sample) for sample in finalized),
                    metadata={
                        "epoch": epoch,
                        "game_id": game_id,
                        "model_family": "hexformer_ar",
                        "compression": config.samples.compression,
                        "extensions": {SAMPLE_NAMESPACE: 1},
                    },
                )
            samples_added += len(finalized)
            if len(debug_games) < config.debug.preview_samples:
                debug_games.append(
                    {
                        "game_id": game_id,
                        "winner": winner,
                        "truncated": truncated,
                        "actions": actions,
                        "samples": len(finalized),
                    }
                )

    elapsed = perf_counter() - started
    debug_path = ctx.diagnostics.write_json(
        f"hexformer_ar.selfplay.epoch_{epoch:06d}.json",
        {
            "epoch": epoch,
            "record_path": str(record_path),
            "preview_games": debug_games,
            "samples_added": samples_added,
            "searched_positions": searched_positions,
            "mcts_simulations": mcts_simulations,
            "search_visits": int(config.selfplay.search_visits),
            "playout_cap_randomization": bool(config.selfplay.playout_cap_randomization),
            "curriculum_seeded": curriculum_seeded,
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
        "debug_path": str(debug_path),
        "elapsed_seconds": elapsed,
        "positions_per_second": searched_positions / max(elapsed, 1.0e-9),
        "curriculum_seeded": curriculum_seeded,
    }


def _seed_curriculum_before_selfplay(*, ctx: Any, components: Any, epoch: int, trainer: Any) -> bool:
    store = getattr(components.shared, "sample_store", None)
    if store is None or not hasattr(trainer, "seed_curriculum_if_needed"):
        return False
    seeded = bool(
        trainer.seed_curriculum_if_needed(
            store=store,
            seed=int(ctx.config.run.seed or 0) + int(epoch),
            epoch=epoch,
        )
    )
    return seeded


def _finalize_pending(game_id: str, pending: list[PendingDecision], winner: str | None, config: Any) -> list[Any]:
    rows = build_selfplay_sample_payloads(
        game_id=game_id,
        states=tuple(decision.state for decision in pending),
        players=tuple(decision.player for decision in pending),
        turn_indices=tuple(decision.turn_index for decision in pending),
        visit_policies=tuple(decision.search.visit_policy for decision in pending),
        root_values=tuple(decision.search.root_value for decision in pending),
        search_visits=tuple(decision.search.visits for decision in pending),
        selected_action_ids=tuple(decision.search.action_id for decision in pending),
        winner=winner,
        architecture=config.architecture,
        candidates=config.candidates,
    )
    return [
        HexformerSample(
            game_id=str(row["game_id"]),
            turn_index=int(row["turn_index"]),
            input_payload=dict(row["input_payload"]),
            metadata=dict(row.get("metadata", {})),
        )
        for row in rows
    ]


def _visits_for_position(config: Any, rng: Random) -> int:
    if config.selfplay.playout_cap_randomization and rng.random() < config.selfplay.low_visit_probability:
        return max(1, int(config.selfplay.low_visit_count))
    return max(1, int(config.selfplay.search_visits))


def _player_label(value: object) -> str:
    return str(getattr(value, "value", value))
