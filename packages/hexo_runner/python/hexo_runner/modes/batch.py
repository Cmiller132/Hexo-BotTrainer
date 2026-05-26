"""Local multiprocessing batch runner mode."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Sequence

from ..engine import HexoEngineAdapter
from ..loop import run_match_loop
from ..player import PlayerFactory, RunnerPlayer, WorkerContext
from ..records import AbortRecord, BatchResult, GameResult, GameStatus, JsonlRecordSink
from ..session import BatchSpec, GameSpec
from ..timing import Timer


def run_batch(spec: BatchSpec) -> BatchResult:
    """Run many games on this machine with local worker processes."""

    timer = Timer.start()
    games = tuple(spec.games)
    total_games = len(games)
    if total_games == 0:
        return BatchResult(
            batch_id=spec.batch_id,
            total_games=0,
            completed=0,
            aborted=0,
            worker_count=0,
            duration_ms=0.0,
            metadata=spec.metadata,
        )

    worker_count = spec.worker_count or min(28, total_games)
    worker_count = max(1, min(worker_count, total_games))
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assigned_games = _assign_games(games, worker_count, max(1, spec.chunk_size))
    tasks = [
        _WorkerTask(
            batch_id=spec.batch_id,
            worker_id=index,
            games=worker_games,
            factories=spec.player_factories,
            output_dir=str(output_dir),
            metadata=dict(spec.metadata),
        )
        for index, worker_games in enumerate(assigned_games)
        if worker_games
    ]

    if worker_count == 1:
        worker_results = [_run_worker_task(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=worker_count) as pool:
            worker_results = pool.map(_run_worker_task, tasks)

    results = tuple(result for worker in worker_results for result in worker.results)
    completed = sum(1 for result in results if result.status == GameStatus.COMPLETED)
    aborted = sum(1 for result in results if result.status == GameStatus.ABORTED)
    record_refs = tuple(worker.record_ref for worker in worker_results)
    aborts = tuple(result.abort for result in results if result.abort is not None)[:10]

    return BatchResult(
        batch_id=spec.batch_id,
        total_games=total_games,
        completed=completed,
        aborted=aborted,
        worker_count=worker_count,
        duration_ms=timer.elapsed_ms(),
        record_refs=record_refs,
        aborts=aborts,
        results=results,
        metadata=spec.metadata,
    )


class _WorkerTask:
    def __init__(
        self,
        *,
        batch_id: str,
        worker_id: int,
        games: Sequence[GameSpec],
        factories: tuple[PlayerFactory, PlayerFactory],
        output_dir: str,
        metadata: dict[str, object],
    ) -> None:
        self.batch_id = batch_id
        self.worker_id = worker_id
        self.games = tuple(games)
        self.factories = factories
        self.output_dir = output_dir
        self.metadata = metadata


class _WorkerResult:
    def __init__(self, *, record_ref: object, results: Sequence[GameResult]) -> None:
        self.record_ref = record_ref
        self.results = tuple(results)


def _run_worker_task(task: _WorkerTask) -> _WorkerResult:
    adapter = HexoEngineAdapter()
    players = tuple(_create_player(factory) for factory in task.factories)
    worker_context = WorkerContext(
        worker_id=task.worker_id,
        engine_metadata=adapter.metadata(),
        metadata=task.metadata,
    )
    for player in players:
        player.setup_worker(worker_context)

    path = Path(task.output_dir) / f"{task.batch_id}-worker-{task.worker_id}.jsonl"
    sink = JsonlRecordSink(path)
    results: list[GameResult] = []
    try:
        for game in task.games:
            results.append(
                run_match_loop(
                    game,
                    players,  # type: ignore[arg-type]
                    sink,
                    engine_adapter=adapter,
                    worker_context=worker_context,
                    setup_players=False,
                    close_players=False,
                )
            )
    finally:
        for player in players:
            try:
                player.close()
            except Exception:
                pass

    return _WorkerResult(record_ref={"path": str(path), "games": len(results)}, results=results)


def _create_player(factory: PlayerFactory) -> RunnerPlayer:
    return factory.create_player()


def _chunks(games: Sequence[GameSpec], size: int) -> list[tuple[GameSpec, ...]]:
    return [tuple(games[index : index + size]) for index in range(0, len(games), size)]


def _assign_games(
    games: Sequence[GameSpec],
    worker_count: int,
    chunk_size: int,
) -> list[tuple[GameSpec, ...]]:
    buckets: list[list[GameSpec]] = [[] for _ in range(worker_count)]
    for index, chunk in enumerate(_chunks(games, chunk_size)):
        buckets[index % worker_count].extend(chunk)
    return [tuple(bucket) for bucket in buckets]
