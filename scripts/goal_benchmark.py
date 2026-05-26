from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for PACKAGE_PATH in (
    ROOT / "packages" / "hexo_engine" / "python",
    ROOT / "packages" / "hexo_runner" / "python",
):
    if str(PACKAGE_PATH) not in sys.path:
        sys.path.insert(0, str(PACKAGE_PATH))

from hexo_engine import (  # noqa: E402
    AxialCoord,
    PlacementAction,
    apply_action,
    legal_action_count,
    legal_action_ids,
    new_game,
    terminal,
)
from hexo_engine.types import pack_coord_id, unpack_coord_id  # noqa: E402
from hexo_runner import DecisionResult, PlayerIdentity, WorkerContext  # noqa: E402
from hexo_runner.engine import HexoEngineAdapter  # noqa: E402
from hexo_runner.loop import run_match_loop  # noqa: E402
from hexo_runner.records import GameStatus, JsonlRecordSink  # noqa: E402
from hexo_runner.session import GameSpec  # noqa: E402


TARGET_MOVES_PER_SECOND = 100_000
TARGET_WORKER_RSS_BYTES = 512 * 1024 * 1024
EXPECTED_MOVES_PER_GAME = 11


@dataclass(frozen=True, slots=True)
class WorkerTask:
    worker_id: int
    measured_games: int
    warmup_games: int
    output_dir: str
    batch_id: str
    barrier_timeout_s: float


class GoalBenchmarkPlayer:
    """Small legal-move policy for runner throughput measurement.

    The player does not carry a game script. It enumerates legal actions from
    the cloned engine state on each decision, then extends the line it has
    observed for itself in the current game.
    """

    def __init__(self, player_id: str, row: int) -> None:
        self.identity = PlayerIdentity(player_id=player_id, label="goal-benchmark")
        self._row = row
        self._owned_q: set[int] = set()
        self.observations = 0

    def setup_worker(self, context: WorkerContext) -> None:
        self.observations = 0

    def start_game(self, context: object) -> None:
        self._owned_q.clear()

    def decide(self, state: object) -> DecisionResult:
        action_ids = legal_action_ids(state)
        if not action_ids:
            raise RuntimeError("no legal actions")

        next_q = 0 if not self._owned_q else max(self._owned_q) + 1
        for q in range(next_q, next_q + 6):
            coord = AxialCoord(q, self._row)
            if pack_coord_id(coord) in action_ids:
                return DecisionResult(PlacementAction(coord))

        return DecisionResult(PlacementAction(unpack_coord_id(action_ids[0])))

    def observe_transition(self, transition: object) -> None:
        self.observations += 1
        legal_action_count(transition.state)
        if transition.player_id == self.identity.player_id and isinstance(transition.action, PlacementAction):
            coord = transition.action.coord
            if coord.r == self._row:
                self._owned_q.add(coord.q)

    def finish_game(self, final_summary: object) -> None:
        pass

    def close(self) -> None:
        pass


def main() -> int:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    measured_games = args.measured_games or math.ceil(args.target_moves / EXPECTED_MOVES_PER_GAME)
    game_counts = _distribute(measured_games, args.workers)

    if args.output_dir is None:
        with tempfile.TemporaryDirectory(prefix="hexo_goal_benchmark_") as tmp:
            return _run(args, game_counts, Path(tmp))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return _run(args, game_counts, output_dir)


def _run(args: argparse.Namespace, game_counts: tuple[int, ...], output_dir: Path) -> int:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue[dict[str, Any]] = ctx.Queue()
    barrier = ctx.Barrier(args.workers)
    batch_id = f"goal-{int(time.time())}"
    tasks = tuple(
        WorkerTask(
            worker_id=index,
            measured_games=game_counts[index],
            warmup_games=args.warmup_games_per_worker,
            output_dir=str(output_dir),
            batch_id=batch_id,
            barrier_timeout_s=args.barrier_timeout_s,
        )
        for index in range(args.workers)
    )

    processes = [
        ctx.Process(target=_worker_entry, args=(task, barrier, result_queue), name=f"hexo-goal-{task.worker_id}")
        for task in tasks
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    worker_results = _collect_worker_results(result_queue, len(processes))
    failed_processes = [process for process in processes if process.exitcode != 0]
    if failed_processes:
        print(json.dumps({"status": "failed", "failed_workers": [p.name for p in failed_processes], "results": worker_results}, indent=2))
        return 1

    errors = [result for result in worker_results if "error" in result]
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return 1

    record_paths = [Path(path) for result in worker_results for path in result["record_paths"]]
    replayed_records = _replay_records(record_paths)

    measured_moves = sum(int(result["measured_moves"]) for result in worker_results)
    measured_seconds = (max(int(result["measured_end_ns"]) for result in worker_results) - min(int(result["measured_start_ns"]) for result in worker_results)) / 1_000_000_000
    moves_per_second = measured_moves / measured_seconds
    median_rss = int(statistics.median(int(result["rss_bytes"]) for result in worker_results))
    aborted = sum(int(result["aborted"]) for result in worker_results)
    observations = sum(int(result["observations"]) for result in worker_results)

    passed = (
        measured_moves >= args.target_moves
        and moves_per_second >= TARGET_MOVES_PER_SECOND
        and median_rss < TARGET_WORKER_RSS_BYTES
        and aborted == 0
        and replayed_records == sum(int(result["measured_games"]) + int(result["warmup_games"]) for result in worker_results)
        and observations > 0
    )

    summary = {
        "status": "passed" if passed else "failed",
        "workers": args.workers,
        "target_moves": args.target_moves,
        "measured_moves": measured_moves,
        "measured_seconds": measured_seconds,
        "moves_per_second": moves_per_second,
        "median_worker_rss_mb": median_rss / (1024 * 1024),
        "aborted_games": aborted,
        "observer_notifications": observations,
        "replayed_records": replayed_records,
        "output_dir": str(output_dir),
        "worker_results": worker_results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


def _worker_entry(task: WorkerTask, barrier: Any, result_queue: mp.Queue[dict[str, Any]]) -> None:
    try:
        result_queue.put(_run_worker(task, barrier))
    except BaseException:
        result_queue.put({"worker_id": task.worker_id, "error": traceback.format_exc()})
        raise


def _run_worker(task: WorkerTask, barrier: Any) -> dict[str, Any]:
    adapter = HexoEngineAdapter()
    engine_metadata = adapter.metadata()
    players = (GoalBenchmarkPlayer("p0", 0), GoalBenchmarkPlayer("p1", 1))
    worker_context = WorkerContext(
        worker_id=task.worker_id,
        engine_metadata=engine_metadata,
        metadata={"benchmark": "goal"},
    )
    for player in players:
        player.setup_worker(worker_context)

    output_dir = Path(task.output_dir)
    warmup_sink = JsonlRecordSink(output_dir / f"{task.batch_id}-warmup-worker-{task.worker_id}.jsonl", flush_on_write=False)
    measured_sink = JsonlRecordSink(output_dir / f"{task.batch_id}-worker-{task.worker_id}.jsonl", flush_on_write=False)
    measured_results = []
    observations = 0
    try:
        for game_index in range(task.warmup_games):
            run_match_loop(
                GameSpec(game_id=f"warmup-{task.worker_id}-{game_index}", seed=game_index),
                players,
                warmup_sink,
                engine_adapter=adapter,
                worker_context=worker_context,
                setup_players=False,
                close_players=False,
            )
        warmup_sink.close()
        barrier.wait(timeout=task.barrier_timeout_s)

        measured_start_ns = time.perf_counter_ns()
        for game_index in range(task.measured_games):
            measured_results.append(
                run_match_loop(
                    GameSpec(game_id=f"measured-{task.worker_id}-{game_index}", seed=game_index),
                    players,
                    measured_sink,
                    engine_adapter=adapter,
                    worker_context=worker_context,
                    setup_players=False,
                    close_players=False,
                )
            )
        measured_end_ns = time.perf_counter_ns()
        observations = sum(player.observations for player in players)
    finally:
        measured_sink.close()
        warmup_sink.close()
        for player in players:
            player.close()

    measured_moves = sum(result.turns for result in measured_results)
    aborted = sum(1 for result in measured_results if result.status == GameStatus.ABORTED)
    return {
        "worker_id": task.worker_id,
        "warmup_games": task.warmup_games,
        "measured_games": task.measured_games,
        "measured_moves": measured_moves,
        "aborted": aborted,
        "measured_start_ns": measured_start_ns,
        "measured_end_ns": measured_end_ns,
        "rss_bytes": _rss_bytes(),
        "observations": observations,
        "record_paths": [str(warmup_sink.path), str(measured_sink.path)],
    }


def _replay_records(record_paths: list[Path]) -> int:
    replayed = 0
    for path in record_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                state = new_game(seed=payload["seed"], scenario=payload["scenario"])
                for action in payload["actions"]:
                    body = action["action"]
                    if body["type"] != "placement":
                        raise AssertionError(f"unsupported action record in {path}: {body!r}")
                    apply_action(state, PlacementAction(AxialCoord(body["q"], body["r"])))
                replay_terminal = terminal(state)
                if payload["status"] == "completed":
                    if replay_terminal is None:
                        raise AssertionError(f"completed record did not replay to terminal: {payload['game_id']}")
                    if str(replay_terminal.winner) != payload["terminal"]["winner"]:
                        raise AssertionError(f"winner mismatch while replaying {payload['game_id']}")
                replayed += 1
    return replayed


def _rss_bytes() -> int:
    if os.name == "nt":
        import ctypes
        import ctypes.wintypes

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            ctypes.wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL

        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)

    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    multiplier = 1024 if sys.platform != "darwin" else 1
    return int(usage.ru_maxrss * multiplier)


def _collect_worker_results(result_queue: mp.Queue[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for _ in range(count):
        try:
            results.append(result_queue.get(timeout=5))
        except queue.Empty:
            break
    return sorted(results, key=lambda result: int(result.get("worker_id", -1)))


def _distribute(total: int, workers: int) -> tuple[int, ...]:
    base, remainder = divmod(total, workers)
    return tuple(base + (1 if index < remainder else 0) for index in range(workers))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Hexo runner performance-goal benchmark.")
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--target-moves", type=int, default=1_000_000)
    parser.add_argument("--measured-games", type=int)
    parser.add_argument("--warmup-games-per-worker", type=int, default=128)
    parser.add_argument("--barrier-timeout-s", type=float, default=300.0)
    parser.add_argument("--output-dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
