"""Threaded multi-game runner for Strix-vs-opponent eval.

Plays ``games`` matches concurrently in a thread pool. Each game runs one Strix
player (whose leaf evaluations coalesce through a shared
:class:`hexo_strix.batch_server.StrixBatchServer`) against an opponent produced
by a caller-supplied factory (any ``hexo_runner`` RunnerPlayer — hexfield today,
another bot later). The Strix Rust search and graph builder release the GIL, so
the pool genuinely overlaps CPU search across cores while the server feeds the
GPU one stream of large batches.

Opponent-agnostic on purpose: the opponent is a ``factory(seed) -> RunnerPlayer``
the caller wires up (e.g. a hexfield checkpoint's native session). Colors
alternate across games. Returns an aggregate result dict (Strix perspective).

Thread-safety contract the caller must uphold: the opponent factory must return
a player that is safe to run in its own thread. For hexfield that means a fresh
session AND a per-game evaluator (do not share one evaluator's mutable buffers
across threads); the Strix side is safe because the shared model lives only in
the batch server's single batcher thread.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from hexo_runner.modes.match import run_match
from hexo_runner.records import GameStatus
from hexo_runner.session import GameSpec

PlayerFactory = Callable[[int], Any]


def run_threaded_matches(
    strix_factory: PlayerFactory,
    opponent_factory: PlayerFactory,
    *,
    games: int,
    seed: int,
    max_actions: int = 1024,
    out_dir: str | Path,
    n_threads: int = 8,
    alternate_colors: bool = True,
    on_game: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Play ``games`` Strix-vs-opponent matches concurrently; aggregate results.

    Parameters
    ----------
    strix_factory / opponent_factory : ``seed -> RunnerPlayer``. Called once per
        game (fresh players per game). ``strix_factory`` should be built with a
        shared batch server so leaves coalesce across the concurrent games.
    games : number of games. seed : base seed; game ``g`` uses ``seed + g``.
    n_threads : pool size (concurrent games in flight).
    alternate_colors : Strix is player0 on even game indices, player1 on odd.
    on_game : optional callback invoked (from a worker thread) with each game's
        result dict as it completes — for live progress logging.

    Returns a dict with ``meta`` / ``score`` (Strix perspective) / ``games``.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def play_one(g: int) -> dict[str, Any]:
        game_seed = seed + g
        strix = strix_factory(game_seed)
        opp = opponent_factory(game_seed)
        strix_is_p0 = (g % 2 == 0) if alternate_colors else True
        players = (strix, opp) if strix_is_p0 else (opp, strix)
        spec = GameSpec(
            game_id=f"strix_match_{g:04d}",
            seed=game_seed,
            is_evaluation=True,
            max_actions=max_actions,
        )
        result = run_match(spec, players, output_dir=out)
        strix_seat = "player0" if strix_is_p0 else "player1"
        status = result.status
        winner = result.winner if status == GameStatus.COMPLETED else None
        if status != GameStatus.COMPLETED:
            outcome = "incomplete"
        elif winner is None:
            outcome = "draw"
        elif str(winner) == strix_seat:
            outcome = "strix_win"
        else:
            outcome = "opp_win"
        row = {
            "game": g,
            "seed": game_seed,
            "strix_seat": strix_seat,
            "status": str(status),
            "winner": str(winner) if winner is not None else None,
            "outcome": outcome,
            # GameResult.turns = recorded stone placements (not two-stone turns).
            "turns": getattr(result, "turns", 0) or 0,
            "duration_ms": getattr(result, "duration_ms", 0.0) or 0.0,
        }
        if on_game is not None:
            on_game(row)
        return row

    rows: list[dict[str, Any]] = [None] * games  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max(1, int(n_threads))) as pool:
        futures = {pool.submit(play_one, g): g for g in range(games)}
        for fut in futures:
            g = futures[fut]
            rows[g] = fut.result()

    completed = sum(1 for r in rows if r["outcome"] != "incomplete")
    strix_wins = sum(1 for r in rows if r["outcome"] == "strix_win")
    opp_wins = sum(1 for r in rows if r["outcome"] == "opp_win")
    draws = sum(1 for r in rows if r["outcome"] == "draw")
    decided = strix_wins + opp_wins
    return {
        "meta": {
            "games": games,
            "seed": seed,
            "n_threads": n_threads,
            "alternate_colors": alternate_colors,
            "out_dir": str(out),
        },
        "score": {
            "completed": completed,
            "strix_wins": strix_wins,
            "opp_wins": opp_wins,
            "draws": draws,
            "decided": decided,
            "strix_winrate_decided": (strix_wins / decided) if decided else None,
            "strix_score": ((strix_wins + 0.5 * draws) / completed) if completed else None,
        },
        "games_detail": rows,
    }
