"""Direct match runner mode."""

from __future__ import annotations

from pathlib import Path

from ..engine import HexoEngineAdapter
from ..loop import run_match_loop
from ..player import RunnerPlayer
from ..records import GameResult, HexoRecordFile
from ..session import GameSpec


def run_match(
    spec: GameSpec,
    players: tuple[RunnerPlayer, RunnerPlayer],
    output_dir: str | Path = Path("data/replay"),
) -> GameResult:
    """Run one game through the same optimized path used by batch workers.

    Single-game and low-throughput use cases are represented as game_count=1,
    not by a separate record or engine path.
    """

    adapter = HexoEngineAdapter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    record_path = output_path / f"{spec.game_id}.hxr"
    with HexoRecordFile.create(record_path, adapter.metadata(), players) as record_file:
        return run_match_loop(spec, players, record_file, engine_adapter=adapter)
