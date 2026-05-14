"""Direct match runner mode."""

from __future__ import annotations

from ..loop import run_match_loop
from ..player import RunnerPlayer
from ..records import EventSink, GameResult, RecordSink
from ..session import GameSpec


def run_match(
    spec: GameSpec,
    players: tuple[RunnerPlayer, RunnerPlayer],
    sink: RecordSink,
    *,
    event_sink: EventSink | None = None,
) -> GameResult:
    """Run one game through session setup and the shared player/engine loop.

    This is the public match-mode entrypoint. It does not inspect engine state
    directly; it delegates to `run_match_loop`, which talks to `hexo_engine`
    through the public API and sends each player a cloned decision view.
    """

    return run_match_loop(spec, players, sink, event_sink=event_sink)
