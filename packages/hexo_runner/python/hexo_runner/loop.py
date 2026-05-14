"""Core runner loop boundary for one game."""

from __future__ import annotations

import hexo_engine as engine

from .player import FinalSummary, TransitionEvent, RunnerPlayer
from .records import GameResult, GameStatus, PositionRecord, RecordSink
from .session import GameSpec, create_session_context


def run_match_loop(
    spec: GameSpec,
    players: tuple[RunnerPlayer, RunnerPlayer],
    sink: RecordSink,
) -> GameResult:
    """Run one game by mediating between generic players and the engine API.

    The runner owns exactly one authoritative state: `primary_state` below.
    Players never receive that object. On every decision, the runner gives the
    active player only a cloned `HexoState`. The returned action is then
    applied back to the primary state.
    """

    if len(players) != 2:
        raise ValueError("run_match_loop requires exactly two players.")

    # Session creation is the only place this loop creates the authoritative
    # engine state. The context is shared with players for setup/provenance, but
    # per-turn decisions receive only a cloned engine state.
    context = create_session_context(spec, players)
    primary_state = context.state
    turn_index = 0
    result = GameResult(game_id=spec.game_id, status=GameStatus.ABORTED)

    try:
        # Give both players the immutable-ish session setup context before any
        # moves are requested. This is for IDs, seed, mode, metadata, and setup.
        for player in players:
            player.initialize(context)

        # The engine is the only authority for terminal state. The loop keeps
        # asking for decisions until the engine reports a terminal result.
        while engine.terminal(primary_state) is None:
            # The runner asks the engine whose turn it is, clones the current
            # state for that player, and keeps the authoritative state private.
            current = engine.current_player(primary_state)
            active_player = players[_player_index(current)]
            player_state = engine.clone_state(primary_state)

            # This cloned state is the entire player payload. Search players may
            # mutate it freely and query tactics/legal moves through the engine.
            decision = active_player.decide(player_state)

            # All real state changes go through the engine public API here.
            # Illegal actions or player errors fall into the single abort path.
            transition = engine.apply_action(primary_state, decision.action)

            outcome = engine.terminal(primary_state)
            sink.write_entry(
                PositionRecord(
                    game_id=spec.game_id,
                    turn_index=turn_index,
                    player_id=active_player.identity.player_id,
                    action=decision.action,
                    terminal=outcome,
                    metadata={
                        "action_id": engine.action_id(decision.action),
                        "decision_diagnostics": dict(decision.diagnostics),
                    },
                )
            )

            # Notify both players about the accepted transition so model/search
            # players can update internal state, caches, or logs.
            transition_event = TransitionEvent(
                game_id=spec.game_id,
                turn_index=turn_index,
                action=decision.action,
                transition=transition,
                state=engine.clone_state(primary_state),
            )
            for player in players:
                player.observe_transition(transition_event)

            turn_index += 1

        # If the loop exits naturally, the engine has produced a terminal state.
        outcome = engine.terminal(primary_state)
        result = GameResult(
            game_id=spec.game_id,
            status=GameStatus.COMPLETED,
            terminal=outcome,
            winner=_terminal_winner(outcome),
            metadata={"turns": turn_index},
        )
    except Exception as exc:
        # The runner does not forfeit or assign an opponent winner. Any player,
        # sink, or engine error aborts the game and preserves the error in the
        # result metadata.
        result = GameResult(
            game_id=spec.game_id,
            status=GameStatus.ABORTED,
            metadata={"reason": "runner_error", "error": str(exc), "turns": turn_index},
        )

    # Finalization happens for completed and aborted games: close durable
    # records and pass a final summary to both players.
    record_ref = sink.close_game(spec.game_id, result.terminal)
    result = GameResult(
        game_id=result.game_id,
        status=result.status,
        terminal=result.terminal,
        winner=result.winner,
        record_ref=record_ref,
        analysis=result.analysis,
        metadata=result.metadata,
    )
    summary = FinalSummary(game_id=spec.game_id, result=result, metadata=result.metadata)
    for player in players:
        player.close(summary)
    return result


def _player_index(player: engine.Player) -> int:
    if player == engine.Player.PLAYER_0:
        return 0
    if player == engine.Player.PLAYER_1:
        return 1
    raise ValueError(f"Unknown engine player: {player!r}")


def _terminal_winner(outcome: object | None) -> object | None:
    if outcome is None:
        return None
    return getattr(outcome, "winner", None)
