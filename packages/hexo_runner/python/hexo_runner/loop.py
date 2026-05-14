"""Core runner loop boundary for one game."""

from __future__ import annotations

from typing import Any, Mapping

import hexo_engine as engine

from .player import DecisionRequest, EngineDecisionView, FinalSummary, TransitionEvent, RunnerPlayer
from .records import EventKind, EventSink, GameResult, GameStatus, PositionRecord, RecordSink, RunnerEvent
from .session import GameSpec, create_session_context


def run_match_loop(
    spec: GameSpec,
    players: tuple[RunnerPlayer, RunnerPlayer],
    sink: RecordSink,
    *,
    event_sink: EventSink | None = None,
) -> GameResult:
    """Run one game by mediating between generic players and the engine API.

    The runner owns exactly one authoritative state: `primary_state` below.
    Players never receive that object. On every decision, the runner snapshots
    the primary state and builds a cloned `EngineDecisionView` for the active
    player. The returned action is then applied back to the primary state.
    """

    if len(players) != 2:
        raise ValueError("run_match_loop requires exactly two players.")

    # Session creation is the only place this loop creates the authoritative
    # engine state. The context is shared with players for setup/provenance, but
    # per-turn decisions receive a cloned state through `EngineDecisionView`.
    context = create_session_context(spec, players)
    primary_state = context.state_ref
    turn_index = 0
    result = GameResult(game_id=spec.game_id, status=GameStatus.ABORTED)

    _emit(event_sink, EventKind.SESSION_STARTED, spec.game_id, payload={"metadata": dict(spec.metadata)})

    try:
        # Give both players the immutable-ish session setup context before any
        # moves are requested. This is for IDs, seed, mode, metadata, and setup.
        for player in players:
            player.initialize(context)

        # The engine is the only authority for terminal state. The loop keeps
        # asking for decisions until the engine reports a terminal result.
        while engine.terminal(primary_state) is None:
            # These are read from the primary state before asking the player:
            # - snapshot: replayable copy source for the player-owned clone
            # - state_id: stable identity for records/diagnostics
            # - current: engine player to move
            # - legal_actions: actions the active player may choose from
            before_snapshot = engine.snapshot(primary_state)
            before_state_id = engine.state_id(primary_state)
            current = engine.current_player(primary_state)
            legal_actions = tuple(engine.legal_actions(primary_state))
            active_player = players[_player_index(current)]

            # This is the full payload sent to the active player. The top-level
            # request repeats the most ergonomic fields, while `state` contains
            # the richer typed engine view built by `_build_decision_view`.
            request = DecisionRequest(
                game_id=spec.game_id,
                turn_index=turn_index,
                current_player=current,
                state=_build_decision_view(primary_state, before_snapshot, before_state_id),
                legal_actions=legal_actions,
                seed=spec.seed,
                is_evaluation=spec.is_evaluation,
                metadata={
                    "mode": spec.mode,
                    "state_id": before_state_id,
                    "engine": dict(context.engine_metadata),
                },
            )
            _emit(
                event_sink,
                EventKind.DECISION_REQUESTED,
                spec.game_id,
                turn_index,
                {"current_player": current.value, "legal_count": len(legal_actions)},
            )

            # The active player only chooses. It does not mutate the primary
            # state. Any search it does should happen against request.state.state_ref,
            # which is a cloned engine state.
            decision = active_player.decide(request)

            # All real state changes go through the engine public API here.
            # Illegal actions or player errors fall into the single abort path.
            transition = engine.apply_action(primary_state, decision.action)

            # After the engine accepts the action, capture the new primary-state
            # snapshot and IDs for durable records and observers.
            after_snapshot = engine.snapshot(primary_state)
            after_state_id = engine.state_id(primary_state)
            outcome = engine.terminal(primary_state)
            sink.write_entry(
                PositionRecord(
                    game_id=spec.game_id,
                    turn_index=turn_index,
                    player_id=active_player.identity.player_id,
                    before_snapshot=before_snapshot,
                    action=decision.action,
                    after_snapshot=after_snapshot,
                    terminal=outcome,
                    metadata={
                        "before_state_id": before_state_id,
                        "after_state_id": after_state_id,
                        "action_id": engine.action_id(decision.action),
                        "decision_diagnostics": dict(decision.diagnostics),
                        "transition_metadata": dict(transition.metadata),
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
            )
            for player in players:
                player.observe_transition(transition_event)

            _emit(
                event_sink,
                EventKind.ACTION_APPLIED,
                spec.game_id,
                turn_index,
                {
                    "player": active_player.identity.player_id,
                    "before_state_id": before_state_id,
                    "after_state_id": after_state_id,
                    "terminal": outcome is not None,
                },
            )
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
        _emit(
            event_sink,
            EventKind.PLAYER_ERROR,
            spec.game_id,
            turn_index,
            {"error": str(exc)},
        )
        result = GameResult(
            game_id=spec.game_id,
            status=GameStatus.ABORTED,
            metadata={"reason": "runner_error", "error": str(exc), "turns": turn_index},
        )

    # Finalization happens for completed and aborted games: close durable
    # records, pass a final summary to both players, and emit a final event.
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
    _emit(
        event_sink,
        EventKind.SESSION_FINISHED,
        spec.game_id,
        payload={"status": result.status.value, "winner": getattr(result.winner, "value", result.winner)},
    )
    return result


def _build_decision_view(
    primary_state: engine.EngineStateRef,
    snapshot: engine.EngineSnapshot,
    state_id: str,
) -> EngineDecisionView:
    """Build the exact typed engine payload sent to a player.

    Every field comes from the public engine API. `snapshot` is captured from
    the primary state before the call, then `load_snapshot(snapshot)` creates a
    separate state ref for the player. That cloned `state_ref` is safe for MCTS
    or other search code to mutate because the runner never applies it back.
    """

    cloned_state = engine.load_snapshot(snapshot)
    return EngineDecisionView(
        # Player-owned mutable clone. This is not `primary_state`.
        state_ref=cloned_state,
        # Replayable primary-state snapshot used to create the clone.
        snapshot=snapshot,
        # Stable ID of the primary state before the player decision.
        state_id=state_id,
        # Convenience reads from the cloned state; these should mirror primary.
        current_player=engine.current_player(cloned_state),
        turn_placement=engine.turn_placement(cloned_state),
        # Raw engine-owned state shape for model/features/UI adapters.
        game_state=engine.game_state(cloned_state),
        # Legal actions generated by the engine from the cloned state.
        legal_actions=tuple(engine.legal_actions(cloned_state)),
        # Raw window/tactical data from the engine, not dashboard interpretation.
        tactics=dict(engine.tactics(cloned_state)),
        terminal=engine.terminal(cloned_state),
        # Provenance: links this clone/view back to the primary state identity.
        metadata={"primary_state_id": engine.state_id(primary_state)},
    )


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


def _emit(
    sink: EventSink | None,
    kind: EventKind,
    game_id: str,
    turn_index: int | None = None,
    payload: Mapping[str, Any] | None = None,
) -> None:
    if sink is not None:
        sink.emit(RunnerEvent(kind=kind, game_id=game_id, turn_index=turn_index, payload=payload or {}))
