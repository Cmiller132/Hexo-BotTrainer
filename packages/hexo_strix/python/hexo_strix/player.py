"""RunnerPlayer adapter exposing a hexo-strix checkpoint as an eval opponent.

Implements the repo's cross-package ``RunnerPlayer`` contract
(``hexo_runner.player``): one placement per ``decide`` call (the engine turn is
autoregressive — opening stone, then FirstStone/SecondStone). Move selection is
raw-policy greedy (argmax over the network's legal-node logits), matching
hexo-strix's ``policy_viewer`` with ``mcts_sims=0``; deterministic, no Gumbel
MCTS, so it is a fair fixed-strength eval anchor.

Board -> graph uses hexo-strix's own ``placement_radius`` (from the checkpoint's
``game_config``, radius 6) to enumerate candidate empty nodes — a subset of this
engine's radius-8 legal set, so every move it picks is legal here. The move is
validated with ``engine.is_legal_action`` regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

import hexo_engine as engine

from hexo_runner.player import (
    DecisionResult,
    FinalSummary,
    GameContext,
    PlayerIdentity,
    TransitionEvent,
    WorkerContext,
)

from .graph import P1, P2, build_axis_graph
from .loader import StrixCheckpoint, load_strix_checkpoint

_ORIGIN = engine.PlacementAction(engine.AxialCoord(q=0, r=0))


def _to_move_int(player: engine.Player) -> int:
    # player0 opens at the origin -> hexo-strix P1; player1 -> P2.
    return P1 if player == engine.Player.PLAYER_0 else P2


def _moves_remaining(phase: engine.TurnPhase) -> int:
    # Duplicates the engine turn rule (see sealbot adapter _moves_left_in_turn):
    # opening stone and the second stone of a turn leave 1, otherwise 2.
    if phase == engine.TurnPhase.OPENING or phase == engine.TurnPhase.SECOND_STONE:
        return 1
    return 2


@dataclass
class StrixPlayer:
    """RunnerPlayer wrapping a ported HeXONet checkpoint (raw-policy greedy)."""

    checkpoint: StrixCheckpoint
    identity_id: str = "hexo-strix"
    device: str = "cpu"
    label: str | None = None
    identity: PlayerIdentity = field(init=False)

    def __post_init__(self) -> None:
        self._win_length = int(self.checkpoint.game_config["win_length"])
        self._placement_radius = int(self.checkpoint.game_config["placement_radius"])
        mc = self.checkpoint.model_config
        self._relative = bool(mc.get("relative_stone_encoding", True))
        self._threat = bool(mc.get("threat_features", True))
        self._prune = bool(mc.get("prune_empty_edges", True))
        self._model = self.checkpoint.model.to(self.device).eval()
        self.identity = PlayerIdentity(
            player_id=self.identity_id,
            label=self.label or f"hexo-strix@{self.checkpoint.train_steps}",
            metadata={
                "adapter": "hexo_strix",
                "checkpoint": str(self.checkpoint.path),
                "train_steps": self.checkpoint.train_steps,
                "win_length": self._win_length,
                "placement_radius": self._placement_radius,
                "search": "raw_policy_greedy",
            },
        )

    # --- RunnerPlayer lifecycle (mostly no-ops; model is stateless) ---
    def setup_worker(self, context: WorkerContext) -> None:
        return

    def start_game(self, context: GameContext) -> None:
        return

    def observe_transition(self, transition: TransitionEvent) -> None:
        return

    def finish_game(self, final_summary: FinalSummary) -> None:
        return

    def close(self) -> None:
        return

    # --- move selection ---
    def decide(self, state: engine.HexoState) -> DecisionResult:
        py = engine.to_python_state(state)

        # Opening: the origin is the only legal cell and hexo-strix always
        # opens there. Skip the network (it needs >= 1 stone to build a graph).
        if py.phase == engine.TurnPhase.OPENING:
            return DecisionResult(
                action=_ORIGIN, diagnostics={"adapter": "hexo_strix", "opening": True}
            )

        stones = [
            ((coord.q, coord.r), _to_move_int(player))
            for coord, player in py.board.stones
        ]
        to_move = _to_move_int(py.current_player)
        graph = build_axis_graph(
            stones,
            to_move=to_move,
            moves_remaining=_moves_remaining(py.phase),
            win_length=self._win_length,
            placement_radius=self._placement_radius,
            prune_empty_edges=self._prune,
            threat_features=self._threat,
            relative_stones=self._relative,
        )

        if not graph.legal_coords:
            # No radius-6 candidate (extremely unlikely with >=1 stone). Fall
            # back to any engine-legal move to avoid aborting the game.
            return self._fallback(state, reason="no_candidates")

        dev = self.device
        with torch.no_grad():
            logits, value = self._model(
                graph.x.to(dev),
                graph.edge_index.to(dev),
                graph.legal_mask.to(dev),
                graph.stone_mask.to(dev),
                edge_attr=graph.edge_attr.to(dev),
            )
            best_idx = int(torch.argmax(logits).item())
            value_f = float(value.item())

        q, r = graph.legal_coords[best_idx]
        action = engine.PlacementAction(engine.AxialCoord(q=q, r=r))
        if not engine.is_legal_action(state, action):
            return self._fallback(state, reason=f"illegal_{q}_{r}")

        return DecisionResult(
            action=action,
            diagnostics={
                "adapter": "hexo_strix",
                "value": value_f,
                "policy_logit": float(logits[best_idx].item()),
                "n_candidates": len(graph.legal_coords),
            },
        )

    def _fallback(self, state: engine.HexoState, *, reason: str) -> DecisionResult:
        legal = engine.legal_actions(state)
        if len(legal) == 0:
            raise ValueError("hexo_strix: no legal actions available.")
        action = legal[0]  # LegalActions[int] -> PlacementAction
        return DecisionResult(
            action=action, diagnostics={"adapter": "hexo_strix", "fallback": reason}
        )


def make_strix_factory(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    identity_id: str = "hexo-strix",
    label: str | None = None,
):
    """Return a ``PlayerFactory`` (seed -> StrixPlayer) for run_head_to_head.

    The checkpoint is loaded once and shared across games (the model is
    stateless for eval); the seed is unused because selection is deterministic.
    """
    ckpt = load_strix_checkpoint(checkpoint_path, device=device)

    def factory(seed: int) -> StrixPlayer:
        return StrixPlayer(
            checkpoint=ckpt, identity_id=identity_id, device=device, label=label
        )

    return factory
