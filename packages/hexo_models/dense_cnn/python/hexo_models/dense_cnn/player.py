"""`hexo_runner` player adapter for dense CNN inference and search.

The runner speaks in generic player lifecycle methods and live engine states.
This adapter creates a dense CNN inference wrapper, keeps one persistent native
MCTS session for the game, searches the current live state in `decide`, and
clears native state when games start or finish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_runner.player import DecisionResult, FinalSummary, GameContext, PlayerIdentity, TransitionEvent, WorkerContext

from .inference import DenseCNNInference
from .mcts import BatchedMctsSession, new_mcts_session


@dataclass(slots=True)
class DenseCNNPlayer:
    """Runner-compatible player backed by dense CNN MCTS."""

    identity_id: str
    model: Any
    trainer: Any
    record_samples: bool = False
    identity: PlayerIdentity = field(init=False)
    inference: DenseCNNInference = field(init=False)
    mcts_session: BatchedMctsSession = field(init=False)

    def __post_init__(self) -> None:
        self.identity = PlayerIdentity(self.identity_id, label="Dense CNN")
        self.inference = DenseCNNInference(
            self.model,
            device=self.trainer.device,
            amp=self.trainer.config.training.amp,
        )
        self.mcts_session = new_mcts_session(
            max_states=self.trainer.config.selfplay.mcts_session_cache_max_states
        )

    def setup_worker(self, context: WorkerContext) -> None:
        _ = context

    def start_game(self, context: GameContext) -> None:
        _ = context
        # Runner games are independent. Clearing prevents a previous game's
        # subtree from surviving under the single player-side key used below.
        self.mcts_session.clear()

    def decide(self, state: object) -> DecisionResult:
        """Search the current live runner state and return one placement action."""

        selfplay = self.trainer.config.selfplay
        search = self.mcts_session.run(
            [0],
            [state],
            self.inference,
            visits=selfplay.search_visits,
            c_puct=selfplay.c_puct,
            temperature=0.0,
            virtual_batch_size=self.trainer.mcts_virtual_batch_size,
            active_root_limit=selfplay.mcts_active_root_limit,
            root_policy_temperature=selfplay.root_policy_temperature,
            fpu_reduction=selfplay.fpu_reduction,
            virtual_loss=selfplay.virtual_loss,
            widening_policy_mass=selfplay.widening_policy_mass,
            widening_max_children=selfplay.widening_max_children,
            widening_min_children=selfplay.widening_min_children,
        )[0]
        action = engine.PlacementAction(unpack_coord_id(search.action_id))
        return DecisionResult(
            action=action,
            diagnostics={
                "model": "hexo_models.dense_cnn",
                "root_value": search.root_value,
                "visits": search.visits,
            },
        )

    def observe_transition(self, transition: TransitionEvent) -> None:
        _ = transition

    def finish_game(self, final_summary: FinalSummary) -> None:
        _ = final_summary

    def close(self) -> None:
        return
