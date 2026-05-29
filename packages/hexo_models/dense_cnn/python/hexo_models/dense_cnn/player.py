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
    # Opening diversification (eval). The first `opening_moves` decisions are
    # sampled at `opening_temperature` with a distinct per-(game, move) seed;
    # afterwards play is greedy (temperature 0). Both default to the old
    # fully-deterministic behavior.
    eval_seed: int = 0
    opening_temperature: float = 0.0
    opening_moves: int = 0
    identity: PlayerIdentity = field(init=False)
    inference: DenseCNNInference = field(init=False)
    mcts_session: BatchedMctsSession = field(init=False)
    _decisions_made: int = field(init=False, default=0)

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
        self._decisions_made = 0

    def decide(self, state: object) -> DecisionResult:
        """Search the current live runner state and return one placement action."""

        selfplay = self.trainer.config.selfplay
        # Sample the opening at a small temperature so eval games diverge; go
        # greedy once past the configured opening. The per-move seed only
        # affects sampling (temperature > 0); greedy selection ignores it.
        move_index = self._decisions_made
        self._decisions_made += 1
        in_opening = move_index < self.opening_moves and self.opening_temperature > 0.0
        temperature = self.opening_temperature if in_opening else 0.0
        move_seed = int(self.eval_seed) * 1_000_003 + move_index
        search = self.mcts_session.run(
            [0],
            [state],
            self.inference,
            visits=selfplay.search_visits,
            c_puct=selfplay.c_puct,
            temperature=temperature,
            seed=move_seed,
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
