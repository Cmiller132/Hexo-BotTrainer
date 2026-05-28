"""Runner player adapter for dense CNN inference and search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_runner.player import DecisionResult, FinalSummary, GameContext, PlayerIdentity, TransitionEvent, WorkerContext

from .inference import DenseCNNInference
from .mcts import run_mcts


@dataclass(slots=True)
class DenseCNNPlayer:
    identity_id: str
    model: Any
    trainer: Any
    record_samples: bool = False
    identity: PlayerIdentity = field(init=False)
    inference: DenseCNNInference = field(init=False)

    def __post_init__(self) -> None:
        self.identity = PlayerIdentity(self.identity_id, label="Dense CNN")
        self.inference = DenseCNNInference(
            self.model,
            device=self.trainer.device,
            amp=self.trainer.config.training.amp,
        )

    def setup_worker(self, context: WorkerContext) -> None:
        _ = context

    def start_game(self, context: GameContext) -> None:
        _ = context

    def decide(self, state: object) -> DecisionResult:
        search = run_mcts(
            state,
            self.inference,
            visits=self.trainer.config.selfplay.search_visits,
            temperature=0.0,
            progressive_widening_initial_actions=self.trainer.config.selfplay.progressive_widening_initial_actions,
            progressive_widening_child_initial_actions=self.trainer.config.selfplay.progressive_widening_child_initial_actions,
            progressive_widening_candidate_actions=self.trainer.config.selfplay.progressive_widening_candidate_actions,
            progressive_widening_growth_interval=self.trainer.config.selfplay.progressive_widening_growth_interval,
            progressive_widening_growth_base=self.trainer.config.selfplay.progressive_widening_growth_base,
            hidden_prior_mass=self.trainer.config.selfplay.hidden_prior_mass,
            fpu_reduction=self.trainer.config.selfplay.fpu_reduction,
            virtual_loss=self.trainer.config.selfplay.virtual_loss,
        )
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
