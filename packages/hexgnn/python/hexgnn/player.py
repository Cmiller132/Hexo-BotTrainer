"""`hexo_runner` player adapter for hexgnn (Model 2) inference + search.

Mirrors `dense_cnn/player.py`: the runner speaks generic player-lifecycle
methods over live engine states; this adapter wraps a `HexgnnInference` evaluator
and one persistent native `HexgnnMctsSession` per game, searches the current live
state in `decide`, and clears the session between games.

The DETERMINISTIC eval path (the default here) is what makes head-to-head
comparisons fair and repeatable: NO root Dirichlet noise, greedy action
selection (`temperature = 0`), and a fixed per-(game, move) seed. An optional
opening window (`opening_moves` > 0 at `opening_temperature` > 0) can be enabled
to diversify openings exactly like the dense_cnn evaluator, but defaults off.
All other search knobs (c_puct, widening, fpu, virtual loss, root-policy
temperature) match self-play so the only eval-vs-train differences are noise +
temperature — the §-plan matched-compute fairness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_runner.player import (
    DecisionResult,
    FinalSummary,
    GameContext,
    PlayerIdentity,
    TransitionEvent,
    WorkerContext,
)

from .config import HexgnnConfig
from .inference import HexgnnInference
from .mcts import HexgnnMctsSession, new_mcts_session


@dataclass(slots=True)
class HexgnnPlayer:
    """Runner-compatible player backed by the hexgnn dynamic-GNN MCTS."""

    identity_id: str
    model: Any
    config: HexgnnConfig
    device: str = "cuda"
    fp16: bool = True
    # Deterministic by default (greedy, no noise). Opening diversification mirrors
    # the dense_cnn evaluator and is OFF unless explicitly configured.
    eval_seed: int = 0
    opening_temperature: float = 0.0
    opening_moves: int = 0
    # Optional eval-only virtual batch override (fatter forwards -> fewer passes).
    # 0 -> use the calibrated self-play virtual batch.
    virtual_batch_size: int = 0
    identity: PlayerIdentity = field(init=False)
    inference: HexgnnInference = field(init=False)
    mcts_session: HexgnnMctsSession = field(init=False)
    _decisions_made: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.identity = PlayerIdentity(self.identity_id, label="hexgnn")
        self.inference = HexgnnInference(self.model, device=self.device, fp16=self.fp16)
        self.mcts_session = new_mcts_session(
            max_states=self.config.selfplay.mcts_session_cache_max_states,
            n=self.config.architecture.candidate_radius,
        )

    def setup_worker(self, context: WorkerContext) -> None:
        _ = context

    def start_game(self, context: GameContext) -> None:
        _ = context
        # Runner games are independent; clear so a prior game's subtree cannot
        # survive under the single player-side key used in decide().
        self.mcts_session.clear()
        self._decisions_made = 0

    def decide(self, state: object) -> DecisionResult:
        """Search the live runner state and return one placement action.

        Deterministic: greedy (temperature 0) with NO Dirichlet noise, except an
        optional opening window sampled at `opening_temperature`."""

        selfplay = self.config.selfplay
        move_index = self._decisions_made
        self._decisions_made += 1
        in_opening = move_index < self.opening_moves and self.opening_temperature > 0.0
        temperature = self.opening_temperature if in_opening else 0.0
        move_seed = int(self.eval_seed) * 1_000_003 + move_index
        vbatch = self.virtual_batch_size if self.virtual_batch_size > 0 else None

        search = self.mcts_session.run(
            [0],
            [state],
            self.inference,
            visits=selfplay.search_visits,
            c_puct=selfplay.c_puct,
            temperature=temperature,
            seed=move_seed,
            virtual_batch_size=vbatch,
            active_root_limit=selfplay.mcts_active_root_limit,
            # No root Dirichlet noise in eval -> repeatable, fair comparisons.
            # `forced_playout_k` is likewise omitted: forced playouts are a
            # self-play EXPLORATION device (guarantee visits to low-prior moves
            # for training diversity), not wanted in deterministic eval.
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
                "model": "hexo_models.hexgnn",
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
