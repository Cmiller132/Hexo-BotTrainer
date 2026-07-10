"""Faithful hexo-strix eval opponent: the REAL Rust Gumbel MCTS.

This is the "identical to how hexo-strix runs eval" path. It runs hexo-strix's
actual Gumbel-AlphaZero MCTS — the compiled Rust ``hexo_rs`` extension
(``gumbel_mcts_with_diagnostics``) built from SootyOwl/hexo-strix — and plays
``legal_moves[argmax(improved_policy)]``, exactly as
``hexo_a0.evaluate.play_eval_game`` does.

The network callback (``eval_fn``) uses THIS repo's dependency-light port of the
model + axis-graph builder (``hexo_strix.model`` / ``hexo_strix.graph``), which
was verified to produce logits/values numerically identical (max |Δlogit| = 0,
max |Δvalue| ≈ 6e-8) to hexo-strix's own torch_geometric model + Rust graph
builder. So this runs the real search over identical priors/values — no
torch_geometric or hexo_a0 needed at runtime, only the ``hexo_rs`` wheel.

Eval search defaults match hexo-strix's ``EvalConfig`` (eval games):
``n_simulations=256, m_actions=16, c_visit=50, c_scale=1.0``. Gumbel root noise
is disabled by default here so paired eval is deterministic/reproducible (the
paper's deterministic eval mode: candidates become top-m by logit, the search
and argmax(improved_policy) are then deterministic). Set
``disable_gumbel_noise=False`` (+ a seed) to mirror hexo-strix's stochastic
default.

A HeXO turn is two placements; the tree searches one placement per node and
eval runs a separate search per placement — matching this repo's autoregressive
one-placement-per-``decide`` runner, so no move buffering is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


import hexo_engine as engine

from hexo_runner.player import (
    DecisionResult,
    FinalSummary,
    GameContext,
    PlayerIdentity,
    TransitionEvent,
    WorkerContext,
)

from .batched_infer import (
    batched_eval_round,
    build_axis_graph_tensors,
    build_axis_round,
)
from .loader import StrixCheckpoint, load_strix_checkpoint

_ORIGIN = engine.PlacementAction(engine.AxialCoord(q=0, r=0))

# hexo-strix EvalConfig defaults (eval games).
DEFAULT_SIMS = 256
DEFAULT_M_ACTIONS = 16
DEFAULT_C_VISIT = 50
DEFAULT_C_SCALE = 1.0


def _player_str(player: engine.Player) -> str:
    # engine player0 opens at the origin -> hexo-strix P1.
    return "P1" if player == engine.Player.PLAYER_0 else "P2"


def _moves_remaining(phase: engine.TurnPhase) -> int:
    if phase == engine.TurnPhase.OPENING or phase == engine.TurnPhase.SECOND_STONE:
        return 1
    return 2


@dataclass
class StrixMctsPlayer:
    """RunnerPlayer running hexo-strix's real Rust Gumbel MCTS for eval."""

    checkpoint: StrixCheckpoint
    identity_id: str = "hexo-strix-mcts"
    device: str = "cpu"
    sims: int = DEFAULT_SIMS
    m_actions: int = DEFAULT_M_ACTIONS
    c_visit: int = DEFAULT_C_VISIT
    c_scale: float = DEFAULT_C_SCALE
    disable_gumbel_noise: bool = True
    # Opening-confined "light noise": when > 0, Gumbel root noise is ON for the
    # first ``noise_opening_plies`` plies of the game (strix SAMPLES its opening
    # among the top-m candidates) and OFF (greedy/deterministic) for every ply
    # after, REGARDLESS of ``disable_gumbel_noise``. This mirrors the hexfield
    # side's ``opening_temperature`` model: variance is injected in the opening,
    # the tail is deterministic. It keeps strix a STABLE anchor (the post-opening
    # play is reproducible, and the opening noise is seeded from ``seed`` so a
    # fixed seed replays identically across epochs) while still diversifying
    # games. The current ply is read from the board stone count, so the switch is
    # keyed on the GAME ply — correct regardless of seat or leader/follower role.
    # ``disable_gumbel_noise`` still governs plies OUTSIDE the opening window
    # (and all plies when ``noise_opening_plies == 0``).
    noise_opening_plies: int = 0
    seed: int | None = None
    label: str | None = None
    # Optional cross-game batch server (hexo_strix.batch_server.StrixBatchServer).
    # When set, the eval callback routes its leaves through the server so they
    # coalesce with other concurrent games' leaves into one GPU forward; the
    # server owns the shared model and device. When None the player runs its own
    # per-call batched forward on ``self.device``.
    server: Any = None
    identity: PlayerIdentity = field(init=False)
    _move_idx: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        import hexo_rs  # hard dependency for this path; imported lazily

        self._hexo_rs = hexo_rs
        self._server = self.server
        gc = self.checkpoint.game_config
        self._win_length = int(gc["win_length"])
        self._placement_radius = int(gc["placement_radius"])
        self._game_config = hexo_rs.GameConfig(
            self._win_length, self._placement_radius, int(gc.get("max_moves", 300))
        )
        self._mcts_config = hexo_rs.MCTSConfig(
            n_simulations=int(self.sims),
            m_actions=int(self.m_actions),
            c_visit=int(self.c_visit),
            c_scale=float(self.c_scale),
            disable_gumbel_noise=bool(self.disable_gumbel_noise),
        )
        # A noise-ON variant used only inside the opening window when
        # ``noise_opening_plies > 0`` (built once; identical knobs otherwise).
        self._mcts_config_noisy = (
            hexo_rs.MCTSConfig(
                n_simulations=int(self.sims),
                m_actions=int(self.m_actions),
                c_visit=int(self.c_visit),
                c_scale=float(self.c_scale),
                disable_gumbel_noise=False,
            )
            if int(self.noise_opening_plies) > 0
            else self._mcts_config
        )
        mc = self.checkpoint.model_config
        self._relative = bool(mc.get("relative_stone_encoding", True))
        self._threat = bool(mc.get("threat_features", True))
        self._prune = bool(mc.get("prune_empty_edges", True))
        # The server owns the shared model + device; a server-backed player must
        # NOT move the shared checkpoint.model (concurrent .to() across game
        # threads on one shared module is a hazard).
        self._model = None if self._server is not None else self.checkpoint.model.to(self.device).eval()
        self._eval_fn = self._make_eval_fn()
        self.identity = PlayerIdentity(
            player_id=self.identity_id,
            label=self.label or f"hexo-strix-mcts@{self.checkpoint.train_steps}",
            metadata={
                "adapter": "hexo_strix_mcts",
                "checkpoint": str(self.checkpoint.path),
                "train_steps": self.checkpoint.train_steps,
                "search": "gumbel_mcts",
                "sims": self.sims,
                "m_actions": self.m_actions,
                "c_visit": self.c_visit,
                "c_scale": self.c_scale,
                "disable_gumbel_noise": self.disable_gumbel_noise,
                "noise_opening_plies": self.noise_opening_plies,
            },
        )

    # --- network callback for the Rust MCTS ---
    def _graph_from_gamestate(self, s: Any):
        """Build the axis graph (CPU tensors) for a hexo_rs GameState.

        Delegates to the shared :func:`hexo_strix.batched_infer.build_axis_graph_tensors`
        so the serial, batched, and batch-server paths build graphs identically.
        Uses hexo-strix's COMPILED Rust builder; output is numerically identical
        to the pure-Python builder and the legal-node order equals
        ``s.legal_moves()`` (what the Rust MCTS expects).
        """
        return build_axis_graph_tensors(
            self._hexo_rs, s,
            prune_empty_edges=self._prune,
            threat_features=self._threat,
            relative_stones=self._relative,
        )

    def _make_eval_fn(self):
        """The Rust MCTS network callback.

        The Rust Gumbel search hands up to ``m_actions`` leaves per call (the
        sequential-halving rounds). The whole round's leaves are built in ONE
        rayon-parallel Rust call (:func:`build_axis_round`, zero-copy byte
        buffers) and scored in ONE forward. When ``self._server`` is set (the
        cross-game path) the pre-built :class:`RoundBatch` is handed to the
        shared batch server, which coalesces it with the other in-flight games'
        rounds into one GPU forward; otherwise the player runs the forward itself
        on ``self.device``.
        """
        model = self._model
        dev = self.device
        server = self._server
        hexo_rs = self._hexo_rs
        flags = dict(
            prune_empty_edges=self._prune,
            threat_features=self._threat,
            relative_stones=self._relative,
        )

        def eval_fn(states: list[Any]) -> tuple[list[list[float]], list[float]]:
            # One Rust call collates the round into disconnected-union byte
            # buffers already in the legal_moves() order the Rust MCTS expects.
            round_batch = build_axis_round(hexo_rs, states, **flags)
            if server is not None:
                return server.evaluate(round_batch)
            return batched_eval_round(model, round_batch, dev)

        return eval_fn

    # --- RunnerPlayer lifecycle ---
    def setup_worker(self, context: WorkerContext) -> None:
        return

    def start_game(self, context: GameContext) -> None:
        self._move_idx = 0

    def _move_seed(self) -> int | None:
        # Noise off => deterministic search, seed irrelevant. Noise on => derive
        # a distinct per-move seed from the base seed so games are diverse yet
        # reproducible (base=None => fully stochastic OS RNG).
        if self.disable_gumbel_noise:
            return self.seed
        if self.seed is None:
            return None
        s = self.seed * 100_003 + self._move_idx
        self._move_idx += 1
        return s

    def _search_config_and_seed(self, ply: int):
        """Pick the (MCTSConfig, seed) for a move at game ``ply``.

        Opening-confined noise (``noise_opening_plies > 0``): the first
        ``noise_opening_plies`` plies use the noise-ON config with a per-ply seed
        derived from ``self.seed`` (reproducible yet distinct per opening move);
        every later ply is greedy. Otherwise fall back to the global regime:
        noise-ON => per-move seed via ``_move_seed``; noise-OFF => deterministic
        (``self.seed``).
        """
        if int(self.noise_opening_plies) > 0:
            if ply < int(self.noise_opening_plies):
                seed = None if self.seed is None else (self.seed * 100_003 + ply)
                return self._mcts_config_noisy, seed
            return self._mcts_config, self.seed  # greedy tail
        return self._mcts_config, self._move_seed()

    def observe_transition(self, transition: TransitionEvent) -> None:
        return

    def finish_game(self, final_summary: FinalSummary) -> None:
        return

    def close(self) -> None:
        return

    # --- move selection ---
    def decide(self, state: engine.HexoState) -> DecisionResult:
        py = engine.to_python_state(state)

        if py.phase == engine.TurnPhase.OPENING:
            return DecisionResult(
                action=_ORIGIN, diagnostics={"adapter": "hexo_strix_mcts", "opening": True}
            )

        stones = [
            ((coord.q, coord.r), _player_str(player))
            for coord, player in py.board.stones
        ]
        game = self._hexo_rs.GameState.from_state(
            stones,
            _player_str(py.current_player),
            _moves_remaining(py.phase),
            self._game_config,
        )

        # The move about to be made is placement index len(stones); this keys the
        # opening-noise window on the true game ply (seat-independent).
        mcts_config, move_seed = self._search_config_and_seed(len(stones))
        (
            _action,
            improved_policy,
            _visit_counts,
            _per_child_q,
            _per_child_prior,
            _candidate_indices,
            _forced,
        ) = self._hexo_rs.gumbel_mcts_with_diagnostics(
            game, self._eval_fn, mcts_config, seed=move_seed
        )

        legal_moves = game.legal_moves()
        best_idx = max(range(len(improved_policy)), key=lambda i: improved_policy[i])
        q, r = legal_moves[best_idx]
        action = engine.PlacementAction(engine.AxialCoord(q=int(q), r=int(r)))
        if not engine.is_legal_action(state, action):
            legal = engine.legal_actions(state)
            return DecisionResult(
                action=legal[0],
                diagnostics={"adapter": "hexo_strix_mcts", "fallback": f"illegal_{q}_{r}"},
            )
        return DecisionResult(
            action=action,
            diagnostics={
                "adapter": "hexo_strix_mcts",
                "search": "gumbel_mcts",
                "sims": self.sims,
                "improved_policy_max": float(improved_policy[best_idx]),
            },
        )


def make_strix_mcts_factory(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    sims: int = DEFAULT_SIMS,
    m_actions: int = DEFAULT_M_ACTIONS,
    c_visit: int = DEFAULT_C_VISIT,
    c_scale: float = DEFAULT_C_SCALE,
    disable_gumbel_noise: bool = True,
    noise_opening_plies: int = 0,
    seed: int | None = None,
    identity_id: str = "hexo-strix-mcts",
    label: str | None = None,
    server: Any = None,
):
    """Return a ``PlayerFactory`` (seed -> StrixMctsPlayer) for run_head_to_head.

    Loads the checkpoint once and shares the model across games. With
    ``disable_gumbel_noise=True`` (default) the search is deterministic, so the
    per-game seed is unused; with noise enabled it is threaded into the Rust RNG
    for reproducible stochastic play.

    ``server`` optionally routes every game's leaf evaluations through a shared
    :class:`hexo_strix.batch_server.StrixBatchServer` (the cross-game batching
    path); the server owns the model + device, so ``device`` is then ignored for
    the forward.
    """
    ckpt = load_strix_checkpoint(checkpoint_path, device=device)

    def factory(game_seed: int) -> StrixMctsPlayer:
        return StrixMctsPlayer(
            checkpoint=ckpt,
            identity_id=identity_id,
            device=device,
            sims=sims,
            m_actions=m_actions,
            c_visit=c_visit,
            c_scale=c_scale,
            disable_gumbel_noise=disable_gumbel_noise,
            noise_opening_plies=noise_opening_plies,
            seed=(game_seed if (not disable_gumbel_noise or noise_opening_plies > 0) else seed),
            label=label,
            server=server,
        )

    return factory
