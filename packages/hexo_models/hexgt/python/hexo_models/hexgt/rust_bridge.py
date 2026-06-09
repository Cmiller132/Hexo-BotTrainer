"""Thin Python import/call boundary for hexgt Rust acceleration.

All native acceleration lives in `hexo_models._rust.hexgt`, registered from
`hexgt/rust/src`. This module keeps the import error message readable and gives
Python code named functions for native calls. Phase 0 exposes only
`capabilities()`; the candidate/window/graph builders (Phase 1) and the MCTS
session search (Phase 5) are added to this boundary as they land.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

try:
    from hexo_models import _rust
except ImportError as exc:  # pragma: no cover - native extension is required on use.
    _rust = None
    _IMPORT_ERROR: BaseException | None = exc
else:
    _IMPORT_ERROR = None


def capabilities() -> Mapping[str, Any]:
    """Return the Rust accelerator capability payload."""

    return _hexgt_module().capabilities()


def candidate_ids(state: object, n: int) -> list[int]:
    """Packed candidate action ids for a live engine state at radius `n` (§4)."""

    return list(_hexgt_module().hexgt_candidate_ids(state, int(n)))


def graph_facts(state: object, n: int) -> Mapping[str, Any]:
    """Full bounded-graph facts for a live engine state at radius `n`.

    The returned dict (consumed by `features.build_graph_tensors`) carries:

    - ``nodes``: ``node_type``/``node_q``/``node_r``/``node_owner``/
      ``node_recency``/``node_wcount``/``node_wempty`` (u8 columns arrive as
      bytes; the rest as ``list[int]``).
    - ``edges``: ``edge_src``/``edge_dst``/``edge_type`` (symmetric typed edges).
    - ``meta``: ``placements``, ``phase`` (turn-phase id), and ``first_stone``
      (this turn's first-stone coord or ``None``).
    - ``candidate_nodes`` / ``candidate_ids``: candidate node rows + their action
      ids in CSR/legal order (the priors order).

    This is the shared construction reused by the Phase-4 expand step and live
    MCTS, so training inputs == search inputs.
    """

    return _hexgt_module().hexgt_graph_facts(state, int(n))


def new_mcts_session(*, max_states: int | None = None, n: int | None = None) -> object:
    """Create a native hexgt MCTS session (transposition cache + subtree reuse).

    `n` is the candidate-set radius (move vocabulary); it is immutable for the
    session's lifetime so the state-hash transposition cache stays sound.
    """

    return _hexgt_module().HexgtMctsSession(max_states, n)


def mcts_session_search(
    session: object,
    game_keys: Sequence[int],
    states: Sequence[object],
    *,
    visits: int,
    c_puct: float,
    temperature: float,
    seed: int,
    evaluator: object,
    virtual_batch_size: int | None = None,
    active_root_limit: int | None = None,
    root_dirichlet_total_alpha: float | None = None,
    root_dirichlet_noise_fraction: float | None = None,
    root_policy_temperature: float | None = None,
    fpu_reduction: float | None = None,
    virtual_loss: float | None = None,
    widening_policy_mass: float | None = None,
    widening_max_children: int | None = None,
    widening_min_children: int | None = None,
    forced_playout_k: float | None = None,
    move_temperatures: Sequence[float] | None = None,
    per_root_visits: Sequence[int] | None = None,
    per_root_forced_playout_k: Sequence[float] | None = None,
    per_root_noise: Sequence[bool] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Search through a native hexgt MCTS session, preserving chosen subtrees.

    Arguments are forwarded in the PyO3 signature order expected by
    `rust/src/mcts.rs`. Each node materializes at most a policy-nucleus subset of
    its candidate moves (top-p widening), identical to dense_cnn.
    """

    return tuple(
        session.search(
            tuple(int(item) for item in game_keys),
            tuple(states),
            visits,
            c_puct,
            temperature,
            int(seed),
            evaluator,
            virtual_batch_size,
            active_root_limit,
            root_dirichlet_total_alpha,
            root_dirichlet_noise_fraction,
            root_policy_temperature,
            fpu_reduction,
            virtual_loss,
            widening_policy_mass,
            widening_max_children,
            widening_min_children,
            forced_playout_k,
            None if move_temperatures is None else [float(t) for t in move_temperatures],
            None if per_root_visits is None else [int(v) for v in per_root_visits],
            None if per_root_forced_playout_k is None else [float(k) for k in per_root_forced_playout_k],
            None if per_root_noise is None else [bool(b) for b in per_root_noise],
        )
    )


def _hexgt_module() -> Any:
    """Return the loaded native hexgt module or raise a clear error."""

    module = getattr(_rust, "hexgt", None) if _rust is not None else None
    if module is None:
        raise RuntimeError(f"hexgt Rust accelerator is unavailable: {_IMPORT_ERROR}")
    return module
