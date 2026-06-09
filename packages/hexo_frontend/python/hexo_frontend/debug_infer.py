"""CPU-only hexgt inference for the dashboard Debug tab.

This module is the *inference library* behind the Debug tab: it loads any hexgt
RL checkpoint (pre- or post-STV-graft), reconstructs a board position from a move
sequence, and returns what the model "thinks" — per-candidate policy prior, the
full 65-bin distributional value (+ scalar), the opponent-policy head, the STV
lookahead heads, and (on demand) a fresh CPU MCTS visit distribution.

Everything here is **CPU-only by construction**: models are built and run on
``torch.device("cpu")`` and the MCTS evaluator is constructed with
``device="cpu", fp16=False``. The worker process that imports this module is also
launched with ``CUDA_VISIBLE_DEVICES=""`` so it can never touch the training GPU.

The loader mirrors the RL driver's resume recipe (``scripts/_rl_train.py``) so an
old checkpoint maps cleanly onto the current architecture:

    build model (STV horizons on)
      -> expand_value_readout_columns   (SIDE-only -> [SIDE|PMA], zero-init PMA)
      -> expand_stv_readout_columns     (same, per STV head)
      -> load_state_dict(strict=False)
      -> zero_init_expanded_feature_columns (if the checkpoint's feature schema
                                             predates the current one)

Checkpoints in ``hexgt_rl_main3`` are self-describing via ``ck["arch"]``; epochs
0-6 are pre-graft (heads read the SIDE hub only), 7+ are post-graft.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id

from hexo_models.hexgt.architecture import (
    HexgtNetwork,
    expand_stv_readout_columns,
    expand_value_readout_columns,
    zero_init_expanded_feature_columns,
)
from hexo_models.hexgt.constants import (
    DEFAULT_CANDIDATE_RADIUS,
    FEATURE_SCHEMA_VERSION,
    VALUE_BINS,
    feature_slots_after,
)
from hexo_models.hexgt import rust_bridge
from hexo_models.hexgt.collate import collate_graphs
from hexo_models.hexgt.features import build_graph_tensors
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.losses import decode_binned_value, value_bins
from hexo_models.hexgt.mcts import new_mcts_session

# STV horizons the current architecture always carries. The driver forces these
# on at resume; we mirror that so every loaded checkpoint exposes the STV heads.
STV_HORIZONS: tuple[int, ...] = (4, 12, 24)


# ---------------------------------------------------------------------------
# Checkpoint loading (graft-aware, CPU-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedModel:
    """A CPU model ready for debug inference, plus provenance metadata."""

    model: HexgtNetwork
    candidate_radius: int
    arch: dict[str, Any]
    rl_epoch: int | None
    step: int | None
    graft: str  # "pre" | "post" — whether the readout heads were SIDE-only
    expanded_value: bool
    expanded_stv: list[str]
    zeroed_feature_cols: list[int]
    load_warnings: list[str] = field(default_factory=list)


def _arch_kwargs(arch: dict[str, Any]) -> dict[str, Any]:
    """Translate a checkpoint ``arch`` dict into HexgtNetwork kwargs, forcing the
    current STV horizons on (so the heads exist to be inspected)."""

    kwargs: dict[str, Any] = {
        "token_dim": int(arch["token_dim"]),
        "gnn_layers": int(arch["gnn_layers"]),
        "ctx_layers": int(arch["ctx_layers"]),
        "ffn_dim": int(arch["ffn_dim"]),
        "attention_heads": int(arch["attention_heads"]),
        "short_term_value_horizons": STV_HORIZONS,
        "value_pma_seeds": int(arch.get("value_pma_seeds", 2)),
    }
    if "value_head_use_side" in arch:
        kwargs["value_head_use_side"] = bool(arch["value_head_use_side"])
    if "node_feature_dim" in arch:
        kwargs["node_feature_dim"] = int(arch["node_feature_dim"])
    return kwargs


def load_checkpoint(path: str | Path) -> LoadedModel:
    """Load a hexgt checkpoint onto CPU, handling pre/post-graft shape drift."""

    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload or "arch" not in payload:
        raise ValueError(
            f"{ckpt_path.name}: not a hexgt RL checkpoint (missing 'model'/'arch')"
        )
    arch = dict(payload["arch"])
    state_dict = payload["model"]

    model = HexgtNetwork(**_arch_kwargs(arch))

    # Detect the graft generation BEFORE expanding: a pre-graft (<=epoch 6) STV
    # head reads the SIDE hub only (width == token_dim); the epoch-7 graft widened
    # it to [SIDE|PMA]. (The value head was widened in an earlier phase, so the STV
    # head width is the signal for the epoch-7 boundary.)
    token_dim = model.token_dim
    stv_w = state_dict.get(f"short_term_value_heads.{STV_HORIZONS[0]}.0.weight")
    graft = "post"
    if stv_w is not None and tuple(stv_w.shape)[1] == token_dim and model.value_readout_blocks > 1:
        graft = "pre"

    expanded_value = expand_value_readout_columns(model, state_dict)
    expanded_stv = expand_stv_readout_columns(model, state_dict)

    result = model.load_state_dict(state_dict, strict=False)
    warnings: list[str] = []
    # Only STV-head params may legitimately be missing (a seed without the heads);
    # anything else missing/unexpected is worth surfacing.
    nonstv_missing = [k for k in result.missing_keys if "short_term_value_heads" not in k]
    if nonstv_missing:
        warnings.append(f"missing keys: {nonstv_missing[:8]}")
    if result.unexpected_keys:
        warnings.append(f"unexpected keys: {list(result.unexpected_keys)[:8]}")

    # Feature-schema graft: zero columns activated after the checkpoint's version,
    # so the first forward matches the checkpoint (mirrors the driver).
    loaded_fsv = int(payload.get("feature_schema_version", 1))
    zeroed: list[int] = []
    if loaded_fsv < FEATURE_SCHEMA_VERSION:
        zeroed = zero_init_expanded_feature_columns(model, feature_slots_after(loaded_fsv))

    model.eval()
    return LoadedModel(
        model=model,
        candidate_radius=int(arch.get("candidate_radius", DEFAULT_CANDIDATE_RADIUS)),
        arch=arch,
        rl_epoch=_maybe_int(payload.get("rl_epoch")),
        step=_maybe_int(payload.get("step")),
        graft=graft,
        expanded_value=expanded_value,
        expanded_stv=expanded_stv,
        zeroed_feature_cols=zeroed,
        load_warnings=warnings,
    )


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Position reconstruction
# ---------------------------------------------------------------------------


def state_from_actions(action_ids: Sequence[int]):
    """Replay a move sequence into a fresh engine state (CPU, no model)."""

    state = engine.new_game()
    for aid in action_ids:
        engine.apply_action(state, engine.PlacementAction(unpack_coord_id(int(aid))))
    return state


def _coord_of(action_id: int) -> dict[str, int]:
    coord = unpack_coord_id(int(action_id))
    return {"q": int(coord.q), "r": int(coord.r)}


# ---------------------------------------------------------------------------
# Inference (all heads) for one position
# ---------------------------------------------------------------------------


@torch.no_grad()
def analyze_position(
    loaded: LoadedModel,
    action_ids: Sequence[int],
    *,
    n: int | None = None,
) -> dict[str, Any]:
    """Full-head readout for the position reached by ``action_ids``.

    Returns per-candidate policy prior + opponent-policy, the 65-bin value
    distribution (+ scalar), and each STV lookahead head (scalar + distribution),
    all from the side-to-move's perspective.
    """

    radius = int(n if n is not None else loaded.candidate_radius)
    state = state_from_actions(action_ids)
    return analyze_state(loaded, state, n=radius)


def _swap_owner(facts: dict[str, Any]) -> dict[str, Any]:
    """Return facts with stone ownership flipped (0<->1), leaving empty/SIDE
    untouched. ``build_graph_tensors`` re-derives every feature from this — so
    the swapped graph is the IDENTICAL board seen from the opponent's side. This
    matches the optimism/calibration probe in ``_optimism_main3.py`` exactly."""

    owners = list(facts["nodes"]["node_owner"])
    swapped = [1 if o == 0 else (0 if o == 1 else o) for o in owners]
    return {**facts, "nodes": {**facts["nodes"], "node_owner": swapped}}


@torch.no_grad()
def analyze_state(loaded: LoadedModel, state: Any, *, n: int) -> dict[str, Any]:
    model = loaded.model
    facts = rust_bridge.graph_facts(state, n)
    batch = collate_graphs([build_graph_tensors(facts)])
    out = model.forward(batch)

    candidate_ids = batch["candidate_ids"].cpu().numpy().astype(np.int64)
    policy_logits = out["policy"].float()
    priors = torch.softmax(policy_logits, dim=0).cpu().numpy()
    candidates = _candidate_rows(candidate_ids, priors)

    opp = None
    if "opp_policy" in out:
        opp_logits = out["opp_policy"].float()
        opp_priors = torch.softmax(opp_logits, dim=0).cpu().numpy()
        opp = _candidate_rows(candidate_ids, opp_priors)

    value_logits = out["value"].float().reshape(-1)
    value_dist = torch.softmax(value_logits, dim=0).cpu().numpy()
    value_scalar = float(decode_binned_value(value_logits.reshape(1, -1)).reshape(()).item())

    stv: dict[str, Any] = {}
    for horizon in STV_HORIZONS:
        key = f"stvalue_{horizon}"
        if key in out:
            logits = out[key].float().reshape(-1)
            stv[str(horizon)] = {
                "scalar": float(decode_binned_value(logits.reshape(1, -1)).reshape(()).item()),
                "dist": [round(float(x), 5) for x in torch.softmax(logits, dim=0).cpu().numpy()],
            }

    # Both-perspectives / optimism probe: evaluate the IDENTICAL board with stone
    # ownership flipped (0<->1). hexgt value is from the side-to-move's view, so a
    # well-calibrated zero-sum model gives v_self + v_swapped ~= 0; a positive sum
    # is "optimism" (both sides think they are winning).
    swapped_batch = collate_graphs([build_graph_tensors(_swap_owner(facts))])
    swapped_logits = model.forward_policy_value(swapped_batch)["value"].float().reshape(-1)
    value_swapped = float(decode_binned_value(swapped_logits.reshape(1, -1)).reshape(()).item())

    current = engine.current_player(state)
    current_role = getattr(current, "value", str(current))
    current_index = 1 if str(current_role).endswith("1") else 0

    return {
        "current_player": current_index,
        "current_role": str(current_role),
        "candidate_count": int(candidate_ids.size),
        "legal_count": int(engine.legal_action_count(state)),
        "value": value_scalar,
        "value_swapped": value_swapped,
        "optimism": round(value_scalar + value_swapped, 5),
        "value_bins": _value_bin_centers(),
        "value_dist": [round(float(x), 5) for x in value_dist],
        "policy": candidates,
        "opp_policy": opp,
        "stvalue": stv,
    }


def _candidate_rows(candidate_ids: np.ndarray, priors: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for aid, prob in zip(candidate_ids.tolist(), priors.tolist()):
        coord = _coord_of(aid)
        rows.append({"action_id": int(aid), "q": coord["q"], "r": coord["r"], "p": round(float(prob), 6)})
    rows.sort(key=lambda r: r["p"], reverse=True)
    return rows


def _value_bin_centers() -> list[float]:
    return [round(float(x), 5) for x in value_bins().cpu().numpy()]


# ---------------------------------------------------------------------------
# Fresh CPU MCTS search
# ---------------------------------------------------------------------------


@torch.no_grad()
def search_position(
    loaded: LoadedModel,
    action_ids: Sequence[int],
    *,
    visits: int = 512,
    c_puct: float = 1.5,
    n: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a fresh CPU MCTS search on the position and return the visit policy,
    root prior, and root value. Deterministic (no Dirichlet root noise)."""

    radius = int(n if n is not None else loaded.candidate_radius)
    state = state_from_actions(action_ids)
    inference = HexgtInference(loaded.model, device="cpu", fp16=False)
    session = new_mcts_session(n=radius)
    # A DEBUG search must be a clean, reproducible read of the model's own
    # judgement: no root exploration noise, neutral root-policy temperature, fixed
    # seed. So the reported root prior is exactly the network prior (matching the
    # analyze view) and re-running the same position yields the same tree.
    result = session.run(
        [0],
        [state],
        inference,
        visits=int(visits),
        c_puct=float(c_puct),
        temperature=1.0,
        seed=int(seed),
        # noise_fraction=0 disables root noise (alpha is then unused but the API
        # requires both); neutral root-policy temperature keeps the prior raw.
        root_dirichlet_total_alpha=1.0,
        root_dirichlet_noise_fraction=0.0,
        root_policy_temperature=1.0,
    )[0]

    visit_rows = _policy_pairs_to_rows(result.visit_policy, normalize=True)
    prior_rows = _policy_pairs_to_rows(result.root_prior_policy, normalize=False)
    return {
        "visits_requested": int(visits),
        "visits": int(result.visits),
        "root_value": float(result.root_value),
        "best_action_id": int(result.action_id),
        "best": _coord_of(int(result.action_id)),
        "visit_policy": visit_rows,
        "root_prior": prior_rows,
    }


def _policy_pairs_to_rows(pairs: Sequence[tuple[int, float]], *, normalize: bool) -> list[dict[str, Any]]:
    items = [(int(a), float(w)) for a, w in pairs]
    total = sum(w for _, w in items) if normalize else 0.0
    rows = []
    for aid, w in items:
        coord = _coord_of(aid)
        p = (w / total) if (normalize and total > 0) else w
        rows.append({"action_id": aid, "q": coord["q"], "r": coord["r"], "p": round(float(p), 6), "w": round(float(w), 4)})
    rows.sort(key=lambda r: r["w"], reverse=True)
    return rows
