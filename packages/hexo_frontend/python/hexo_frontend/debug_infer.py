"""CPU-only, lineage-aware inference for the dashboard Debug tab.

This module is the *inference library* behind the Debug tab. It loads a training
checkpoint, reconstructs a board position from a move sequence, and returns what
the model "thinks" — per-candidate policy prior, the distributional value head
(+ scalar), auxiliary heads (opponent-policy / short-term-value / moves-left),
and (on demand) a fresh CPU MCTS visit distribution.

It is **lineage-aware**. The dashboard serves several model lineages and their
checkpoints have different payload shapes and architectures:

* ``hexgt`` (and graph variants) — payload carries ``ck["model"]`` (a state
  dict) and ``ck["arch"]`` (architecture metadata); the network is the graph
  ``HexgtNetwork`` and the position is featurized as a candidate graph.
* ``dense_cnn_restnet`` — payload carries ``ck["model"]`` == the string
  ``"dense_cnn_restnet"`` plus ``ck["model_state"]``; the network is the ResTNet
  CNN trunk and the position is featurized as the 13-plane 41x41 disk crop.
* ``dense_cnn`` (plain) — payload ``ck["model"]`` == ``"hexo_models.dense_cnn"``
  plus ``ck["model_state"]``; a conv-only CNN over the same 41x41 crop.

The lineage is detected from the checkpoint payload (see ``_detect_lineage``),
the correct model is built, the correct featurizer/encoder is used, and the SAME
debug-output schema is returned for every lineage (heads that a lineage does not
have are returned as ``None`` / omitted, so the UI marks them N/A).

Everything here is **CPU-only by construction**: models are built and run on
``torch.device("cpu")`` and the MCTS evaluator is constructed with ``device=
"cpu"`` (no AMP). The worker process that imports this module is also launched
with ``CUDA_VISIBLE_DEVICES=""`` so it can never touch the training GPU.

Heavy per-lineage imports are lazy (loaded only when a checkpoint of that lineage
is opened) so that a venv missing one lineage's package can still debug the
others, and the worker's ``ping`` never pays an import it does not need.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id

# STV horizons the hexgt architecture always carries. The driver forces these on
# at resume; we mirror that so every loaded hexgt checkpoint exposes the STV
# heads. (Dense lineages read their own horizons from the checkpoint instead.)
STV_HORIZONS: tuple[int, ...] = (4, 12, 24)

# Lineage tags.
HEXGT = "hexgt"
DENSE_RESTNET = "dense_cnn_restnet"
DENSE_PLAIN = "dense_cnn"


# ---------------------------------------------------------------------------
# Loaded-model container + lineage detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedModel:
    """A CPU model ready for debug inference, plus provenance metadata.

    The hexgt-specific fields (``graft``/``expanded_*``/``zeroed_feature_cols``/
    ``candidate_radius``) are kept for the graph lineage and left at neutral
    defaults for the dense lineages so the worker's metadata view is uniform.
    """

    lineage: str
    model: Any
    arch: dict[str, Any]
    rl_epoch: int | None
    step: int | None
    candidate_radius: int | None = None
    graft: str | None = None  # hexgt: "pre" | "post"
    expanded_value: bool = False
    expanded_stv: list[str] = field(default_factory=list)
    zeroed_feature_cols: list[int] = field(default_factory=list)
    load_warnings: list[str] = field(default_factory=list)
    stv_horizons: tuple[int, ...] = ()
    has_moves_left: bool = False


def _detect_lineage(payload: Any) -> str:
    """Classify a checkpoint payload into a model lineage.

    Dense checkpoints tag themselves with a string ``payload["model"]`` (the
    model/plugin name) and store weights under ``payload["model_state"]``. The
    hexgt graph lineage instead stores the state dict directly under
    ``payload["model"]`` alongside an ``payload["arch"]`` dict.
    """

    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a dict")
    tag = payload.get("model")
    if isinstance(tag, str):
        lowered = tag.lower()
        if "restnet" in lowered:
            return DENSE_RESTNET
        if "dense_cnn" in lowered:
            return DENSE_PLAIN
        if "model_state" in payload:
            # Unknown dense-family tag, but it carries a separate state dict: the
            # ResTNet builder is the superset CNN, so default to it.
            return DENSE_RESTNET
        raise ValueError(f"unrecognized checkpoint model tag {tag!r}")
    # Graph lineage (hexgt / hexgnn): state dict + arch metadata.
    if "model" in payload and "arch" in payload:
        return HEXGT
    raise ValueError(
        "checkpoint has no recognizable lineage "
        "(expected a dense 'model' string tag + 'model_state', or hexgt 'model'+'arch')"
    )


def load_checkpoint(path: str | Path) -> LoadedModel:
    """Load a checkpoint onto CPU, dispatching on its detected lineage."""

    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lineage = _detect_lineage(payload)
    if lineage in (DENSE_RESTNET, DENSE_PLAIN):
        return _load_dense_checkpoint(ckpt_path, payload, lineage)
    return _load_hexgt_checkpoint(ckpt_path, payload)


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Shared position reconstruction
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


# ---------------------------------------------------------------------------
# Top-level dispatch (called by the worker)
# ---------------------------------------------------------------------------


@torch.no_grad()
def analyze_position(loaded: LoadedModel, action_ids: Sequence[int], *, n: int | None = None) -> dict[str, Any]:
    """Full-head readout for the position reached by ``action_ids``."""

    if loaded.lineage in (DENSE_RESTNET, DENSE_PLAIN):
        return _analyze_dense(loaded, action_ids)
    return _analyze_hexgt(loaded, action_ids, n=n)


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
    """Run a fresh, reproducible CPU MCTS on the position (no root noise)."""

    if loaded.lineage in (DENSE_RESTNET, DENSE_PLAIN):
        return _search_dense(loaded, action_ids, visits=visits, c_puct=c_puct, seed=seed)
    return _search_hexgt(loaded, action_ids, visits=visits, c_puct=c_puct, n=n, seed=seed)


# ===========================================================================
# Dense lineages (dense_cnn_restnet, dense_cnn) — 13-plane 41x41 disk crop
# ===========================================================================


@lru_cache(maxsize=None)
def _dense_pkg(lineage: str) -> SimpleNamespace:
    """Lazily import the package modules backing a dense lineage."""

    if lineage == DENSE_RESTNET:
        from dense_cnn_restnet import architecture as arch_mod
        from dense_cnn_restnet import inference as inf_mod
        from dense_cnn_restnet import losses as losses_mod
        from dense_cnn_restnet import mcts as mcts_mod
        from dense_cnn_restnet import rust_bridge as rb_mod

        net_cls = arch_mod.RestnetNetwork
    elif lineage == DENSE_PLAIN:
        from hexo_models.dense_cnn import architecture as arch_mod
        from hexo_models.dense_cnn import inference as inf_mod
        from hexo_models.dense_cnn import losses as losses_mod
        from hexo_models.dense_cnn import mcts as mcts_mod
        from hexo_models.dense_cnn import rust_bridge as rb_mod

        net_cls = arch_mod.Model1Network
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"not a dense lineage: {lineage!r}")
    return SimpleNamespace(
        arch=arch_mod,
        inference=inf_mod,
        losses=losses_mod,
        mcts=mcts_mod,
        rust_bridge=rb_mod,
        net_cls=net_cls,
    )


def _read_manifest_arch(ckpt_path: Path) -> dict[str, Any]:
    """Best-effort read of the run's architecture from ``manifest.json``.

    Dense checkpoints do not embed the architecture, so structural fields are
    inferred from the state dict (always consistent with the weights). A few
    behavioral flags that are not derivable from weights (residual_conv,
    attention_scope/impl) come from the run manifest when present, else from the
    package defaults. ``manifest.json`` lives at the run root (the checkpoint's
    grandparent dir: ``<run>/checkpoints/<file>.pt``)."""

    try:
        manifest = ckpt_path.resolve().parent.parent / "manifest.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            arch = data.get("model", {}).get("config", {}).get("architecture", {})
            if isinstance(arch, dict):
                return arch
    except Exception:
        pass
    return {}


def _infer_dense_arch(lineage: str, state_dict: dict[str, Any], manifest_arch: dict[str, Any]) -> dict[str, Any]:
    """Infer constructor kwargs for a dense network from its state dict.

    Structure (channels, block layout, head set) is read from the weights so it
    always matches; non-derivable behavioral flags fall back to the manifest then
    package defaults."""

    horizons = tuple(
        sorted(
            {
                int(key.split(".")[1])
                for key in state_dict
                if key.startswith("short_term_value_heads.")
            }
        )
    )

    if lineage == DENSE_RESTNET:
        stem = state_dict["stem_conv.weight"]
        channels = int(stem.shape[0])
        in_channels = int(stem.shape[1])
        embed_kernel_size = int(stem.shape[2])
        n_blocks = max(int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")) + 1
        kinds = [
            ("T" if any(k.startswith(f"blocks.{i}.attn.") for k in state_dict) else "R")
            for i in range(n_blocks)
        ]
        blocks_type = "_".join(kinds)
        attention_heads = 4
        for key, value in state_dict.items():
            if key.endswith("attn.relative_bias_table"):
                attention_heads = int(value.shape[1])
                break
        mlp_ratio = 2
        for key, value in state_dict.items():
            if key.endswith(".mlp.0.weight"):
                mlp_ratio = max(1, int(value.shape[0]) // channels)
                break
        return {
            "in_channels": in_channels,
            "channels": channels,
            "blocks_type": blocks_type,
            "attention_heads": attention_heads,
            "mlp_ratio": mlp_ratio,
            "embed_kernel_size": embed_kernel_size,
            "residual_conv": str(manifest_arch.get("residual_conv", "hex")),
            "attention_impl": str(manifest_arch.get("attention_impl", "sdpa")),
            "attention_scope": str(manifest_arch.get("attention_scope", "disk")),
            "short_term_value_horizons": horizons,
            "moves_left_head": "moves_left_head.weight" in state_dict,
        }

    # Plain dense_cnn.
    conv = state_dict["conv_in.weight"]
    channels = int(conv.shape[0])
    in_channels = int(conv.shape[1])
    n_blocks = max(int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")) + 1
    return {
        "in_channels": in_channels,
        "channels": channels,
        "blocks": n_blocks,
        "short_term_value_horizons": horizons,
        "moves_left_head": False,
    }


def _build_dense_model(pkg: SimpleNamespace, lineage: str, arch: dict[str, Any]) -> torch.nn.Module:
    if lineage == DENSE_RESTNET:
        return pkg.net_cls(
            in_channels=arch["in_channels"],
            channels=arch["channels"],
            blocks_type=arch["blocks_type"],
            attention_heads=arch["attention_heads"],
            mlp_ratio=arch["mlp_ratio"],
            embed_kernel_size=arch["embed_kernel_size"],
            residual_conv=arch["residual_conv"],
            attention_impl=arch["attention_impl"],
            attention_scope=arch["attention_scope"],
            short_term_value_horizons=arch["short_term_value_horizons"],
            moves_left_head=arch["moves_left_head"],
        )
    return pkg.net_cls(
        in_channels=arch["in_channels"],
        channels=arch["channels"],
        blocks=arch["blocks"],
        short_term_value_horizons=arch["short_term_value_horizons"],
    )


def _load_dense_checkpoint(ckpt_path: Path, payload: dict[str, Any], lineage: str) -> LoadedModel:
    pkg = _dense_pkg(lineage)
    state_dict = payload.get("model_state")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{ckpt_path.name}: dense checkpoint missing 'model_state'")

    arch = _infer_dense_arch(lineage, state_dict, _read_manifest_arch(ckpt_path))
    model = _build_dense_model(pkg, lineage, arch)

    warnings: list[str] = []
    # Strict load (the inferred arch is derived from the weights, so it should be
    # exact). If a benign head drift makes strict load fail, fall back to a
    # non-strict load and surface the drift in the UI rather than 500-ing.
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            warnings.append(f"missing keys: {list(result.missing_keys)[:8]}")
        if result.unexpected_keys:
            warnings.append(f"unexpected keys: {list(result.unexpected_keys)[:8]}")
        if not result.missing_keys and not result.unexpected_keys:
            warnings.append(f"load mismatch: {exc}")

    model.eval()
    return LoadedModel(
        lineage=lineage,
        model=model,
        arch=arch,
        rl_epoch=_maybe_int(payload.get("epoch")),
        step=_maybe_int(payload.get("step")),
        candidate_radius=None,
        graft=None,
        load_warnings=warnings,
        stv_horizons=tuple(int(h) for h in arch["short_term_value_horizons"]),
        has_moves_left=bool(arch.get("moves_left_head", False)),
    )


def _dense_inputs(pkg: SimpleNamespace, state: Any) -> tuple[torch.Tensor, list[int], list[int]]:
    """Encode one engine state into the dense 41x41 crop via the Rust accelerator.

    Returns the input planes ``(1, C, 41, 41)`` plus the per-row legal action ids
    and their flat crop indices (used to softmax the policy over the legal set,
    exactly as production inference does)."""

    payload = pkg.rust_bridge.model1_batch_inputs([state])
    shape = tuple(int(x) for x in payload["shape"])
    # The Rust buffer is read-only; copy into a writable tensor before reshape.
    inputs = torch.frombuffer(bytearray(payload["inputs"]), dtype=torch.float32).reshape(shape)
    legal_action_ids = [int(a) for a in payload["legal_action_ids"][0]]
    legal_flat = [int(x) for x in payload["legal_flat_indices"][0]]
    return inputs, legal_action_ids, legal_flat


def _dense_legal_rows(logits: torch.Tensor, legal_action_ids: list[int], legal_flat: list[int]) -> list[dict[str, Any]]:
    """Softmax a flat crop policy over the legal cells -> per-candidate rows."""

    if not legal_action_ids:
        return []
    idx = torch.as_tensor(legal_flat, dtype=torch.long, device=logits.device)
    priors = torch.softmax(logits.index_select(0, idx), dim=0).cpu().numpy()
    rows = []
    for aid, prob in zip(legal_action_ids, priors.tolist()):
        coord = _coord_of(aid)
        rows.append({"action_id": int(aid), "q": coord["q"], "r": coord["r"], "p": round(float(prob), 6)})
    rows.sort(key=lambda r: r["p"], reverse=True)
    return rows


def _decode_dist(losses_mod: Any, logits: torch.Tensor) -> dict[str, Any]:
    """Scalar + 65-bin distribution for one value-style head's logits."""

    flat = logits.float().reshape(-1)
    scalar = float(losses_mod.decode_binned_value(flat.reshape(1, -1)).reshape(()).item())
    dist = [round(float(x), 5) for x in torch.softmax(flat, dim=0).cpu().numpy()]
    return {"scalar": scalar, "dist": dist}


@torch.no_grad()
def _analyze_dense(loaded: LoadedModel, action_ids: Sequence[int]) -> dict[str, Any]:
    pkg = _dense_pkg(loaded.lineage)
    state = state_from_actions(action_ids)
    inputs, legal_action_ids, legal_flat = _dense_inputs(pkg, state)
    out = loaded.model.forward(inputs)

    policy = _dense_legal_rows(out["policy"][0].float(), legal_action_ids, legal_flat)
    opp = None
    if "opp_policy" in out:
        opp = _dense_legal_rows(out["opp_policy"][0].float(), legal_action_ids, legal_flat)

    value = _decode_dist(pkg.losses, out["value"][0])

    stv: dict[str, Any] = {}
    for horizon in loaded.stv_horizons:
        key = f"stvalue_{horizon}"
        if key in out:
            stv[str(horizon)] = _decode_dist(pkg.losses, out[key][0])

    moves_left = _decode_dist(pkg.losses, out["moves_left"][0]) if "moves_left" in out else None

    current = engine.current_player(state)
    current_role = getattr(current, "value", str(current))
    current_index = 1 if str(current_role).endswith("1") else 0

    return {
        "current_player": current_index,
        "current_role": str(current_role),
        "candidate_count": len(legal_action_ids),
        "legal_count": int(engine.legal_action_count(state)),
        "value": value["scalar"],
        # The owner-swap "optimism" probe is a hexgt-graph-specific calibration
        # check; the dense crop encodes side-to-move ownership in Rust, so it is
        # marked N/A here (UI hides the panel when value_swapped is null).
        "value_swapped": None,
        "optimism": None,
        "value_bins": [round(float(x), 5) for x in pkg.losses.value_bins().cpu().numpy()],
        "value_dist": value["dist"],
        "policy": policy,
        "opp_policy": opp,
        "stvalue": stv,
        "moves_left": moves_left,
    }


@torch.no_grad()
def _search_dense(
    loaded: LoadedModel,
    action_ids: Sequence[int],
    *,
    visits: int,
    c_puct: float,
    seed: int,
) -> dict[str, Any]:
    pkg = _dense_pkg(loaded.lineage)
    state = state_from_actions(action_ids)
    inference = pkg.inference.DenseCNNInference(
        loaded.model, device="cpu", amp=False, optimize_for_inference=False
    )
    session = pkg.mcts.new_mcts_session()
    # A DEBUG search must be a clean, reproducible read of the model's own
    # judgement: no root exploration noise, neutral root-policy temperature, fixed
    # seed. So the reported root prior is exactly the network prior and re-running
    # the same position yields the same tree.
    result = session.run(
        [0],
        [state],
        inference,
        visits=int(visits),
        c_puct=float(c_puct),
        temperature=1.0,
        seed=int(seed),
        root_dirichlet_total_alpha=1.0,
        root_dirichlet_noise_fraction=0.0,
        root_policy_temperature=1.0,
    )[0]

    return {
        "visits_requested": int(visits),
        "visits": int(result.visits),
        "root_value": float(result.root_value),
        "best_action_id": int(result.action_id),
        "best": _coord_of(int(result.action_id)),
        "visit_policy": _policy_pairs_to_rows(result.visit_policy, normalize=True),
        "root_prior": _policy_pairs_to_rows(result.root_prior_policy, normalize=False),
    }


# ===========================================================================
# hexgt graph lineage — candidate-graph featurizer (pre/post-STV-graft aware)
# ===========================================================================


@lru_cache(maxsize=1)
def _hexgt() -> SimpleNamespace:
    """Lazily import the hexgt graph-model modules."""

    from hexo_models.hexgt.architecture import (
        HexgtNetwork,
        expand_stv_readout_columns,
        expand_value_readout_columns,
        zero_init_expanded_feature_columns,
    )
    from hexo_models.hexgt.collate import collate_graphs
    from hexo_models.hexgt.constants import (
        DEFAULT_CANDIDATE_RADIUS,
        FEATURE_SCHEMA_VERSION,
        feature_slots_after,
    )
    from hexo_models.hexgt.features import build_graph_tensors
    from hexo_models.hexgt.inference import HexgtInference
    from hexo_models.hexgt.losses import decode_binned_value, value_bins
    from hexo_models.hexgt.mcts import new_mcts_session
    from hexo_models.hexgt import rust_bridge

    return SimpleNamespace(
        HexgtNetwork=HexgtNetwork,
        expand_stv_readout_columns=expand_stv_readout_columns,
        expand_value_readout_columns=expand_value_readout_columns,
        zero_init_expanded_feature_columns=zero_init_expanded_feature_columns,
        collate_graphs=collate_graphs,
        DEFAULT_CANDIDATE_RADIUS=DEFAULT_CANDIDATE_RADIUS,
        FEATURE_SCHEMA_VERSION=FEATURE_SCHEMA_VERSION,
        feature_slots_after=feature_slots_after,
        build_graph_tensors=build_graph_tensors,
        HexgtInference=HexgtInference,
        decode_binned_value=decode_binned_value,
        value_bins=value_bins,
        new_mcts_session=new_mcts_session,
        rust_bridge=rust_bridge,
    )


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


def _load_hexgt_checkpoint(ckpt_path: Path, payload: dict[str, Any]) -> LoadedModel:
    """Load a hexgt checkpoint onto CPU, handling pre/post-graft shape drift."""

    hx = _hexgt()
    arch = dict(payload["arch"])
    state_dict = payload["model"]

    model = hx.HexgtNetwork(**_arch_kwargs(arch))

    # Detect the graft generation BEFORE expanding: a pre-graft (<=epoch 6) STV
    # head reads the SIDE hub only (width == token_dim); the epoch-7 graft widened
    # it to [SIDE|PMA]. (The value head was widened in an earlier phase, so the STV
    # head width is the signal for the epoch-7 boundary.)
    token_dim = model.token_dim
    stv_w = state_dict.get(f"short_term_value_heads.{STV_HORIZONS[0]}.0.weight")
    graft = "post"
    if stv_w is not None and tuple(stv_w.shape)[1] == token_dim and model.value_readout_blocks > 1:
        graft = "pre"

    expanded_value = hx.expand_value_readout_columns(model, state_dict)
    expanded_stv = hx.expand_stv_readout_columns(model, state_dict)

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
    if loaded_fsv < hx.FEATURE_SCHEMA_VERSION:
        zeroed = hx.zero_init_expanded_feature_columns(model, hx.feature_slots_after(loaded_fsv))

    model.eval()
    return LoadedModel(
        lineage=HEXGT,
        model=model,
        arch=arch,
        rl_epoch=_maybe_int(payload.get("rl_epoch")),
        step=_maybe_int(payload.get("step")),
        candidate_radius=int(arch.get("candidate_radius", hx.DEFAULT_CANDIDATE_RADIUS)),
        graft=graft,
        expanded_value=expanded_value,
        expanded_stv=expanded_stv,
        zeroed_feature_cols=zeroed,
        load_warnings=warnings,
        stv_horizons=STV_HORIZONS,
        has_moves_left=False,
    )


def _swap_owner(facts: dict[str, Any]) -> dict[str, Any]:
    """Return facts with stone ownership flipped (0<->1), leaving empty/SIDE
    untouched. ``build_graph_tensors`` re-derives every feature from this — so
    the swapped graph is the IDENTICAL board seen from the opponent's side. This
    matches the optimism/calibration probe in ``_optimism_main3.py`` exactly."""

    owners = list(facts["nodes"]["node_owner"])
    swapped = [1 if o == 0 else (0 if o == 1 else o) for o in owners]
    return {**facts, "nodes": {**facts["nodes"], "node_owner": swapped}}


@torch.no_grad()
def _analyze_hexgt(loaded: LoadedModel, action_ids: Sequence[int], *, n: int | None = None) -> dict[str, Any]:
    hx = _hexgt()
    radius = int(n if n is not None else loaded.candidate_radius)
    state = state_from_actions(action_ids)
    model = loaded.model
    facts = hx.rust_bridge.graph_facts(state, radius)
    batch = hx.collate_graphs([hx.build_graph_tensors(facts)])
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
    value_scalar = float(hx.decode_binned_value(value_logits.reshape(1, -1)).reshape(()).item())

    stv: dict[str, Any] = {}
    for horizon in STV_HORIZONS:
        key = f"stvalue_{horizon}"
        if key in out:
            logits = out[key].float().reshape(-1)
            stv[str(horizon)] = {
                "scalar": float(hx.decode_binned_value(logits.reshape(1, -1)).reshape(()).item()),
                "dist": [round(float(x), 5) for x in torch.softmax(logits, dim=0).cpu().numpy()],
            }

    # Both-perspectives / optimism probe: evaluate the IDENTICAL board with stone
    # ownership flipped (0<->1). hexgt value is from the side-to-move's view, so a
    # well-calibrated zero-sum model gives v_self + v_swapped ~= 0; a positive sum
    # is "optimism" (both sides think they are winning).
    swapped_batch = hx.collate_graphs([hx.build_graph_tensors(_swap_owner(facts))])
    swapped_logits = model.forward_policy_value(swapped_batch)["value"].float().reshape(-1)
    value_swapped = float(hx.decode_binned_value(swapped_logits.reshape(1, -1)).reshape(()).item())

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
        "value_bins": _value_bin_centers(hx),
        "value_dist": [round(float(x), 5) for x in value_dist],
        "policy": candidates,
        "opp_policy": opp,
        "stvalue": stv,
        "moves_left": None,
    }


def _candidate_rows(candidate_ids: np.ndarray, priors: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for aid, prob in zip(candidate_ids.tolist(), priors.tolist()):
        coord = _coord_of(aid)
        rows.append({"action_id": int(aid), "q": coord["q"], "r": coord["r"], "p": round(float(prob), 6)})
    rows.sort(key=lambda r: r["p"], reverse=True)
    return rows


def _value_bin_centers(hx: SimpleNamespace) -> list[float]:
    return [round(float(x), 5) for x in hx.value_bins().cpu().numpy()]


@torch.no_grad()
def _search_hexgt(
    loaded: LoadedModel,
    action_ids: Sequence[int],
    *,
    visits: int,
    c_puct: float,
    n: int | None,
    seed: int,
) -> dict[str, Any]:
    hx = _hexgt()
    radius = int(n if n is not None else loaded.candidate_radius)
    state = state_from_actions(action_ids)
    inference = hx.HexgtInference(loaded.model, device="cpu", fp16=False)
    session = hx.new_mcts_session(n=radius)
    result = session.run(
        [0],
        [state],
        inference,
        visits=int(visits),
        c_puct=float(c_puct),
        temperature=1.0,
        seed=int(seed),
        root_dirichlet_total_alpha=1.0,
        root_dirichlet_noise_fraction=0.0,
        root_policy_temperature=1.0,
    )[0]

    return {
        "visits_requested": int(visits),
        "visits": int(result.visits),
        "root_value": float(result.root_value),
        "best_action_id": int(result.action_id),
        "best": _coord_of(int(result.action_id)),
        "visit_policy": _policy_pairs_to_rows(result.visit_policy, normalize=True),
        "root_prior": _policy_pairs_to_rows(result.root_prior_policy, normalize=False),
    }
