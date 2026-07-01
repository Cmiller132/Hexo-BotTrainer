"""main_6 Gumbel S6[10]: gated end-to-end smoke + transition gates.

A torch-free smoke that drives the real Rust search (`HexfieldMctsSession.search`)
with the four Gumbel divergence flags ON (the `Divergences::gumbel()` profile,
expressed via `divergence_overrides`) over a handful of real decision states,
using a stub evaluator that emits the raw `priors_logits_bytes` column (Gumbel
S2) when the search requests it.

Asserts: no panic; Gumbel-Top-k root + Sequential-Halving + #2 non-root select +
#3 target export run end-to-end; and the exported improved-policy target
(`gumbel_policy_weights_bytes`) is a normalized distribution over its support.

Also pins the config-load + strict-key guard (S6[9]).
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from hexfield_testkit import api, sample_decision_states

try:
    from hexfield import _rust as hexfield_rust
except ImportError:  # pragma: no cover
    hexfield_rust = None

needs_rust = pytest.mark.skipif(
    hexfield_rust is None, reason="hexfield native module not built"
)


class GumbelStub:
    """Minimal dense-equivalent stub evaluator.

    Emits a normalized prior per legal action (deterministic, mildly peaked so
    the Gumbel-Top-k draw and σ have non-trivial structure) plus, when the
    search sets ``request_logits`` (Gumbel S2), the RAW pre-softmax logits in
    the SAME positional layout as ``priors_bytes`` (``log(prior)`` recovers a
    consistent logit set since softmax(log p) == p)."""

    def __call__(self, payload: dict) -> dict:
        b, _total = payload["shape"]
        legal_counts = np.frombuffer(payload["legal_counts"], dtype=np.int32)
        values: list[float] = []
        priors: list[float] = []
        logits: list[float] = []
        for g in range(b):
            l = int(legal_counts[g])
            # Deterministic, descending, peaked prior over this row's legal set.
            raw = np.array([1.0 / (1 + i) for i in range(l)], dtype=np.float64)
            p = raw / raw.sum()
            priors.extend(float(x) for x in p)
            # Raw logits == log(prior) (any affine shift is softmax-invariant; the
            # search stores them RAW and only the relative structure matters).
            logits.extend(float(math.log(x)) for x in p)
            # A small non-zero value so completedQ / σ are exercised.
            values.append(0.15 if (g % 2 == 0) else -0.1)
        reply = {
            "values_bytes": struct.pack(f"<{b}f", *values),
            "priors_bytes": struct.pack(f"<{len(priors)}f", *priors),
        }
        if payload.get("request_moves_left"):
            reply["moves_left_bytes"] = struct.pack(f"<{b}f", *([60.0] * b))
        if payload.get("request_logits"):
            reply["priors_logits_bytes"] = struct.pack(f"<{len(logits)}f", *logits)
        return reply


def _gumbel_overrides() -> dict:
    """The Divergences::gumbel() bool set, plus the canonical σ/candidate
    scalars, as a divergence_overrides dict (mirrors build_divergence_overrides
    with the gumbel knobs enabled)."""
    return {
        "gumbel_target": True,
        "gumbel_root": True,
        "gumbel_sequential_halving": True,
        "gumbel_nonroot_select": True,
        "gumbel_c_visit": 50.0,
        "gumbel_c_scale": 1.0,
        "gumbel_m": 8,
        "gumbel_target_min_visits": 1,
    }


@needs_rust
def test_gumbel_profile_smoke_runs_and_exports_normalized_target() -> None:
    states = sample_decision_states(range(40), (3, 4, 5, 6, 7, 8))
    assert len(states) >= 4, "need a few decision states for the smoke"
    states = states[:6]

    session = hexfield_rust.HexfieldMctsSession(max_states=65536)
    stub = GumbelStub()
    overrides = _gumbel_overrides()

    produced_target = 0
    for index, state in enumerate(states):
        key = 50_000 + index
        results = session.search(
            [key],
            (state,),
            evaluator=stub,
            visits=64,
            c_puct=1.5,
            temperature=1.0,
            seed=1234 + index * 7919,
            virtual_batch_size=8,
            fpu_reduction=0.2,
            virtual_loss=1.0,
            widening_policy_mass=0.95,
            widening_max_children=96,
            widening_min_children=2,
            forced_playout_k=0.0,
            root_policy_temperature=1.0,
            tss_enabled=False,
            divergence_overrides=overrides,
        )
        assert len(results) == 1
        r = results[0]
        # The search produced a legal played move without panicking.
        assert isinstance(r["action_id"], int)
        assert r["visits"] > 0

        # The #3 improved-policy target column must be present (gumbel_target on).
        assert "gumbel_policy_weights_bytes" in r, "gumbel target column missing"
        assert "gumbel_policy_action_ids_bytes" in r
        assert "root_prior_logits_bytes" in r
        weights = np.frombuffer(
            bytes(r["gumbel_policy_weights_bytes"]), dtype=np.float32
        )
        ids = np.frombuffer(
            bytes(r["gumbel_policy_action_ids_bytes"]), dtype=np.uint32
        )
        assert r["gumbel_policy_count"] == len(ids) == len(weights)
        if len(weights) > 0:
            produced_target += 1
            # Normalized improved-policy target over its support.
            assert np.all(np.isfinite(weights)), "target weights must be finite"
            assert np.all(weights >= -1e-6), "target weights must be non-negative"
            assert abs(float(weights.sum()) - 1.0) < 1e-4, (
                f"gumbel target must sum to 1, got {float(weights.sum())}"
            )
            # Support floor (min_visits=1): every exported action was searched.
            assert len(set(ids.tolist())) == len(ids), "duplicate target action ids"
        session.discard(key)

    assert produced_target >= 1, "no Full root produced a Gumbel target across the smoke"


@needs_rust
def test_gumbel_continuous_reuse_rebuilds_sh_state_per_move() -> None:
    """Regression: run_continuous with the Gumbel flags on must re-run a full
    Gumbel-Top-k + SH search on every Full move, including reused (promoted)
    roots after an ('advance', state) response.

    The bug this guards against: the advance/keep_promoted path did not rebuild
    the Gumbel root state, so the previous move's finished SH schedule (stale
    survivors + met round caps) persisted onto the new root. The slot then made
    no root progress and the force-stuck safety net finalized the move with
    ZERO net visits over the reuse baseline (payload['visits'] == 0)."""
    from hexo_engine import api as engine_api
    from hexo_engine.types import AxialCoord, PlacementAction

    from hexfield.geometry import unpack_action_id

    budget = 96
    max_plies = 10

    class _Driver:
        def __init__(self) -> None:
            self.states: dict = {}
            self.plies: dict = {}
            self.rows: list = []

        def start(self, key: int):
            self.states[key] = engine_api.new_game()
            self.plies[key] = 0
            return self.states[key]

        def __call__(self, game_key: int, payload: dict):
            ply = self.plies[game_key]
            self.rows.append(
                (ply, bool(payload.get("pcr_full")), int(payload["visits"]))
            )
            q, r = unpack_action_id(payload["action_id"])
            state = self.states[game_key]
            result = engine_api.apply_action(
                state, PlacementAction(AxialCoord(q=q, r=r))
            )
            self.plies[game_key] = ply + 1
            if result.terminal or self.plies[game_key] >= max_plies:
                del self.states[game_key]
                return None
            return ("advance", state)

    session = hexfield_rust.HexfieldMctsSession(max_states=65536)
    driver = _Driver()
    keys = [91_000, 91_001]
    states = tuple(driver.start(k) for k in keys)
    session.run_continuous(
        keys,
        states,
        evaluator=GumbelStub(),
        on_move=driver,
        visits=budget,
        c_puct=1.5,
        base_seed=424242,
        virtual_batch_size=8,
        flush_target=64,
        active_root_limit=len(keys),
        temperature_by_ply=[1.0] * 64,
        root_policy_temperature=1.0,
        fpu_reduction=0.2,
        virtual_loss=1.0,
        widening_policy_mass=0.95,
        widening_max_children=96,
        widening_min_children=2,
        forced_playout_k=0.0,
        pcr_full_proportion=1.0,  # every move Full: each must search
        pcr_fast_visits=32,
        policy_init_fraction=0.0,
        policy_init_avg_plies=0.0,
        policy_init_max_plies=0,
        policy_init_temperature=1.0,
        tss_enabled=False,
        root_fpu_reduction=0.2,
        root_fpu_zero_under_noise=False,
        search_parity_mode=False,
        divergence_overrides=_gumbel_overrides(),
    )

    full_rows = [(ply, visits) for ply, full, visits in driver.rows if full]
    assert len(full_rows) >= 2 * (max_plies - 1), "expected full games of Full moves"
    reused = [(ply, visits) for ply, visits in full_rows if ply >= 1]
    assert reused, "no reused-root moves decided"
    zero_visit = [row for row in reused if row[1] == 0]
    assert not zero_visit, (
        f"reused-root Full moves finalized with zero net visits: {zero_visit}"
    )
    mean_visits = sum(v for _, v in reused) / len(reused)
    assert mean_visits >= budget * 0.5, (
        f"reused-root Full moves under-searched: mean {mean_visits:.1f} of {budget}"
    )


@needs_rust
def test_gumbel_flags_off_omits_target_column() -> None:
    """Contract (§1): with the Gumbel bools OFF (parity/production), the export
    omits the new keys entirely — byte-identical payload schema to today."""
    states = sample_decision_states(range(40), (3, 4, 5, 6, 7, 8))[:3]
    session = hexfield_rust.HexfieldMctsSession(max_states=65536)
    stub = GumbelStub()
    for index, state in enumerate(states):
        key = 60_000 + index
        results = session.search(
            [key],
            (state,),
            evaluator=stub,
            visits=48,
            c_puct=1.5,
            temperature=1.0,
            seed=777 + index,
            virtual_batch_size=8,
            fpu_reduction=0.2,
            virtual_loss=1.0,
            widening_policy_mass=0.95,
            widening_max_children=96,
            widening_min_children=2,
            forced_playout_k=0.0,
            root_policy_temperature=1.0,
            tss_enabled=False,
            # No divergence_overrides ⇒ production defaults ⇒ all gumbel bools off.
        )
        r = results[0]
        assert "gumbel_policy_weights_bytes" not in r
        assert "gumbel_policy_action_ids_bytes" not in r
        assert "root_prior_logits_bytes" not in r
        session.discard(key)


def test_main6_config_loads_with_gumbel_on() -> None:
    """S6[9]: hexfield_main_6.toml parses; main_6 OPTS IN to all four Gumbel
    mechanisms (the mandate: main_6 is the full-Gumbel run, opts in via config);
    σ / candidate scalars at canonical defaults; policy_target='gumbel'."""
    import tomllib
    from pathlib import Path

    from hexfield.config import parse_hexfield_config

    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "hexfield_main_6.toml", "rb") as f:
        raw = tomllib.load(f)
    cfg = parse_hexfield_config(raw["model"]["config"])
    sp = cfg.selfplay
    assert sp.gumbel_target_enabled is True
    assert sp.gumbel_root_enabled is True
    assert sp.gumbel_sequential_halving is True
    assert sp.gumbel_nonroot_select is True
    assert sp.gumbel_c_visit == 50.0
    assert sp.gumbel_c_scale == 1.0
    assert sp.gumbel_m == 32
    assert sp.gumbel_target_min_visits == 1
    assert cfg.training.policy_target == "gumbel"


def test_misplaced_policy_target_raises() -> None:
    """S6[9] guard: policy_target belongs under [model.config.training]; placing
    it under [model.config.selfplay] must raise ValueError at load."""
    from hexfield.config import parse_hexfield_config

    with pytest.raises(ValueError):
        parse_hexfield_config({"selfplay": {"policy_target": "gumbel"}})


def test_build_divergence_overrides_emits_gumbel_knobs() -> None:
    """S1 contract: the overrides dict carries the four gumbel bools + the σ /
    candidate scalars as concrete bool/float/int (never None)."""
    from hexfield.config import SelfplayConfig, build_divergence_overrides

    sp = SelfplayConfig(
        gumbel_target_enabled=True,
        gumbel_root_enabled=True,
        gumbel_sequential_halving=True,
        gumbel_nonroot_select=True,
    )
    ov = build_divergence_overrides(sp)
    assert ov["gumbel_target"] is True
    assert ov["gumbel_root"] is True
    assert ov["gumbel_sequential_halving"] is True
    assert ov["gumbel_nonroot_select"] is True
    assert ov["gumbel_c_visit"] == pytest.approx(50.0)
    assert ov["gumbel_c_scale"] == pytest.approx(1.0)
    assert isinstance(ov["gumbel_m"], int) and ov["gumbel_m"] == 16
    assert isinstance(ov["gumbel_target_min_visits"], int)
    for k, v in ov.items():
        assert v is not None, k
        assert isinstance(v, (bool, float, int)), (k, type(v))
