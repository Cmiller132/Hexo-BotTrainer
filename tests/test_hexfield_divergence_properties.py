"""M6 property gates for the §5.4 divergences.

- LCB vs closed-form on synthetic visit/Q/sigma tables, incl. eligibility
  fallback (None when no child qualifies).
- Moves-left utility: identically zero at/below the |Q| gate (no sign
  discontinuity), correct sign (shorter-when-winning preferred), monotone in
  (M_e - M_node), bounded by w.
- Early-stop with LCB lesioned off ≡ early-stop off: EXACT chosen-move
  equality over many greedy searches (pure visit-overtake is provably sound),
  and the stop actually fires (non-vacuous).
- Reused-root policy temperature is applied exactly once per root lifetime
  (the main1 reuse-bug regression, hexfield-side).
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from hexfield_testkit import api, sample_decision_states
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield.geometry import unpack_action_id
from test_hexfield_search_parity import HexfieldStub, _corpus

try:
    from hexfield import _rust as hexfield_rust
except ImportError:  # pragma: no cover
    hexfield_rust = None

needs_native = pytest.mark.skipif(hexfield_rust is None, reason="native module not built")


@needs_native
def test_lcb_matches_closed_form_tables() -> None:
    rng = random.Random(5)
    for _ in range(300):
        n_edges = rng.randint(1, 12)
        stats = []
        for i in range(n_edges):
            visits = rng.randint(0, 200)
            delta = rng.randint(0, visits) if visits else 0
            q = rng.uniform(-1, 1)
            value_sum = q * visits
            # sigma^2 in [0, 0.5]
            var = rng.uniform(0, 0.5)
            value_sq_sum = (var + q * q) * visits
            stats.append((1000 + i, delta, visits, value_sum, value_sq_sum))
        z, minv, frac = 1.6, 8, 0.1
        got = hexfield_rust.debug_lcb_pick(stats, z, minv, frac)

        max_delta = max((s[1] for s in stats), default=0)
        if max_delta == 0:
            assert got is None
            continue
        threshold = max(minv, frac * max_delta)
        best = None
        for aid, delta, visits, vsum, vsq in stats:
            if delta < threshold or visits == 0:
                continue
            q = np.float32(vsum) / np.float32(visits)
            var = max(np.float32(vsq) / np.float32(visits) - q * q, np.float32(0.0))
            lcb = float(q - np.float32(z) * math.sqrt(var) / math.sqrt(visits))
            if best is None or lcb > best[0] + 1e-9 or (abs(lcb - best[0]) <= 1e-9 and aid < best[1]):
                best = (lcb, aid)
        expected = best[1] if best else None
        assert got == expected, (stats, got, expected)


@needs_native
def test_lcb_eligibility_fallback() -> None:
    # All children below the visit threshold -> None (caller falls back).
    stats = [(1, 3, 3, 1.5, 1.0), (2, 2, 2, -0.5, 0.5)]
    assert hexfield_rust.debug_lcb_pick(stats, 1.6, 8, 0.1) is None


@needs_native
def test_ml_bonus_properties() -> None:
    w, scale, gate = 0.03, 32.0, 0.6
    bonus = lambda q, me, mn: hexfield_rust.debug_ml_bonus(q, me, mn, w, scale, gate)
    # Identically zero at and below the gate — no sign discontinuity exists.
    for q in (-1.0, -0.7, 0.0, 0.3, 0.6):
        assert bonus(q, 10.0, 50.0) == 0.0
        assert bonus(q, 50.0, 10.0) == 0.0
    # Winning + child predicts FEWER moves left -> positive bonus (prefer it).
    assert bonus(0.9, 20.0, 40.0) > 0.0
    # Winning + child predicts MORE moves left -> negative bonus.
    assert bonus(0.9, 60.0, 40.0) < 0.0
    # Monotone decreasing in (M_e - M_node); bounded by w.
    last = None
    for me in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0):
        b = bonus(0.9, me, 35.0)
        assert abs(b) <= w + 1e-9
        if last is not None:
            assert b < last
        last = b
    # Delta form: invariant to a shared absolute bias.
    assert bonus(0.9, 20.0, 40.0) == pytest.approx(bonus(0.9, 120.0, 140.0), abs=1e-7)


@needs_native
def test_ml_bonus_two_sided() -> None:
    w, scale, gate = 0.03, 32.0, 0.6
    one = lambda q, me, mn: hexfield_rust.debug_ml_bonus(q, me, mn, w, scale, gate, False)
    two = lambda q, me, mn: hexfield_rust.debug_ml_bonus(q, me, mn, w, scale, gate, True)
    # Dead-zone is identical either way: zero for |q| <= gate (no discontinuity).
    for q in (-0.6, -0.3, 0.0, 0.3, 0.6):
        assert one(q, 10.0, 50.0) == 0.0
        assert two(q, 10.0, 50.0) == 0.0
    # One-sided: the losing side never fires.
    assert one(-0.9, 60.0, 40.0) == 0.0
    assert one(-0.9, 20.0, 40.0) == 0.0
    # Two-sided losing side prefers the SLOWER loss: more moves left -> positive
    # bonus (drag it out), fewer moves left -> negative (avoid the quick loss).
    assert two(-0.9, 60.0, 40.0) > 0.0
    assert two(-0.9, 20.0, 40.0) < 0.0
    # Symmetric with the winning side at equal |q| and mirrored delta.
    assert two(-0.9, 60.0, 40.0) == pytest.approx(two(0.9, 20.0, 40.0), abs=1e-7)
    # Still bounded by w on the losing side.
    assert abs(two(-0.95, 500.0, 0.0)) <= w + 1e-9


def test_build_divergence_overrides() -> None:
    from hexfield.config import SelfplayConfig, build_divergence_overrides

    sp = SelfplayConfig()  # production defaults: MLH on, two-sided, final-pick
    on = build_divergence_overrides(sp)
    assert on["moves_left_utility"] is True
    assert on["ml_two_sided"] is True
    assert on["ml_final_pick"] is True
    assert on["ml_weight"] == pytest.approx(0.03)
    assert on["ml_scale"] == pytest.approx(32.0)
    assert on["ml_q_gate"] == pytest.approx(0.6)
    # The heal-gate / manual disable forces the lever off but keeps the constants
    # (so a later re-enable uses the validated values).
    off = build_divergence_overrides(sp, disabled=True)
    assert off["moves_left_utility"] is False
    assert off["ml_two_sided"] is False
    assert off["ml_final_pick"] is False
    assert off["ml_weight"] == pytest.approx(0.03)
    # Concrete bool/float only — resolve_divergences calls v.extract().
    for value in on.values():
        assert isinstance(value, (bool, float))


@needs_native
def test_early_stop_without_lcb_is_exact() -> None:
    """Pure visit-overtake early-stop is provably selection-invariant for the
    raw greedy argmax, so this gate runs TSS-OFF: the tactical guard is an
    independent overlay that re-ranks among guard-positive moves, where the
    (unvisited tail of the) distribution legitimately depends on when the
    search stopped. The guard's own correctness has its own parity coverage."""

    states = _corpus(60)
    on = hexfield_rust.HexfieldMctsSession(max_states=65536)
    off = hexfield_rust.HexfieldMctsSession(max_states=65536)
    overrides_on = {
        "lcb_move_selection": False,
        "early_stop": True,
        "visit_scaled_c_puct": False,
        "moves_left_utility": False,
        # main_4 KataGo-faithful divergences default ON in production() (the base
        # this overrides dict sits on top of); neutralize them to the parity()
        # legacy value so this gate isolates early_stop vs the search_parity_mode
        # session below.
        "nucleus_f64": False,
        "new_child_fpu": False,
        "lazy_widening": False,
        "clean_root_prior_cache": False,
        "dirichlet_shaped": False,
        "pruned_dynamic_cpuct": False,
    }
    stops = 0
    for index, state in enumerate(states):
        kwargs = dict(
            visits=192, c_puct=1.5, temperature=0.0, seed=7 + index,
            evaluator=HexfieldStub(), virtual_batch_size=8,
            widening_policy_mass=0.95, widening_max_children=96,
            widening_min_children=2, fpu_reduction=0.2, tss_enabled=False,
        )
        a = on.search([index], (state,), divergence_overrides=overrides_on, **kwargs)[0]
        b = off.search([index], (state,), search_parity_mode=True, **kwargs)[0]
        assert a["action_id"] == b["action_id"], index
        stops += int(a["early_stopped"])
        on.discard(index)
        off.discard(index)
    assert stops > 0, "early-stop never fired — gate is vacuous"


@needs_native
def test_reused_root_temperature_applied_once() -> None:
    """Regression (main1 reuse bug, hexfield side): a promoted root gets the
    root-policy temperature exactly once — its exported prior equals
    normalize(raw_eval_prior ** (1/T)), not raw and not double-softened."""

    state = api.new_game()
    for q, r in [(0, 0), (1, 1), (-1, 0)]:
        api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    stub = HexfieldStub()
    session = hexfield_rust.HexfieldMctsSession(max_states=65536)
    temp = 1.5
    kwargs = dict(
        visits=24, c_puct=1.5, temperature=0.0, evaluator=stub,
        virtual_batch_size=8, widening_policy_mass=0.95,
        widening_max_children=96, widening_min_children=2,
        fpu_reduction=0.2, tss_enabled=False, search_parity_mode=True,
        root_policy_temperature=temp,
    )
    r0 = session.search([7], (state,), seed=1, **kwargs)[0]
    q, rr = unpack_action_id(int(r0["action_id"]))
    api.apply_action(state, PlacementAction(AxialCoord(q=q, r=rr)))

    r1 = session.search([7], (state,), seed=2, **kwargs)[0]
    ids = np.frombuffer(bytes(r1["root_prior_policy_action_ids_bytes"]), dtype=np.uint32)
    weights = np.frombuffer(bytes(r1["root_prior_policy_weights_bytes"]), dtype=np.float32)

    # Expected: the stub's raw priors for THIS state, normalized, then ^(1/T),
    # renormalized — computed independently through the Rust featurizer path.
    payload_rows = hexfield_rust.featurize_states([state])
    row = payload_rows[0]
    # Rebuild the stub's view via the actual evaluator payload contract:
    feats_f16 = (
        np.frombuffer(row["feats"], dtype=np.float32).astype(np.float16).tobytes()
    )  # featurize_states emits f32; the evaluator wire carries f16
    fake_payload = {
        "shape": (1, row["num_nodes"]),
        "legal_counts": np.asarray([row["legal_count"]], dtype=np.int32).tobytes(),
        "node_row_offsets": [0, row["num_nodes"]],
        "node_qr": row["coords"],
        "node_feats": feats_f16,
        "abi": 1,
    }
    reply = stub(fake_payload)
    raw = np.frombuffer(reply["priors_bytes"], dtype=np.float32).astype(np.float64)
    coords = np.frombuffer(row["coords"], dtype=np.int16).reshape(-1, 2)
    legal_ids = np.asarray(
        [((int(q) + (1 << 15)) << 16) | (int(r) + (1 << 15)) for q, r in coords[: row["legal_count"]]],
        dtype=np.uint64,
    )
    norm = raw / raw.sum()
    softened = np.where(norm > 0, norm ** (1.0 / temp), 0.0)
    softened = softened / softened.sum()

    expected = dict(zip(legal_ids.astype(np.uint32).tolist(), softened.tolist()))
    got = dict(zip(ids.tolist(), weights.astype(np.float64).tolist()))
    # zero-prior (out-of-disk) cells are dropped from the export on both sides
    expected = {k: v for k, v in expected.items() if v > 0}
    assert set(got) == set(expected)
    for aid, w in got.items():
        assert w == pytest.approx(expected[aid], rel=2e-3), aid