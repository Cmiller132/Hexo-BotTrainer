"""M6 gates: continuous-scheduler differential parity vs dense_cnn, plus the
end-to-end ABI round-trip through the real model.

Both sides drive `run_continuous` over the same fresh games with the same
base_seed and the full per-move machinery (PCR full/fast coin, policy-init
openings, Dirichlet noise, temperature schedules, TSS, forced playouts,
as-built quarantined knobs). The on_move streams must match exactly: move
classes, chosen actions, visit counts, exported (pruned) visit-policy bytes.
"""

from __future__ import annotations

import pytest

from hexfield_testkit import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield.geometry import unpack_action_id
from test_hexfield_search_parity import DenseStub, HexfieldStub

try:
    from hexfield import _rust as hexfield_rust
except ImportError:  # pragma: no cover
    hexfield_rust = None

try:
    from hexo_models._rust import dense_cnn as dense_rust
except ImportError:  # pragma: no cover
    dense_rust = None

needs_native = pytest.mark.skipif(
    hexfield_rust is None or dense_rust is None,
    reason="native modules not built",
)

# Keeps every searched state (incl. noised ROOTS, where an out-of-disk legal
# would change the Dirichlet candidate count: hexfield's noise spans its full
# legal vocabulary, dense's only the in-crop set) deep inside dense's crop —
# the structural boundary of the comparison, not a search-semantics limit.
# The Recorder ASSERTS the §5.1 corpus constraint at every decision state.
MAX_PLIES = 6


class Recorder:
    """on_move driver: applies actions to its own engine states and records
    the full decision stream."""

    def __init__(self):
        self.states = {}
        self.records = []

    def add_game(self, key: int):
        self.states[key] = api.new_game()

    def __call__(self, game_key: int, payload: dict):
        from test_hexfield_search_parity import _fully_in_crop

        state = self.states[game_key]
        # Deep-leaf out-of-disk legals are handled exactly by the stub
        # (zero prior, dense-equivalent); only noised ROOTS require full
        # in-disk legality (the Dirichlet candidate count must match).
        assert _fully_in_crop(state, margin=0), (
            "decision-state legal set left dense's crop — lower MAX_PLIES "
            "(spec §5.1: the differential corpus must stay in-crop)"
        )
        action_id = payload["action_id"]
        self.records.append(
            (
                game_key,
                action_id,
                bool(payload["pcr_full"]),
                bool(payload["policy_init"]),
                int(payload["visits"]),
                bytes(payload["visit_policy_action_ids_bytes"]),
                bytes(payload["visit_policy_weights_bytes"]),
                round(float(payload["root_value"]), 6),
            )
        )
        q, r = unpack_action_id(action_id)
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        ply = sum(1 for rec in self.records if rec[0] == game_key)
        if result.terminal or ply >= MAX_PLIES:
            return None
        return ("advance", state)


def _continuous_kwargs():
    return dict(
        visits=48,
        c_puct=1.5,
        base_seed=20260613,
        virtual_batch_size=8,
        flush_target=24,
        active_root_limit=16,
        temperature_by_ply=[1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
        root_dirichlet_total_alpha=10.83,
        root_dirichlet_noise_fraction=0.25,
        root_policy_temperature=1.1,
        root_policy_temperature_early=1.4,
        root_policy_temperature_halflife=12.0,
        fpu_reduction=0.2,
        virtual_loss=1.0,
        widening_policy_mass=0.95,
        widening_max_children=96,
        widening_min_children=2,
        forced_playout_k=2.0,
        pcr_full_proportion=0.33,
        pcr_fast_visits=16,
        policy_init_fraction=0.25,
        policy_init_avg_plies=4.0,
        policy_init_max_plies=8,
        policy_init_temperature=1.4,
        tss_enabled=True,
        # dense AS-BUILT quarantined knob value, passed to both sides.
        root_fpu_zero_under_noise=True,
    )


@needs_native
def test_continuous_parity_full_machinery() -> None:
    keys = list(range(700, 706))

    hex_recorder = Recorder()
    dense_recorder = Recorder()
    for key in keys:
        hex_recorder.add_game(key)
        dense_recorder.add_game(key)

    hex_session = hexfield_rust.HexfieldMctsSession(max_states=65536)
    hex_stats = hex_session.run_continuous(
        keys,
        tuple(hex_recorder.states[k] for k in keys),
        evaluator=HexfieldStub(),
        on_move=hex_recorder,
        search_parity_mode=True,
        **_continuous_kwargs(),
    )

    dense_session = dense_rust.Model1MctsSession(65536)
    dense_stats = dense_session.run_continuous(
        keys,
        tuple(dense_recorder.states[k] for k in keys),
        evaluator=DenseStub(),
        on_move=dense_recorder,
        **_continuous_kwargs(),
    )

    assert hex_stats["moves_decided"] == dense_stats["moves_decided"]
    assert hex_stats["full_moves"] == dense_stats["full_moves"]
    assert hex_stats["fast_moves"] == dense_stats["fast_moves"]
    assert hex_stats["init_moves"] == dense_stats["init_moves"]

    assert len(hex_recorder.records) == len(dense_recorder.records)
    mismatches = [
        (i, h, d)
        for i, (h, d) in enumerate(zip(hex_recorder.records, dense_recorder.records))
        if h != d
    ]
    assert not mismatches, f"{len(mismatches)} record mismatches; first: {mismatches[:3]}"


@needs_native
def test_divergences_off_equals_parity_mode() -> None:
    """(all divergences off via overrides) ≡ search_parity_mode bit-for-bit."""

    keys = [900, 901, 902]
    rec_a = Recorder()
    rec_b = Recorder()
    for key in keys:
        rec_a.add_game(key)
        rec_b.add_game(key)
    kwargs = _continuous_kwargs()

    session_a = hexfield_rust.HexfieldMctsSession(max_states=65536)
    session_a.run_continuous(
        keys, tuple(rec_a.states[k] for k in keys), evaluator=HexfieldStub(),
        on_move=rec_a, search_parity_mode=True, **kwargs,
    )
    session_b = hexfield_rust.HexfieldMctsSession(max_states=65536)
    session_b.run_continuous(
        keys, tuple(rec_b.states[k] for k in keys), evaluator=HexfieldStub(),
        on_move=rec_b,
        divergence_overrides={
            "lcb_move_selection": False,
            "early_stop": False,
            "visit_scaled_c_puct": False,
            "moves_left_utility": False,
        },
        **kwargs,
    )
    assert rec_a.records == rec_b.records


@needs_native
def test_e2e_abi_roundtrip_with_real_model() -> None:
    """Payload -> packer -> HexfieldNet -> reply -> tree, divergences ON
    (exercises moves_left_bytes end to end). Sanity: search completes, visits
    land, the chosen move is legal."""

    import torch

    from hexfield.inference import HexfieldEvaluator
    from hexfield.model import HexfieldNet

    torch.manual_seed(0)
    evaluator = HexfieldEvaluator(HexfieldNet(), device="cpu")
    session = hexfield_rust.HexfieldMctsSession(max_states=4096)

    state = api.new_game()
    for q, r in [(0, 0), (1, 1), (-1, 0), (2, -1), (0, 2)]:
        api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    legal = set(api.legal_action_ids(state))

    results = session.search(
        [4242],
        (state,),
        visits=24,
        c_puct=1.5,
        temperature=0.0,
        seed=5,
        evaluator=evaluator,
        virtual_batch_size=8,
        widening_policy_mass=0.95,
        widening_max_children=96,
        widening_min_children=2,
    )
    result = results[0]
    assert result["visits"] == 24
    assert result["action_id"] in legal
    assert result["visit_policy_count"] >= 1
    # Production defaults: divergences ON => the evaluator was asked for
    # moves_left and the reply carried it (no exception got here), and the
    # LCB/early-stop fields exist in the payload.
    assert "lcb_override" in result
    assert "early_stopped" in result