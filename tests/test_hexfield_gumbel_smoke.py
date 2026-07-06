"""End-to-end smoke and config gates for the Gumbel search flags.

Drives the Rust search (`HexfieldMctsSession.search`) with the four Gumbel
divergence flags enabled (expressed via `divergence_overrides`) over a handful
of decision states, using a stub evaluator that emits the raw
`priors_logits_bytes` column when the search requests it. Requires no torch.

Asserts: the search runs without panicking with Gumbel-Top-k root, Sequential
Halving, non-root select, and target export enabled; and the exported
improved-policy target (`gumbel_policy_weights_bytes`) is a normalized
distribution over its support.

Also covers config load and the strict-key guard.
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
    """Stub evaluator.

    Emits a deterministic, descending (peaked) normalized prior per legal
    action, plus, when the search sets ``request_logits``, raw pre-softmax
    logits in the same positional layout as ``priors_bytes``. The logits are
    ``log(prior)``, so softmax over them reproduces the priors."""

    def __call__(self, payload: dict) -> dict:
        b, _total = payload["shape"]
        legal_counts = np.frombuffer(payload["legal_counts"], dtype=np.int32)
        values: list[float] = []
        priors: list[float] = []
        logits: list[float] = []
        for g in range(b):
            l = int(legal_counts[g])
            # Deterministic, descending prior over this row's legal set.
            raw = np.array([1.0 / (1 + i) for i in range(l)], dtype=np.float64)
            p = raw / raw.sum()
            priors.extend(float(x) for x in p)
            # Raw logits == log(prior); softmax over them reproduces the priors.
            logits.extend(float(math.log(x)) for x in p)
            # Small non-zero values, alternating sign across rows.
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
    """divergence_overrides dict with the four Gumbel bools enabled plus the
    σ/candidate scalars."""
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

        # With gumbel_target on, the improved-policy target columns are present.
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
            # Weights form a normalized distribution over the support.
            assert np.all(np.isfinite(weights)), "target weights must be finite"
            assert np.all(weights >= -1e-6), "target weights must be non-negative"
            assert abs(float(weights.sum()) - 1.0) < 1e-4, (
                f"gumbel target must sum to 1, got {float(weights.sum())}"
            )
            # Action ids in the support are unique.
            assert len(set(ids.tolist())) == len(ids), "duplicate target action ids"
        session.discard(key)

    assert produced_target >= 1, "no state produced a non-empty Gumbel target"


@needs_rust
def test_gumbel_lockstep_search_engages_sh_on_fresh_and_reused_roots() -> None:
    """The lockstep ``session.search`` path (the eval-arena driver) builds the
    Gumbel-Top-k + SH root for fresh AND reused (advanced) roots when
    ``gumbel_root`` is on.

    Under SH every one of the m candidates must reach its round-0 quota before
    any halving, so the delta-visit distribution spreads over at least m
    actions. The PUCT root path with this stub's peaked prior concentrates on
    far fewer. Guards the lockstep init (fresh + reuse blocks) — before it was
    added, session.search never engaged the SH root at all."""
    from hexo_engine.types import AxialCoord, PlacementAction

    from hexfield.geometry import unpack_action_id

    m = 8
    overrides = _gumbel_overrides()
    overrides["gumbel_m"] = m
    session = hexfield_rust.HexfieldMctsSession(max_states=65536)
    stub = GumbelStub()
    key = 70_000
    # Mid-game decision state: the empty board has a single forced legal move
    # (support 1 is CORRECT there), so SH engagement is only observable from a
    # position with a real branching factor.
    state = sample_decision_states(range(40), (6, 7, 8))[0]

    def one_move(state, seed):
        # 96 visits: m=8 needs floor(96/(3*8)) >= 4 round-0 visits per
        # candidate to survive the in-tree budget calibration of gumbel_m.
        results = session.search(
            [key],
            (state,),
            evaluator=stub,
            visits=96,
            c_puct=1.5,
            temperature=1.0,  # >0 disables the lockstep early stop
            seed=seed,
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
        return results[0]

    # Fresh root: SH spreads the delta-visit support over all m candidates.
    r1 = one_move(state, seed=31337)
    assert r1["visit_policy_count"] >= m, (
        f"fresh lockstep root did not engage SH: support {r1['visit_policy_count']} < {m}"
    )

    # Advance the engine state by the played move; the session stored the
    # advanced tree under `key`, so this second call takes the REUSE block.
    q, r = unpack_action_id(r1["action_id"])
    api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    r2 = one_move(state, seed=31337)  # same per-call seed: root-hash mix decorrelates
    assert r2["visit_policy_count"] >= m, (
        f"reused lockstep root did not rebuild SH: support {r2['visit_policy_count']} < {m}"
    )
    session.discard(key)


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
                (
                    ply,
                    bool(payload.get("pcr_full")),
                    int(payload["visits"]),
                    int(payload.get("gumbel_policy_count", 0)),
                )
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

    full_rows = [(ply, visits, gc) for ply, full, visits, gc in driver.rows if full]
    assert len(full_rows) >= 2 * (max_plies - 1), "expected full games of Full moves"
    reused = [(ply, visits, gc) for ply, visits, gc in full_rows if ply >= 1]
    assert reused, "no reused-root moves decided"
    zero_visit = [row for row in reused if row[1] == 0]
    assert not zero_visit, (
        f"reused-root Full moves finalized with zero net visits: {zero_visit}"
    )
    mean_visits = sum(v for _, v, _ in reused) / len(reused)
    assert mean_visits >= budget * 0.5, (
        f"reused-root Full moves under-searched: mean {mean_visits:.1f} of {budget}"
    )
    # Gumbel-specific observable: every reused-root Full move exports a π'
    # target over more than one action. A plain-PUCT .so (version skew) or a
    # gumbel profile that silently reverted to PUCT passes the visit assertions
    # above but has no gumbel_policy_count key — this catches that class.
    thin_targets = [row for row in reused if row[2] < 2]
    assert not thin_targets, (
        f"reused-root Full moves exported degenerate π' targets: {thin_targets}"
    )


@needs_rust
def test_gumbel_flags_off_omits_target_column() -> None:
    """With the Gumbel bools off (the default), the search result omits the
    Gumbel target keys."""
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
            # No divergence_overrides: defaults leave all gumbel bools off.
        )
        r = results[0]
        assert "gumbel_policy_weights_bytes" not in r
        assert "gumbel_policy_action_ids_bytes" not in r
        assert "root_prior_logits_bytes" not in r
        session.discard(key)


def test_main6_config_loads_with_gumbel_on() -> None:
    """hexfield_main_6.toml parses and enables all four Gumbel mechanisms, with
    the σ/candidate scalars set and training.policy_target == 'gumbel'."""
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
    """policy_target belongs under [model.config.training]; placing it under
    [model.config.selfplay] raises ValueError at load."""
    from hexfield.config import parse_hexfield_config

    with pytest.raises(ValueError):
        parse_hexfield_config({"selfplay": {"policy_target": "gumbel"}})


def test_build_divergence_overrides_emits_gumbel_knobs() -> None:
    """build_divergence_overrides emits the four gumbel bools plus the σ/
    candidate scalars as concrete bool/float/int values (never None)."""
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
    assert ov["gumbel_play_prune"] is False  # default off; main_6 enables it
    for k, v in ov.items():
        assert v is not None, k
        assert isinstance(v, (bool, float, int)), (k, type(v))


@needs_rust
def test_gumbel_play_prune_zeroes_quota_losers_without_touching_targets() -> None:
    """gumbel_play_prune: the PLAYED move samples the quota-pruned histogram
    (action_selection == 'gumbel_play_policy', play stats counted), while the
    RECORDED visit-policy target keeps the full SH support (round-0 losers
    included). Off by default: the same run without the flag keeps the legacy
    'delta_visit_policy' selection."""
    from hexo_engine import api as engine_api
    from hexo_engine.types import AxialCoord, PlacementAction

    from hexfield.geometry import unpack_action_id

    budget = 96
    m = 8
    max_plies = 8

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
                (
                    ply,
                    bool(payload.get("pcr_full")),
                    str(payload.get("action_selection")),
                    bool(payload.get("play_pruned")),
                    int(payload.get("visit_policy_count", 0)),
                )
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

    def run(play_prune: bool):
        overrides = _gumbel_overrides()
        overrides["gumbel_m"] = m
        overrides["gumbel_play_prune"] = play_prune
        session = hexfield_rust.HexfieldMctsSession(max_states=65536)
        driver = _Driver()
        keys = [93_000]
        states = tuple(driver.start(k) for k in keys)
        stats = session.run_continuous(
            keys,
            states,
            evaluator=GumbelStub(),
            on_move=driver,
            visits=budget,
            c_puct=1.5,
            base_seed=777_777,
            virtual_batch_size=8,
            flush_target=64,
            active_root_limit=1,
            temperature_by_ply=[1.0] * 64,
            root_policy_temperature=1.0,
            fpu_reduction=0.2,
            virtual_loss=1.0,
            widening_policy_mass=0.95,
            widening_max_children=96,
            widening_min_children=2,
            forced_playout_k=0.0,
            pcr_full_proportion=1.0,
            pcr_fast_visits=32,
            policy_init_fraction=0.0,
            policy_init_avg_plies=0.0,
            policy_init_max_plies=0,
            policy_init_temperature=1.0,
            tss_enabled=False,
            root_fpu_reduction=0.2,
            root_fpu_zero_under_noise=False,
            search_parity_mode=False,
            divergence_overrides=overrides,
        )
        return driver.rows, stats

    rows_on, stats_on = run(play_prune=True)
    # Reused-root Full moves (ply >= 1; ply 0 is the forced single opening
    # move) select via the pruned play distribution...
    pruned = [r for r in rows_on if r[0] >= 1 and r[1]]
    assert pruned, "no reused-root Full moves decided"
    assert all(r[2] == "gumbel_play_policy" and r[3] for r in pruned), pruned
    # ...but the RECORDED visit-policy target keeps the full SH candidate
    # support (round-0 losers included): support >= m for a full-budget move.
    assert all(r[4] >= m for r in pruned), (
        f"recorded target support shrank under play prune: {pruned}"
    )
    # Scheduler telemetry counts the pruned selections and the winner rate.
    assert int(stats_on["gumbel_play_moves"]) >= len(pruned)
    assert (
        0
        <= int(stats_on["gumbel_play_winner_moves"])
        <= int(stats_on["gumbel_play_moves"])
    )

    rows_off, stats_off = run(play_prune=False)
    legacy = [r for r in rows_off if r[0] >= 1 and r[1]]
    assert legacy and all(r[2] == "delta_visit_policy" and not r[3] for r in legacy)
    assert int(stats_off["gumbel_play_moves"]) == 0


def test_build_fast_divergence_overrides_golden_and_folded() -> None:
    """main_8 per-class maps: with no fast_* levers the fast map equals the base
    map (golden invariant); with fast_* levers set, only the fast map's Gumbel
    fields change (base map untouched)."""
    from hexfield.config import (
        SelfplayConfig,
        build_divergence_overrides,
        build_fast_divergence_overrides,
    )

    # Golden: absent fast_* => fast map == base map, byte-for-byte.
    sp0 = SelfplayConfig(gumbel_root_enabled=False)
    base0 = build_divergence_overrides(sp0)
    fast0 = build_fast_divergence_overrides(sp0)
    assert fast0 == base0, "fast map must equal base map when no fast_* levers set"

    # Folded: Full stays PUCT (base gumbel_root False); Fast turns Gumbel on with
    # its own c_scale/m. The base map keeps the Full values.
    sp = SelfplayConfig(
        gumbel_root_enabled=False,
        gumbel_sequential_halving=False,
        gumbel_c_scale=1.0,
        gumbel_m=16,
        fast_gumbel_root_enabled=True,
        fast_gumbel_sequential_halving=True,
        fast_gumbel_nonroot_select=True,
        fast_gumbel_c_scale=0.5,
        fast_gumbel_m=8,
        fast_gumbel_play_prune=True,
    )
    base = build_divergence_overrides(sp)
    fast = build_fast_divergence_overrides(sp)
    # Base (Full) view keeps PUCT + the base scalars.
    assert base["gumbel_root"] is False
    assert base["gumbel_sequential_halving"] is False
    assert base["gumbel_c_scale"] == pytest.approx(1.0)
    assert base["gumbel_m"] == 16
    # Fast view folds the fast_* values onto the gumbel fields.
    assert fast["gumbel_root"] is True
    assert fast["gumbel_sequential_halving"] is True
    assert fast["gumbel_nonroot_select"] is True
    assert fast["gumbel_c_scale"] == pytest.approx(0.5)
    assert fast["gumbel_m"] == 8
    assert fast["gumbel_play_prune"] is True
    # Non-gumbel levers are shared between the two maps.
    for k in ("moves_left_utility", "lcb_z", "c_base", "nucleus_f64"):
        assert base[k] == fast[k], k
    # Unset fast numeric lever falls back to the base value.
    sp_partial = SelfplayConfig(fast_gumbel_root_enabled=True)  # c_visit/m unset
    fp = build_fast_divergence_overrides(sp_partial)
    bp = build_divergence_overrides(sp_partial)
    assert fp["gumbel_c_visit"] == bp["gumbel_c_visit"]
    assert fp["gumbel_m"] == bp["gumbel_m"]


@needs_rust
def test_continuous_cross_class_full_puct_fast_gumbel_reuse() -> None:
    """main_8 per-class divergences end-to-end: run_continuous with a base
    (Full/PUCT) override map and a SECOND fast (Gumbel+SH) override map. Turns
    alternate Full/Fast by the per-turn PCR hash; the driver must not panic and
    every decided move — Full or Fast — must produce a sane, budget-respecting
    visit distribution. Fast moves (Gumbel) additionally export a π' target."""
    from hexo_engine import api as engine_api
    from hexo_engine.types import AxialCoord, PlacementAction

    from hexfield.geometry import unpack_action_id

    budget = 96
    fast_budget = 48
    max_plies = 12

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
                (
                    ply,
                    bool(payload.get("pcr_full")),
                    int(payload["visits"]),
                    int(payload.get("gumbel_policy_count", 0)),
                )
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

    # Base map: Full/Init = PUCT (base gumbel_root off). Fast map: base map with
    # the Fast Gumbel view (root + SH + non-root select) folded in.
    from hexfield.config import (
        SelfplayConfig,
        build_divergence_overrides as _bdo,
        build_fast_divergence_overrides,
    )

    sp = SelfplayConfig(
        gumbel_root_enabled=False,  # Full stays PUCT
        gumbel_target_enabled=True,  # export π' so Fast rows carry a target
        gumbel_sequential_halving=False,
        gumbel_nonroot_select=False,
        gumbel_m=8,
        fast_gumbel_root_enabled=True,
        fast_gumbel_sequential_halving=True,
        fast_gumbel_nonroot_select=True,
        fast_gumbel_m=8,
    )
    base_overrides = _bdo(sp)
    fast_overrides = build_fast_divergence_overrides(sp)
    # Sanity: the two maps really differ (Full PUCT vs Fast Gumbel root).
    assert base_overrides["gumbel_root"] is False
    assert fast_overrides["gumbel_root"] is True

    session = hexfield_rust.HexfieldMctsSession(max_states=65536)
    driver = _Driver()
    keys = [95_000 + i for i in range(6)]
    states = tuple(driver.start(k) for k in keys)
    session.run_continuous(
        keys,
        states,
        evaluator=GumbelStub(),
        on_move=driver,
        visits=budget,
        c_puct=1.5,
        base_seed=20260706,
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
        pcr_full_proportion=0.5,  # turns alternate Full/Fast by the PCR hash
        pcr_fast_visits=fast_budget,
        policy_init_fraction=0.0,
        policy_init_avg_plies=0.0,
        policy_init_max_plies=0,
        policy_init_temperature=1.0,
        tss_enabled=False,
        root_fpu_reduction=0.2,
        root_fpu_zero_under_noise=False,
        search_parity_mode=False,
        divergence_overrides=base_overrides,
        fast_divergence_overrides=fast_overrides,
    )

    full_rows = [(p, v, gc) for p, full, v, gc in driver.rows if full]
    fast_rows = [(p, v, gc) for p, full, v, gc in driver.rows if not full]
    # Both classes must actually occur (per-turn hash at 0.5 over 6 games x many
    # plies is overwhelmingly likely to yield both).
    assert full_rows, "no Full-class moves decided"
    assert fast_rows, "no Fast-class moves decided"
    # No move finalized with zero net visits; each respects its class budget.
    for p, v, _gc in full_rows:
        assert 0 < v <= budget, f"Full move ply {p} visits {v} out of (0, {budget}]"
    for p, v, _gc in fast_rows:
        assert 0 < v <= fast_budget, (
            f"Fast move ply {p} visits {v} out of (0, {fast_budget}]"
        )
    # Fast moves run under Gumbel+SH: reused-root Fast moves export a π' target
    # over more than one action (the observable that the Fast regime engaged).
    fast_reused = [(p, v, gc) for p, v, gc in fast_rows if p >= 1]
    assert fast_reused, "no reused-root Fast moves decided"
    assert any(gc >= 2 for _p, _v, gc in fast_reused), (
        f"Fast (Gumbel) moves never exported a multi-action target: {fast_reused}"
    )


def test_future_opponent_policy_prefers_gumbel_target() -> None:
    """The opp-policy target uses the opponent decision's improved policy π'
    when it carries one (``future_opponent_gumbel``), falling back to the visit
    policy otherwise — same schedule-artifact reasoning as the main/soft
    policy target selection. Fast-masking still wins over both."""
    from hexfield.samples import HexfieldSampleData, _future_opponent_policy

    def _decision(player: int, *, policy=(), gumbel=(), full=True):
        return (
            player,
            HexfieldSampleData(
                game_id="g",
                turn_index=0,
                current_player=player,
                phase="Opening",
                records=(),
                first_stone=None,
                own_hot=(),
                opp_hot=(),
                own_win=(),
                opp_win=(),
                policy=tuple(policy),
                gumbel_policy=tuple(gumbel),
                metadata={"pcr_full": full},
            ),
            0.0,
        )

    visit = ((5, 0.6), (6, 0.4))
    pi = ((5, 0.1), (7, 0.9))  # π' disagrees with visits and adds action 7

    # Opponent decision carries π' -> π' wins.
    decisions = [_decision(0), _decision(1, policy=visit, gumbel=pi)]
    target, source = _future_opponent_policy(decisions, 0, 0, mask_from_fast=True)
    assert source == "future_opponent_gumbel" and target == pi

    # No π' on the opponent decision -> visit fallback.
    decisions = [_decision(0), _decision(1, policy=visit)]
    target, source = _future_opponent_policy(decisions, 0, 0, mask_from_fast=True)
    assert source == "future_opponent_mcts" and target == visit

    # Fast (unrecorded) opponent decision still masks, π' or not.
    decisions = [_decision(0), _decision(1, policy=(), gumbel=(), full=False)]
    target, source = _future_opponent_policy(decisions, 0, 0, mask_from_fast=True)
    assert source == "fast_unrecorded_masked" and target == ()


def test_shard_v3_gumbel_csr_roundtrip_preserves_off_support_mass(tmp_path) -> None:
    """Schema v3: the π' target survives write -> read -> packed-window load
    with its FULL support, including actions outside the recorded visit-policy
    support (inherited edges on reused roots). Guards the v2 defect where the
    aligned column silently truncated π' to pol_act and renormalized the
    dropped mass away."""
    from hexfield.samples import HexfieldSampleData
    from hexfield.shards import read_compact_shard, write_compact_shard
    from hexfield.window import load_packed_shard

    # Visit support {10, 11}; π' support {10, 11, 12} with the argmax OFF the
    # visit support (action 12 = the underrated inherited edge).
    sample = HexfieldSampleData(
        game_id="g",
        turn_index=3,
        current_player=0,
        phase="Opening",
        records=((0, 0, 0, 1), (1, 0, 1, 2), (0, 1, 0, 3)),
        first_stone=None,
        own_hot=(),
        opp_hot=(),
        own_win=(),
        opp_win=(),
        policy=((10, 0.75), (11, 0.25)),
        q_policy=((10, 0.1), (11, -0.2)),
        gumbel_policy=((10, 0.2), (11, 0.1), (12, 0.7)),
        prior_logit=((10, -1.0), (11, -2.0)),
        opp_policy=(),
        value=1.0,
        short_term_value=(),
        moves_left=0.5,
        policy_surprise=0.0,
        metadata={"pcr_full": True},
    )
    path = tmp_path / "game_1.npz"
    write_compact_shard(path, [sample])

    # Reader round-trip: full π' support survives, off-support action included.
    (row,) = read_compact_shard(path)
    got_read = dict(row.gumbel_policy)
    assert set(got_read.keys()) == {10, 11, 12}
    assert abs(got_read[12] - 0.7) < 1e-6
    assert abs(sum(got_read.values()) - 1.0) < 1e-6

    # Packed-window round-trip (the production rust-expand input path).
    win = load_packed_shard(path)
    view = win.row_view(0)
    got = dict(view.gumbel_policy())
    assert set(got.keys()) == {10, 11, 12}
    assert abs(got[12] - 0.7) < 1e-6, "off-support π' mass must survive packing"
    # The visit policy itself is untouched.
    assert dict(view.policy()) == {10: 0.75, 11: 0.25}
