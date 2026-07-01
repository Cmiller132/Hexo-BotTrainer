"""Stub-evaluator differential parity between the hexfield and dense_cnn MCTS
sessions.

The corpus is constrained (and asserted) to positions whose full legal set
lies inside dense's radius-20 crop, so both engines share an identical move
vocabulary. The stub evaluator keys priors/values by the legal cells' crop
flats; both payloads expose the legal set in ascending-id order (asserted in
`_fully_in_crop`). hexfield runs with `search_parity_mode=True`; the same knob
values are passed to both sides. Identical PUCT constants, seed streams, and
priors are expected to yield identical visit counts, chosen moves, root values,
and exported visit-policy targets.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from hexfield_testkit import api, sample_decision_states

from hexfield.geometry import hex_dist, unpack_action_id

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


def _stub_prior_from_flat(flat: int, row_hash: int) -> float:
    return float((flat * 2654435761 + row_hash * 97) % 1000 + 1)


def _stub_value_from_hash(row_hash: int) -> float:
    return float(row_hash % 2001 - 1000) / 1000.0


def _row_hash(flats: list[int]) -> int:
    h = 1469598103
    for flat in flats:
        h = (h ^ flat) * 1099511628211 % (1 << 61)
    return h % 1000003


def _stub_reply(rows: list[list[int]], request_ml: bool = False) -> dict:
    """Build values/priors reply from per-row dense-crop legal flats.

    `rows` is a list of per-row flat-index lists. Returns a dict with
    little-endian float32 `values_bytes` (one per row) and `priors_bytes`
    (one per legal flat, row-concatenated); `moves_left_bytes` is added when
    `request_ml` is set.
    """

    values = []
    priors: list[float] = []
    for flats in rows:
        rh = _row_hash(flats)
        values.append(_stub_value_from_hash(rh))
        priors.extend(_stub_prior_from_flat(f, rh) for f in flats)
    b = len(rows)
    reply = {
        "values_bytes": struct.pack(f"<{b}f", *values),
        "priors_bytes": struct.pack(f"<{len(priors)}f", *priors),
    }
    if request_ml:
        reply["moves_left_bytes"] = struct.pack(f"<{b}f", *([100.0] * b))
    return reply


def _python_round(numerator: int, denominator: int) -> int:
    """Integer division rounding half to even. Matches the round-half-to-even
    behavior used by dense's encoding."""

    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled < denominator:
        return quotient
    if doubled > denominator:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


def _hexd(dq: int, dr: int) -> int:
    return max(abs(dq), abs(dr), abs(dq + dr))


class HexfieldStub:
    """Evaluator over hexfield's CSR payload.

    Derives the crop center from the stones' rounded centroid, then maps each
    legal cell to its radius-20 crop flat. Legal cells outside the radius-20
    hex disk are assigned prior 0.0. Values are keyed by the row hash of the
    in-disk flats. Returns the same reply layout as `_stub_reply`.
    """

    def __call__(self, payload: dict) -> dict:
        b, total = payload["shape"]
        legal_counts = np.frombuffer(payload["legal_counts"], dtype=np.int32)
        offsets = np.asarray(payload["node_row_offsets"], dtype=np.int64)
        qr = np.frombuffer(payload["node_qr"], dtype=np.int16).reshape(total, 2)
        feats = np.frombuffer(payload["node_feats"], dtype=np.float16).reshape(total, 15)
        values = []
        priors: list[float] = []
        for g in range(b):
            o, e = int(offsets[g]), int(offsets[g + 1])
            l = int(legal_counts[g])
            legal = qr[o : o + l]
            seg = feats[o:e]
            stones = qr[o:e][(seg[:, 0] + seg[:, 1]) > 0.5]
            if len(stones):
                cq = _python_round(int(stones[:, 0].astype(np.int64).sum()), len(stones))
                cr = _python_round(int(stones[:, 1].astype(np.int64).sum()), len(stones))
            else:
                cq, cr = 0, 0
            in_disk_flats = []
            row_priors = []
            for q, r in legal:
                dq, dr = int(q) - cq, int(r) - cr
                if _hexd(dq, dr) <= 20:
                    flat = (dr + 20) * 41 + (dq + 20)
                    in_disk_flats.append((flat, len(row_priors)))
                    row_priors.append(None)  # filled below
                else:
                    row_priors.append(0.0)
            rh = _row_hash([f for f, _ in in_disk_flats])
            for flat, idx in in_disk_flats:
                row_priors[idx] = _stub_prior_from_flat(flat, rh)
            values.append(_stub_value_from_hash(rh))
            priors.extend(row_priors)
        reply = {
            "values_bytes": struct.pack(f"<{b}f", *values),
            "priors_bytes": struct.pack(f"<{len(priors)}f", *priors),
        }
        if payload.get("request_moves_left"):
            reply["moves_left_bytes"] = struct.pack(f"<{b}f", *([100.0] * b))
        return reply


class DenseStub:
    """Evaluator over dense's payload, which supplies crop flats directly via
    `legal_flat_indices` per row."""

    def __call__(self, payload: dict) -> dict:
        offsets = payload["legal_row_offsets"]
        b = payload["shape"][0]
        flats_all = np.frombuffer(
            bytes(payload["legal_flat_indices_bytes"]), dtype=np.int64
        )
        rows = []
        for row in range(b):
            rows.append([int(f) for f in flats_all[int(offsets[row]) : int(offsets[row + 1])]])
        return _stub_reply(rows)


def _crop_center(stones: list[tuple[int, int]]) -> tuple[int, int]:
    if not stones:
        return (0, 0)
    q = round(sum(s[0] for s in stones) / len(stones))
    r = round(sum(s[1] for s in stones) / len(stones))
    return int(q), int(r)


def _fully_in_crop(state, margin: int) -> bool:
    """Return True when every legal cell and every stone lies within
    `20 - margin` of the crop center. dense recomputes its centroid crop per
    state, so `margin` bounds how far leaf states reached during search can
    shift the crop while keeping their legal sets in-crop.

    Also asserts the engine's legal action ids are ascending.
    """

    mirror = api.to_python_state(state)
    stones = [(c.q, c.r) for c, _p in mirror.board.stones]
    cq, cr = _crop_center(stones)
    ids = api.legal_action_ids(state)
    if list(ids) != sorted(ids):
        raise AssertionError("engine legal ids not ascending — stub keying invalid")
    limit = 20 - margin
    for aid in ids:
        q, r = unpack_action_id(aid)
        if hex_dist(q - cq, r - cr) > limit:
            return False
    for q, r in stones:
        if hex_dist(q - cq, r - cr) > limit:
            return False
    return True


def _corpus(min_positions: int = 100, margin: int = 9):
    states = sample_decision_states(range(200), (1, 2, 3, 4, 5, 6, 7, 8))
    in_crop = [s for s in states if _fully_in_crop(s, margin)]
    assert len(in_crop) >= min_positions, f"only {len(in_crop)} in-crop positions"
    return in_crop[:min_positions]


def _run_pair(states, *, visits, seed, temperature, noise, forced_k, root_temp,
              fpu_zero_under_noise, tss, virtual_batch):
    hex_session = hexfield_rust.HexfieldMctsSession(max_states=65536)
    dense_session = dense_rust.Model1MctsSession(65536)
    hex_stub = HexfieldStub()
    dense_stub = DenseStub()
    mismatches = []
    for index, state in enumerate(states):
        key = 10_000 + index
        kwargs = dict(
            visits=visits,
            c_puct=1.5,
            temperature=temperature,
            seed=seed + index * 7919,
            virtual_batch_size=virtual_batch,
            fpu_reduction=0.2,
            virtual_loss=1.0,
            widening_policy_mass=0.95,
            widening_max_children=96,
            widening_min_children=2,
            forced_playout_k=forced_k,
            root_policy_temperature=root_temp,
            tss_enabled=tss,
            root_fpu_zero_under_noise=fpu_zero_under_noise,
        )
        if noise is not None:
            kwargs["root_dirichlet_total_alpha"] = noise[0]
            kwargs["root_dirichlet_noise_fraction"] = noise[1]
        hex_results = hex_session.search(
            [key], (state,), evaluator=hex_stub, search_parity_mode=True, **kwargs
        )
        dense_results = dense_session.search(
            [key], (state,), evaluator=dense_stub, **kwargs
        )
        h = hex_results[0]
        d = dense_results[0]
        for field in ("action_id", "visits", "visit_policy_count"):
            if h[field] != d[field]:
                mismatches.append((index, field, h[field], d[field]))
        for field in (
            "visit_policy_action_ids_bytes",
            "visit_policy_weights_bytes",
            "root_prior_policy_action_ids_bytes",
            "root_prior_policy_weights_bytes",
        ):
            if bytes(h[field]) != bytes(d[field]):
                mismatches.append((index, field, "bytes differ", ""))
        if abs(h["root_value"] - d["root_value"]) > 1e-6:
            mismatches.append((index, "root_value", h["root_value"], d["root_value"]))
        hex_session.discard(key)
        dense_session.discard(key)
    assert not mismatches, f"{len(mismatches)} mismatches; first 5: {mismatches[:5]}"


@needs_native
def test_lockstep_parity_greedy_no_noise() -> None:
    states = _corpus(100)
    _run_pair(
        states, visits=32, seed=11, temperature=0.0, noise=None, forced_k=0.0,
        root_temp=1.0, fpu_zero_under_noise=True, tss=True, virtual_batch=8,
    )


@needs_native
def test_lockstep_parity_full_exploration_machinery() -> None:
    # Exercises noise, sampling temperature, forced playouts, target pruning,
    # and the fpu-zero-under-noise / root-temp knobs together.
    states = _corpus(60)
    _run_pair(
        states, visits=48, seed=23, temperature=1.0, noise=(10.83, 0.25),
        forced_k=2.0, root_temp=1.1, fpu_zero_under_noise=True, tss=True,
        virtual_batch=16,
    )


@needs_native
def test_lockstep_parity_tss_disabled() -> None:
    states = _corpus(40)
    _run_pair(
        states, visits=32, seed=31, temperature=0.0, noise=None, forced_k=0.0,
        root_temp=1.0, fpu_zero_under_noise=True, tss=False, virtual_batch=8,
    )
