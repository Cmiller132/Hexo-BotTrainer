"""Bounded serve graph-key set (_GraphCache ladder / hit threshold / key cap).

Every captured CUDA graph permanently pins host-side exec structures plus
statics and pool VRAM. Under the original uniform step-4 batch ladder a warmed
selfplay serve captured a graph for nearly every (B_bucket, Npad) remainder
shape — hundreds of keys, ~5-7GB of host RSS per driver life — which pushed
the warm selfplay->train boundary past the guest memory ceiling (an earlyoom
kill every epoch) and contended with the concurrent eval's own device use
(the ep20/ep30 "device not ready" partial evals). The bounds under test:

  1. B_LADDER — bucket() maps every group size 1..MAX_B onto the bounded
     ladder: member values map to themselves, results are >= g, monotone,
     and out-of-range sizes return None (caller falls back).
  2. HEXFIELD_GRAPH_MIN_HITS — a key must recur MIN_HITS times before it is
     captured; earlier sightings return None (compiled-path fallback).
  3. HEXFIELD_GRAPH_MAX_KEYS — a hard cap on captured graphs: past it no new
     key captures (existing entries keep replaying); 0 = unlimited.
  4. Failed keys stay failed: a capture exception marks the key and no
     recapture is attempted.

CPU-only: torch.cuda.graph_pool_handle is stubbed out and _capture is
monkeypatched, so no CUDA device (and no model) is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
for _p in ("hexo_engine/python", "hexfield_eq/python"):
    _src = str(_REPO / "packages" / _p)
    if _src not in sys.path:
        sys.path.insert(0, _src)

import torch  # noqa: E402

from hexfield_eq.inference import _GraphCache  # noqa: E402


@pytest.fixture()
def make_cache(monkeypatch):
    """Build a _GraphCache with CUDA and the model factored out.

    graph_pool_handle is stubbed (CPU-only interpreter), and _capture is
    replaced by a counter that returns a unique sentinel entry per key.
    """

    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)

    def _make(*, min_hits=None, max_keys=None):
        if min_hits is not None:
            monkeypatch.setenv("HEXFIELD_GRAPH_MIN_HITS", str(min_hits))
        else:
            monkeypatch.delenv("HEXFIELD_GRAPH_MIN_HITS", raising=False)
        if max_keys is not None:
            monkeypatch.setenv("HEXFIELD_GRAPH_MAX_KEYS", str(max_keys))
        else:
            monkeypatch.delenv("HEXFIELD_GRAPH_MAX_KEYS", raising=False)
        cache = _GraphCache(
            fwd=None, autocast_on=False, device=torch.device("cpu")
        )
        captures: list[tuple] = []

        def _fake_capture(key, npad, request_ml):
            captures.append(key)
            return {"key": key, "npad": npad, "ml": request_ml}

        cache._capture = _fake_capture
        return cache, captures

    return _make


# --------------------------------------------------------------------------- #
# 1. The bucket ladder.
# --------------------------------------------------------------------------- #
def test_ladder_covers_every_group_size(make_cache):
    cache, _ = make_cache()
    prev = 0
    for g in range(1, cache.MAX_B + 1):
        bb = cache.bucket(g)
        assert bb is not None
        assert bb in cache.B_LADDER
        assert bb >= g, f"bucket({g}) = {bb} would truncate real rows"
        assert bb >= prev, "bucket must be monotone in g"
        prev = bb


def test_ladder_members_map_to_themselves(make_cache):
    cache, _ = make_cache()
    for value in cache.B_LADDER:
        assert cache.bucket(value) == value


def test_ladder_is_bounded_and_rejects_out_of_range(make_cache):
    cache, _ = make_cache()
    assert cache.bucket(cache.MAX_B) == cache.MAX_B
    assert cache.bucket(cache.MAX_B + 1) is None
    assert cache.bucket(0) is None
    assert cache.bucket(-3) is None
    # The whole point: the reachable B set is the ladder, not per-4 buckets.
    reachable = {cache.bucket(g) for g in range(1, cache.MAX_B + 1)}
    assert reachable == set(cache.B_LADDER)


def test_ladder_padding_waste_is_bounded(make_cache):
    """Relative row padding (bb - g) / g stays under ~27% for g >= 4 (padded
    rows ride the idle GPU; this pins the trade from growing silently)."""
    cache, _ = make_cache()
    worst = max(
        (cache.bucket(g) - g) / g for g in range(4, cache.MAX_B + 1)
    )
    assert worst <= 0.27, f"worst-case pad waste {worst:.2%}"


# --------------------------------------------------------------------------- #
# 2. Hit threshold.
# --------------------------------------------------------------------------- #
def test_min_hits_defers_capture_until_key_recurs(make_cache):
    cache, captures = make_cache(min_hits=2)
    # First sighting: not captured, caller falls back.
    assert cache.entry_for(16, 384, True) is None
    assert captures == []
    # Second sighting of the SAME key: captured.
    entry = cache.entry_for(16, 384, True)
    assert entry is not None and entry["key"] == (16, 384, True)
    assert len(captures) == 1
    # Third sighting: served from the cache, no recapture.
    assert cache.entry_for(16, 384, True) is entry
    assert len(captures) == 1


def test_min_hits_counts_per_key_not_globally(make_cache):
    cache, captures = make_cache(min_hits=2)
    assert cache.entry_for(16, 384, True) is None
    # A different key does not inherit the first key's count.
    assert cache.entry_for(16, 448, True) is None
    assert cache.entry_for(32, 384, True) is None
    assert captures == []
    assert cache.entry_for(16, 448, True) is not None
    assert len(captures) == 1


def test_min_hits_one_restores_capture_on_first_sight(make_cache):
    cache, captures = make_cache(min_hits=1)
    assert cache.entry_for(16, 384, True) is not None
    assert len(captures) == 1


def test_bucketed_sizes_share_one_key(make_cache):
    """Group sizes that ladder to the same bucket count toward ONE key."""
    cache, captures = make_cache(min_hits=2)
    assert cache.bucket(33) == cache.bucket(40) == 40
    assert cache.entry_for(33, 384, False) is None  # hit 1 for (40, 384)
    assert cache.entry_for(40, 384, False) is not None  # hit 2 -> capture
    assert len(captures) == 1


# --------------------------------------------------------------------------- #
# 3. Key cap.
# --------------------------------------------------------------------------- #
def test_max_keys_caps_new_captures_but_keeps_existing(make_cache):
    cache, captures = make_cache(min_hits=1, max_keys=2)
    e1 = cache.entry_for(16, 384, True)
    e2 = cache.entry_for(32, 384, True)
    assert e1 is not None and e2 is not None
    # Third key: at the cap, never captures no matter how often it recurs.
    for _ in range(10):
        assert cache.entry_for(64, 448, True) is None
    assert len(captures) == 2
    # Existing keys keep replaying.
    assert cache.entry_for(16, 384, True) is e1
    assert cache.entry_for(32, 384, True) is e2


def test_max_keys_zero_is_unlimited(make_cache):
    cache, captures = make_cache(min_hits=1, max_keys=0)
    for bb in cache.B_LADDER:
        assert cache.entry_for(bb, 384, False) is not None
    assert len(captures) == len(cache.B_LADDER)


def test_cap_warns_once(make_cache, caplog):
    cache, _ = make_cache(min_hits=1, max_keys=1)
    assert cache.entry_for(16, 384, True) is not None
    with caplog.at_level("WARNING", logger="hexfield_eq.inference"):
        assert cache.entry_for(32, 384, True) is None
        assert cache.entry_for(64, 384, True) is None
    warnings = [r for r in caplog.records if "graph-key cap" in r.getMessage()]
    assert len(warnings) == 1


# --------------------------------------------------------------------------- #
# 4. Failure latching + stats.
# --------------------------------------------------------------------------- #
def test_failed_capture_latches_and_never_retries(make_cache):
    cache, _ = make_cache(min_hits=1)
    calls = []

    def _boom(key, npad, request_ml):
        calls.append(key)
        raise RuntimeError("capture failed")

    cache._capture = _boom
    assert cache.entry_for(16, 384, True) is None
    assert cache.entry_for(16, 384, True) is None
    assert len(calls) == 1  # no recapture attempt on a failed key
    assert (16, 384, True) in cache._failed


def test_failed_key_beats_cached_entry(make_cache):
    """_failed takes precedence over _graphs: once a key is latched failed
    (e.g. a post-capture replay failure in run_group), entry_for must never
    hand back a cached (broken) graph for it."""
    cache, _ = make_cache(min_hits=1)
    entry = cache.entry_for(16, 384, True)
    assert entry is not None
    cache._failed.add((16, 384, True))
    assert cache.entry_for(16, 384, True) is None


def test_run_group_replay_failure_latches_and_evicts(make_cache):
    """A failure AFTER capture (inside run_group's fill/replay) latches the
    key into _failed AND evicts the broken entry from _graphs, so later
    requests fall back to the compiled path instead of replaying it."""
    cache, _ = make_cache(min_hits=1)
    # The fake capture returns an entry without "in"/"out"/"graph", so
    # run_group's static-fill raises exactly like a broken replay would.
    d_feats = torch.zeros(4, 384, 1)
    out = cache.run_group(d_feats, None, None, None, 4, True)
    assert out is None
    key = (4, 384, True)
    assert key in cache._failed
    assert key not in cache._graphs
    # And the key stays dead (compiled-path fallback, no recapture).
    assert cache.entry_for(4, 384, True) is None


def test_malformed_env_knobs_fall_back_to_defaults(monkeypatch):
    """Evaluator construction must survive garbage tuning knobs (the driver
    auto-restarts and imports this module with whatever env it inherits)."""
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setenv("HEXFIELD_GRAPH_MIN_HITS", "banana")
    monkeypatch.setenv("HEXFIELD_GRAPH_MAX_KEYS", "-5")
    cache = _GraphCache(fwd=None, autocast_on=False, device=torch.device("cpu"))
    assert cache._min_hits == 2  # default
    assert cache._max_keys == 0  # negative clamps to unlimited, not disable


def test_stats_reports_capture_set(make_cache):
    cache, _ = make_cache(min_hits=2, max_keys=8)
    cache.entry_for(16, 384, True)  # seen once, not captured
    cache.entry_for(32, 384, True)
    cache.entry_for(32, 384, True)  # captured
    s = cache.stats()
    assert s["captured"] == 1
    assert s["keys_seen"] == 2
    assert s["max_keys"] == 8
    assert s["min_hits"] == 2
