"""Feature canaries — fixtures where an enabled feature MUST visibly fire.

Both directions are enforced by the runner: an arm declaring a feature must
see its canary fire; an arm NOT declaring it must see it NOT fire (always-on
bleed). Fixtures live in sets/canaries_v1.jsonl (built by sets.py from the
frozen V1 data — positions where V1 measured the behavior directly, so every
canary is anchored to an observed firing, not a hoped-for one).

Canary sources (V1/rerun evidence, SOLVER_NOTES):
- warmth: game-28 ply sequence where the fragment rerun measured store
  engagement (71->2 nodes at sp_28_p47); detection via the now-real
  stats_fragment_* counters from the persistent solver.
- unbounded_horizon: a V1 position proven WIN by the unbounded arm at cert
  depth > 16 where the h16 arm returned Unknown.
- wide: V1 paired dominance — a position the wide profile proves that the
  narrow profile cannot... narrow is not reachable through this adapter
  (wholesale-wide port), so the wide canary instead asserts the manifest
  echoes vcf_pair_complete=True AND a known wide-proven fixture solves.
"""

from __future__ import annotations

import json
from pathlib import Path

from .contract import Position, WIN, UNKNOWN
from .gates import register_canary

_FIXTURES = Path(__file__).resolve().parent / "sets" / "canaries_v1.jsonl"


def _load(kind: str) -> list[Position]:
    out = []
    for line in open(_FIXTURES):
        row = json.loads(line)
        if row["canary"] == kind:
            out.append(Position(
                pos_id=row["pos_id"], source="canary",
                moves=tuple(row["moves"]), meta=row.get("meta", {}),
            ))
    if not out:
        raise FileNotFoundError(f"no {kind!r} fixtures in {_FIXTURES}")
    return out


@register_canary("warmth")
def warmth_canary(make_adapter) -> tuple[bool, str]:
    """Fragment store must import+lookup on the warm sequence when the arm
    claims warmth, and stay at exactly zero when it does not."""
    seq = _load("warmth_sequence")
    on = make_adapter({"shared_fragments": True}).solve_sequence(seq)
    off = make_adapter({"shared_fragments": False}).solve_sequence(seq)
    imports_on = sum(r.counters.get("stats_fragment_imports", 0) for r in on)
    lookups_on = sum(r.counters.get("stats_fragment_lookups", 0) for r in on)
    any_off = sum(
        r.counters.get("stats_fragment_imports", 0)
        + r.counters.get("stats_fragment_lookups", 0)
        for r in off
    )
    if imports_on <= 0 or lookups_on <= 0:
        return False, (
            f"warmth claimed but store never engaged "
            f"(imports={imports_on}, lookups={lookups_on})"
        )
    if any_off:
        return False, f"warmth OFF but fragment counters nonzero ({any_off})"
    return True, f"imports={imports_on} lookups={lookups_on}, off=0"


@register_canary("unbounded_horizon")
def unbounded_horizon_canary(make_adapter) -> tuple[bool, str]:
    """The deep-win fixture must be WIN under horizon=0 and Unknown under
    h16 — proves the horizon parameter actually reaches the search."""
    fx = _load("deep_win")
    unbounded = make_adapter({"horizon": 0}).solve_sequence(fx)
    bounded = make_adapter({"horizon": 16}).solve_sequence(fx)
    deep_ok = [r for r in unbounded if r.status == WIN and r.verified]
    shallow_ok = [r for r in bounded if r.status == UNKNOWN]
    if not deep_ok:
        return False, "deep-win fixture not proven under unbounded horizon"
    if len(shallow_ok) != len(fx):
        return False, "h16 arm decided the deep-win fixture (horizon inert?)"
    return True, f"{len(deep_ok)}/{len(fx)} deep wins unbounded-only"


@register_canary("wide")
def wide_canary(make_adapter) -> tuple[bool, str]:
    """Manifest must echo the wide width profile and the wide-proven fixture
    must solve. (Narrow is unreachable in this engine build — wholesale-wide
    port — so the OFF direction is manifest-only.)"""
    adapter = make_adapter({"wide": True})
    m = adapter.manifest()
    if not m.get("vcf_pair_complete"):
        return False, "wide claimed but manifest lacks vcf_pair_complete"
    fx = _load("wide_win")
    got = adapter.solve_sequence(fx)
    wins = [r for r in got if r.status == WIN and r.verified]
    if len(wins) != len(fx):
        return False, f"wide fixture wins {len(wins)}/{len(fx)}"
    return True, f"manifest wide + {len(wins)}/{len(fx)} fixture wins"
