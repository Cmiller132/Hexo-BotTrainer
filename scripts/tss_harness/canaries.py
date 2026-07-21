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
- loss_detection: V1 goal=loss positions both loss arms proved LOSS. Guards
  the structural fact (SOLVER_NOTES §5) that SolveGoal::Both under the wide
  profile gives the loss attempt ZERO budget — an arm declaring goal=both
  today will honestly FAIL this canary until that is fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .contract import Position, LOSS, WIN, UNKNOWN
from .gates import register_canary

_FIXTURES = Path(__file__).resolve().parent / "sets" / "canaries_v2.jsonl"


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
    # goal pinned: canaries test THEIR feature independent of the arm's goal
    # (a goal=loss arm must not spuriously fail win-fixture canaries).
    on = make_adapter({"shared_fragments": True, "goal": "win"}).solve_sequence(seq)
    off = make_adapter({"shared_fragments": False, "goal": "win"}).solve_sequence(seq)
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
    unbounded = make_adapter({"horizon": 0, "goal": "win"}).solve_sequence(fx)
    bounded = make_adapter({"horizon": 16, "goal": "win"}).solve_sequence(fx)
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
    adapter = make_adapter({"wide": True, "goal": "win"})
    m = adapter.manifest()
    if not m.get("vcf_pair_complete"):
        return False, "wide claimed but manifest lacks vcf_pair_complete"
    fx = _load("wide_win")
    got = adapter.solve_sequence(fx)
    wins = [r for r in got if r.status == WIN and r.verified]
    if len(wins) != len(fx):
        return False, f"wide fixture wins {len(wins)}/{len(fx)}"
    return True, f"manifest wide + {len(wins)}/{len(fx)} fixture wins"


@register_canary("group2")
def group2_canary(make_adapter) -> tuple[bool, str]:
    """The manifest must echo group2 truthfully in both directions, and the
    fixture battery must return IDENTICAL verified verdicts with the flag on
    and off: the feature reduces search work, it must not change verdicts.
    (The Rust side guarantees this structurally — a failed Group-2 attempt
    re-solves with the selector off — so a mismatch here is an engine bug.)"""
    fx = _load("wide_win") + _load("loss_pos")
    on_adapter = make_adapter({"group2": True, "goal": "both"})
    off_adapter = make_adapter({"group2": False, "goal": "both"})
    m_on = on_adapter.manifest()
    m_off = off_adapter.manifest()
    if m_on.get("group2") is not True:
        return False, f"group2 ON but manifest echoes {m_on.get('group2')!r}"
    if m_off.get("group2") is not False:
        return False, f"group2 OFF but manifest echoes {m_off.get('group2')!r}"
    on = on_adapter.solve_sequence(fx)
    off = off_adapter.solve_sequence(fx)
    mismatches = [
        (a.pos_id, a.status, a.verified, b.status, b.verified)
        for a, b in zip(on, off)
        if (a.status, a.verified) != (b.status, b.verified)
    ]
    if mismatches:
        return False, f"verdicts diverge with group2 on/off: {mismatches[:4]}"
    vf = sum(r.verify_failed for r in on)
    if vf:
        return False, f"group2 arm reported {vf} verifier failures"
    return True, (
        f"manifest truthful both ways; {len(fx)} fixture verdicts identical "
        f"on/off, zero verify failures"
    )


@register_canary("loss_detection")
def loss_detection_canary(make_adapter) -> tuple[bool, str]:
    """An arm claiming loss detection (goal loss/both) must prove the known
    V1 losses as verified LOSS with ITS OWN goal setting; a win-goal arm must
    return zero loss verdicts on the same fixtures (no bleed). NOTE: goal=
    both under the wide profile currently allocates the loss attempt zero
    budget (tss_solver.rs solve_goal budget split) — a both arm failing here
    is the harness telling the truth, not a fixture problem."""
    fx = _load("loss_pos")
    claimed = make_adapter({}).solve_sequence(fx)   # arm's own goal
    off = make_adapter({"goal": "win"}).solve_sequence(fx)
    losses = [r for r in claimed if r.status == LOSS and r.verified]
    bleed = [r for r in off if r.status == LOSS]
    if len(losses) != len(fx):
        return False, (
            f"loss detection claimed but proved {len(losses)}/{len(fx)} "
            f"known losses (goal budget never reaches the loss attempt?)"
        )
    if bleed:
        return False, f"goal=win arm returned {len(bleed)} loss verdicts"
    return True, f"{len(losses)}/{len(fx)} known losses proven, win-arm clean"
