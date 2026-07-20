"""Harness gates — each traces to a real measurement incident (PLAN §3).

A gate returns a GateResult; HARD failures abort/invalidate the run. The
self-test suite (selftest.py) deliberately violates every gate and requires
the violation to be caught: a gate that cannot fail is not a gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .contract import (
    LOSS,
    UNKNOWN,
    VERDICTS,
    WIN,
    ArmSpec,
    Position,
    SolveRecord,
    manifest_subset_mismatches,
)


@dataclass
class GateResult:
    gate: str
    passed: bool
    hard: bool
    detail: str = ""

    @property
    def fatal(self) -> bool:
        return self.hard and not self.passed


# --------------------------------------------------------------------------- #
# Canary registry: feature name -> canary callable. A canary receives the
# adapter (feature ON arm), a paired cold/off adapter factory when needed,
# and returns (fired: bool, detail). The runner enforces BOTH directions:
# declared+enabled => canary must fire; feature absent => must NOT fire.
# --------------------------------------------------------------------------- #
CanaryFn = Callable[[Any], tuple[bool, str]]
_CANARIES: dict[str, CanaryFn] = {}


def register_canary(feature: str):
    def deco(fn: CanaryFn) -> CanaryFn:
        _CANARIES[feature] = fn
        return fn
    return deco


def canary_for(feature: str) -> CanaryFn | None:
    return _CANARIES.get(feature)


def gate_features_have_canaries(arm: ArmSpec) -> GateResult:
    """PLAN §3.2 binding rule: an arm may not declare a feature that has no
    canary — you cannot claim what cannot be checked."""
    missing = [f for f in arm.features if f not in _CANARIES]
    return GateResult(
        gate="features_have_canaries",
        passed=not missing,
        hard=True,
        detail=f"features without canaries: {missing}" if missing else "",
    )


def gate_manifest(arm: ArmSpec, echoed: dict[str, Any]) -> GateResult:
    """Warmth env-gate incident: requested-but-not-effective config aborts."""
    problems = manifest_subset_mismatches(arm.declared, echoed)
    return GateResult(
        gate="manifest_subset_match",
        passed=not problems,
        hard=True,
        detail="; ".join(problems),
    )


def gate_soundness(records: list[SolveRecord]) -> GateResult:
    """deep_verify_failed == 0 everywhere; verdict vocabulary closed; a
    decided verdict must be verified to count."""
    vf = sum(r.verify_failed for r in records)
    bad_status = [r.pos_id for r in records if r.status not in VERDICTS]
    unverified = [
        r.pos_id for r in records if r.status in (WIN, LOSS) and not r.verified
    ]
    problems = []
    if vf:
        problems.append(f"verify_failed total = {vf}")
    if bad_status:
        problems.append(f"unknown status vocabulary: {bad_status[:5]}")
    if unverified:
        problems.append(f"decided-but-unverified: {unverified[:5]}")
    return GateResult(
        gate="soundness",
        passed=not problems,
        hard=True,
        detail="; ".join(problems),
    )


def gate_ground_truth(
    positions: list[Position], records: list[SolveRecord]
) -> GateResult:
    """Puzzle-set hard gate: losing a known verdict or contradicting a label
    fails the run. Unknown on a labeled position is a COVERAGE miss (soft,
    counted in the report) unless the label says the position is inside the
    arm's claimed reach (label key 'must_solve': true)."""
    by_id = {r.pos_id: r for r in records}
    contradictions = []
    lost_must = []
    for p in positions:
        if not p.labels or "verdict" not in p.labels:
            continue
        rec = by_id.get(p.pos_id)
        if rec is None:
            continue
        want = p.labels["verdict"]
        if rec.status in (WIN, LOSS) and rec.status != want:
            contradictions.append(f"{p.pos_id}: label {want}, got {rec.status}")
        elif rec.status == UNKNOWN and p.labels.get("must_solve"):
            lost_must.append(p.pos_id)
    problems = []
    if contradictions:
        problems.append(f"label contradictions: {contradictions[:5]}")
    if lost_must:
        problems.append(f"lost must-solve verdicts: {lost_must[:5]}")
    return GateResult(
        gate="ground_truth",
        passed=not problems,
        hard=True,
        detail="; ".join(problems),
    )


def gate_determinism(
    first: list[SolveRecord], second: list[SolveRecord]
) -> GateResult:
    """Re-solve of a shard must be bit-identical in verdict and cost (wall
    excluded — wall is never load-bearing)."""
    diffs = []
    by_id = {r.pos_id: r for r in second}
    for a in first:
        b = by_id.get(a.pos_id)
        if b is None:
            diffs.append(f"{a.pos_id}: missing in re-run")
        elif (a.status, a.cost) != (b.status, b.cost):
            diffs.append(
                f"{a.pos_id}: {a.status}/{a.cost} vs {b.status}/{b.cost}"
            )
    return GateResult(
        gate="determinism",
        passed=not diffs,
        hard=True,
        detail="; ".join(diffs[:5]),
    )


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)

    def add(self, result: GateResult) -> None:
        self.results.append(result)

    @property
    def fatal(self) -> bool:
        return any(r.fatal for r in self.results)

    def to_json(self) -> list[dict[str, Any]]:
        return [
            {
                "gate": r.gate,
                "passed": r.passed,
                "hard": r.hard,
                "detail": r.detail,
            }
            for r in self.results
        ]
