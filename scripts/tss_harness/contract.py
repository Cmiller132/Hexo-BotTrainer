"""The solver CONTRACT — the only surface the harness core depends on.

Design: docs/PLAN_TSS_HARNESS.md §1. Everything above this contract (sets,
gates, archive, diff, reports) is solver-agnostic and frozen; everything
solver-specific (config vocabulary, canaries, counter semantics) lives in an
adapter implementing this contract. The strict verifier is the fixed point:
a verdict counts for coverage only when ``verified`` is True (independent
cert replay inside the adapter's engine, pinned verifier version).

Anti-cheat invariants enforced by the harness core (gates.py):
- Manifest subset-match: every key an arm DECLARES must be echoed by the
  adapter's manifest with the matching effective value (warmth env-gate
  incident, SOLVER_NOTES §3).
- No declared feature without a registered canary (you cannot claim what
  cannot be checked).
- Node/cost counters are compared only within an architecture; cross-
  architecture comparisons use verified coverage at matched wall tiers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

SCHEMA_VERSION = 1

# Verdict vocabulary — closed set, part of the contract.
WIN = "win"
LOSS = "loss"
UNKNOWN = "unknown"
VERDICTS = (WIN, LOSS, UNKNOWN)


@dataclass(frozen=True)
class Position:
    """One frozen benchmark position. ``moves`` is the axial move list from
    the empty board (the only state encoding sets are allowed to use, so any
    adapter can rebuild the state); ``labels`` carries ground truth when the
    position belongs to a puzzle set ({"verdict": ..., "provenance": ...})."""

    pos_id: str
    source: str            # selfplay | human | forcing | fixture | atlas
    moves: tuple[str, ...]
    meta: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] | None = None


@dataclass
class SolveRecord:
    """One solve result in contract vocabulary. ``counters`` is an OPEN dict
    (adapter-specific vocabulary, archived verbatim, diffed on shared keys);
    only the named fields are load-bearing for gates."""

    pos_id: str
    status: str            # in VERDICTS
    verified: bool         # independent verifier replayed the cert
    verify_failed: int     # FATAL if nonzero anywhere
    wall_nanos: int
    cost: int              # adapter's primary deterministic budget currency
    counters: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "pos_id": self.pos_id,
            "status": self.status,
            "verified": self.verified,
            "verify_failed": self.verify_failed,
            "wall_nanos": self.wall_nanos,
            "cost": self.cost,
            "counters": self.counters,
        }


@dataclass(frozen=True)
class ArmSpec:
    """One configuration under test. ``declared`` is the intent dict the
    manifest gate subset-matches against the adapter's echo; ``features`` is
    the list of claimed features — each MUST have a registered canary."""

    name: str
    adapter: str           # adapter registry key
    config: dict[str, Any] = field(default_factory=dict)
    declared: dict[str, Any] = field(default_factory=dict)
    features: tuple[str, ...] = ()


@runtime_checkable
class SolverAdapter(Protocol):
    """The pluggable solver surface. Implementations wrap one engine build +
    one config; a dramatically refactored solver ships a new adapter and the
    harness core never changes."""

    name: str
    architecture: str      # cost counters comparable only within this key

    def manifest(self) -> dict[str, Any]:
        """Effective configuration, self-described. MUST come from the
        engine's own resolution path (echo), never re-derived in Python."""
        ...

    def solve_sequence(self, positions: list[Position]) -> list[SolveRecord]:
        """Solve in order with whatever persistence the config implies.
        Sequence-based so warmth-like semantics are expressible; adapters
        without cross-solve state may ignore ordering."""
        ...


def stable_hash(obj: Any) -> str:
    """Canonical content hash used for set pinning and fingerprints."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def manifest_subset_mismatches(
    declared: dict[str, Any], echoed: dict[str, Any]
) -> list[str]:
    """The manifest gate predicate: every declared key must be present in
    the echo with an equal value. Returns human-readable mismatch strings
    (empty = pass). Extra echoed keys are fine (archived opaquely)."""
    problems = []
    for key, want in declared.items():
        if key not in echoed:
            problems.append(f"declared {key!r} absent from manifest echo")
        elif echoed[key] != want:
            problems.append(
                f"declared {key}={want!r} but effective {key}={echoed[key]!r}"
            )
    return problems
