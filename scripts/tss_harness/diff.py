"""Paired comparison between two runs (or two arms of one run). PLAN §5.

Churn is never netted: upgrades and downgrades are separate lists. Cost
deltas are reported only when both sides share an architecture (contract
rule). Significance: McNemar exact-style readout on the paired
decided/undecided table so small coverage deltas are labeled honestly.
"""

from __future__ import annotations

from math import comb
from typing import Any

from .contract import LOSS, UNKNOWN, WIN, SolveRecord


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar on the discordant pair counts
    (b = A-only successes, c = B-only successes)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def compare(
    a: list[SolveRecord],
    b: list[SolveRecord],
    *,
    label_a: str = "A",
    label_b: str = "B",
    same_architecture: bool = True,
) -> dict[str, Any]:
    by_b = {r.pos_id: r for r in b}
    upgrades, downgrades, flips = [], [], []
    a_only_decided = b_only_decided = 0
    cost_delta_total = 0
    n_paired = 0
    for ra in a:
        rb = by_b.get(ra.pos_id)
        if rb is None:
            continue
        n_paired += 1
        a_dec = ra.status in (WIN, LOSS) and ra.verified
        b_dec = rb.status in (WIN, LOSS) and rb.verified
        if a_dec and not b_dec:
            a_only_decided += 1
            downgrades.append({"pos_id": ra.pos_id, "a": ra.status, "b": rb.status})
        elif b_dec and not a_dec:
            b_only_decided += 1
            upgrades.append({"pos_id": ra.pos_id, "a": ra.status, "b": rb.status})
        elif a_dec and b_dec and ra.status != rb.status:
            # WIN on one side, LOSS on the other: both verified => one of the
            # engines is unsound. This is a FATAL soundness alarm, not churn.
            flips.append({"pos_id": ra.pos_id, "a": ra.status, "b": rb.status})
        if same_architecture:
            cost_delta_total += rb.cost - ra.cost

    p = mcnemar_p(a_only_decided, b_only_decided)
    out: dict[str, Any] = {
        "label_a": label_a,
        "label_b": label_b,
        "paired": n_paired,
        "coverage_a": sum(
            1 for r in a if r.status in (WIN, LOSS) and r.verified
        ),
        "coverage_b": sum(
            1 for r in b if r.status in (WIN, LOSS) and r.verified
        ),
        "upgrades_b_over_a": upgrades,
        "downgrades_b_under_a": downgrades,
        "verified_contradictions": flips,   # nonempty = soundness incident
        "discordant": {"a_only": a_only_decided, "b_only": b_only_decided},
        "mcnemar_p": p,
        "significant": p < 0.05,
    }
    if same_architecture:
        out["cost_delta_total"] = cost_delta_total
    else:
        out["cost_delta_total"] = None  # cross-architecture: wall Pareto only
    return out


def unknown_wall_share(records: list[SolveRecord], cap: int) -> dict[str, Any]:
    """The V1 §10 economics readout, standardized: where does wall sit."""
    total = sum(r.wall_nanos for r in records) or 1
    unk = [r for r in records if r.status == UNKNOWN]
    capb = [r for r in unk if r.cost >= cap]
    return {
        "total_wall_s": total / 1e9,
        "unknown_share": sum(r.wall_nanos for r in unk) / total,
        "cap_bound_n": len(capb),
        "cap_bound_share": sum(r.wall_nanos for r in capb) / total,
    }
