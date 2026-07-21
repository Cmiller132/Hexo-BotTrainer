#!/usr/bin/env python3
"""Exhaust rank-1/2 threat families on up to six abstract cells.

Checks the purely combinatorial half of the direct K2 defender-pair theorem.
"""

from __future__ import annotations

import json


def popcount(x: int) -> int:
    return x.bit_count()


def tau(edges: tuple[int, ...], n: int) -> int | None:
    if not edges:
        return 0
    for x in range(n):
        if all(edge & (1 << x) for edge in edges):
            return 1
    for x in range(n):
        for y in range(x + 1, n):
            cover = (1 << x) | (1 << y)
            if all(edge & cover for edge in edges):
                return 2
    return None


def pairs(edges: tuple[int, ...], n: int) -> set[tuple[int, int]]:
    return {
        (x, y)
        for x in range(n)
        for y in range(x + 1, n)
        if all(edge & ((1 << x) | (1 << y)) for edge in edges)
    }


def components(edges: tuple[int, ...]) -> list[tuple[int, ...]]:
    remaining = set(range(len(edges)))
    out: list[tuple[int, ...]] = []
    while remaining:
        root = remaining.pop()
        comp = {root}
        stack = [root]
        while stack:
            i = stack.pop()
            joined = {j for j in remaining if edges[i] & edges[j]}
            remaining.difference_update(joined)
            comp.update(joined)
            stack.extend(joined)
        out.append(tuple(edges[i] for i in sorted(comp)))
    return out


def common(comp: tuple[int, ...], n: int) -> set[int]:
    return {x for x in range(n) if all(edge & (1 << x) for edge in comp)}


def main() -> None:
    checked = 0
    tau2 = 0
    split = 0
    max_minimum_pairs = 0
    violations: list[dict] = []
    by_n: dict[int, dict[str, int]] = {}
    # n=6 has 2^21 families; n<=5 already exhausts every local shape seen in
    # the measured corpus (union max five), so cap at five for a quick gate.
    for n in range(1, 6):
        possible = tuple(1 << x for x in range(n)) + tuple(
            (1 << x) | (1 << y) for x in range(n) for y in range(x + 1, n)
        )
        local_checked = 0
        local_tau2 = 0
        for family_mask in range(1, 1 << len(possible)):
            edges = tuple(possible[i] for i in range(len(possible)) if family_mask & (1 << i))
            checked += 1
            local_checked += 1
            if tau(edges, n) != 2:
                continue
            tau2 += 1
            local_tau2 += 1
            ps = pairs(edges, n)
            max_minimum_pairs = max(max_minimum_pairs, len(ps))
            comps = components(edges)
            comp_taus = [tau(comp, n) for comp in comps]
            reasons: list[str] = []
            if sum(int(t) for t in comp_taus if t is not None) != 2:
                reasons.append(f"component tau sum {comp_taus}")
            if len(comps) not in (1, 2):
                reasons.append(f"component count {len(comps)}")
            for x, y in ps:
                residual_x = tuple(edge for edge in edges if not edge & (1 << x))
                residual_y = tuple(edge for edge in edges if not edge & (1 << y))
                if tau(residual_x, n) != 1 or tau(residual_y, n) != 1:
                    reasons.append(f"pair {(x,y)} residual tau")
            if len(comps) == 2:
                split += 1
                if sorted(comp_taus) != [1, 1]:
                    reasons.append(f"split component taus {comp_taus}")
                expected = {
                    tuple(sorted((x, y)))
                    for x in common(comps[0], n)
                    for y in common(comps[1], n)
                    if x != y
                }
                if ps != expected:
                    reasons.append("split pair product mismatch")
            if reasons:
                violations.append({"n": n, "edges": edges, "reasons": reasons})
                break
        by_n[n] = {"families": local_checked, "tau2": local_tau2}
        if violations:
            break
    print(
        json.dumps(
            {
                "checked_nonempty_families": checked,
                "tau2_families": tau2,
                "split_tau2_families": split,
                "max_minimum_pairs": max_minimum_pairs,
                "by_n": by_n,
                "violations": violations,
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(1 if violations else 0)


if __name__ == "__main__":
    main()
