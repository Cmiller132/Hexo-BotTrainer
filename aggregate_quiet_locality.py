#!/usr/bin/env python3
"""Aggregate NQ2 quiet-locality specimen JSONL into report tables.

Usage: python aggregate_quiet_locality.py FILE [FILE ...]

Each input record = one quiet placement. D6 images of a specimen (spec_id
contains "_d6_") are isometric duplicates and are excluded from the DISTINCT
population (they reproduce identical measures up to symmetry -- reported as a
covariance check).
"""
import json
import sys
from collections import Counter


def load(paths):
    recs = []
    for p in paths:
        try:
            fh = open(p, "r", encoding="utf-8")
        except FileNotFoundError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    if n % 2:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def hist(xs, hi=10):
    c = Counter()
    for x in xs:
        if x is None or x < 0:
            c["na"] += 1
        elif x >= hi:
            c[f">={hi}"] += 1
        else:
            c[x] += 1
    parts = []
    if c.get("na"):
        parts.append(f"na:{c['na']}")
    for k in range(0, hi):
        if c.get(k):
            parts.append(f"{k}:{c[k]}")
    if c.get(f">={hi}"):
        parts.append(f">={hi}:{c[f'>={hi}']}")
    return " ".join(parts)


def frac(pred, pop):
    m = len(pop)
    n = sum(1 for r in pop if pred(r))
    return n, m, (100.0 * n / m if m else 0.0)


def report(pop, label):
    m = len(pop)
    print(f"\n===== {label}  (n={m} quiet placements) =====")
    if m == 0:
        return
    print("by_source:", dict(Counter(r["source"] for r in pop)))
    strict = [r for r in pop if r.get("strict_quiet")]
    print(f"strict_quiet (no >=4 threat after turn): {len(strict)}   "
          f"loose_quiet (>=4 threat, defender slack): {m - len(strict)}")
    print("by_subclass:", dict(Counter(r["subclass"] for r in pop)))
    print("stone_role:", dict(Counter(r["stone_role"] for r in pop)))
    print()
    for field in ("d_stone", "d_used", "d_two"):
        xs = [r[field] for r in pop]
        valid = [x for x in xs if x >= 0]
        print(f"  {field:8s} median={median(valid)!s:>4}  hist[{hist(xs)}]")
    print()
    # Key locality predicates.
    for desc, pred in [
        ("d_stone <= 1 (adjacent to attacker stone)", lambda r: 0 <= r["d_stone"] <= 1),
        ("d_stone <= 2", lambda r: 0 <= r["d_stone"] <= 2),
        ("d_used  <= 1 (touches/near a served family)", lambda r: 0 <= r["d_used"] <= 1),
        ("d_used  <= 2", lambda r: 0 <= r["d_used"] <= 2),
        ("d_used  <= 4", lambda r: 0 <= r["d_used"] <= 4),
        ("joins a live attacker window (join_live hit)", lambda r: r["candidates"]["join_live"]["hit"]),
        ("d_two   <= 3 (two served families near)", lambda r: 0 <= r["d_two"] <= 3),
    ]:
        n, mm, pc = frac(pred, pop)
        print(f"  {desc:48s}: {n}/{mm} = {pc:.1f}%")
    print()
    # Candidate C(P) coverage + size (median across nodes).
    names = list(pop[0]["candidates"].keys())
    med_legal = median([r["node_full_legal"] for r in pop])
    print(f"  {'candidate':>14} {'coverage':>9} {'medianC':>8} {'med_legal':>10} {'shrink':>7}")
    for nm in names:
        hits = sum(1 for r in pop if r["candidates"][nm]["hit"])
        sizes = [r["candidates"][nm]["size"] for r in pop]
        medc = median(sizes)
        shrink = (medc / med_legal) if med_legal else 0
        print(f"  {nm:>14} {100*hits/m:>8.1f}% {medc:>8} {med_legal:>10} {shrink:>6.3f}x")


def main():
    recs = load(sys.argv[1:])
    print(f"# loaded {len(recs)} quiet-placement records from {len(sys.argv)-1} file(s)")

    distinct = [r for r in recs if "_d6_" not in r["spec_id"]]
    d6 = [r for r in recs if "_d6_" in r["spec_id"]]
    print(f"# distinct (non-D6): {len(distinct)}   D6 covariance copies: {len(d6)}")

    report(distinct, "DISTINCT (all sources)")
    for src in ("specimen", "human", "leafwidth"):
        sub = [r for r in distinct if r["source"] == src]
        if sub:
            report(sub, f"source={src}")

    # Per-specimen one-liner table.
    print("\n===== per quiet placement (distinct) =====")
    for r in sorted(distinct, key=lambda r: (r["source"], r["spec_id"], r["stone_role"])):
        cj = r["candidates"]
        print(f"  {r['source']:9s} {r['spec_id']:24s} {str(tuple(r['placement'])):9s} "
              f"{r['stone_role']:6s} strict={int(r.get('strict_quiet',False))} "
              f"d_stone={r['d_stone']} d_used={r['d_used']} d_two={r['d_two']} "
              f"nfam={r['n_served_families']} sub={r['subclass']:10s} "
              f"legal={r['node_full_legal']} adj1={cj['adj_stone_k1']['size']} "
              f"joinadj1={cj['join_adj1']['size']}")

    # Outliers among DISTINCT: farthest d_used and farthest d_stone.
    print("\n===== outliers (d_used desc, then d_stone desc) =====")
    for r in sorted(distinct, key=lambda r: (-r["d_used"], -r["d_stone"]))[:12]:
        print(f"  {r['source']:9s} {r['spec_id']:24s} place={tuple(r['placement'])} "
              f"d_used={r['d_used']} d_stone={r['d_stone']} d_two={r['d_two']} "
              f"strict={int(r.get('strict_quiet',False))} sub={r['subclass']}")


if __name__ == "__main__":
    main()
