"""Diff two aligned MCTS runs: visit-policy total-variation + argmax agreement."""

from __future__ import annotations

import json
import sys


def tv_distance(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def argmax(p: dict):
    return max(p.items(), key=lambda kv: (kv[1], -kv[0]))[0] if p else None


def main() -> None:
    a = json.load(open(sys.argv[1]))["positions"]
    b = json.load(open(sys.argv[2]))["positions"]
    index = {(x["move"], x["game"]): x["visit_policy"] for x in b}
    tvs, argmatch, n = [], 0, 0
    for x in a:
        key = (x["move"], x["game"])
        if key not in index:
            continue
        n += 1
        pa, pb = {int(k): v for k, v in x["visit_policy"].items()}, {int(k): v for k, v in index[key].items()}
        tvs.append(tv_distance(pa, pb))
        if argmax(pa) == argmax(pb):
            argmatch += 1
    tvs.sort()
    mean_tv = sum(tvs) / max(len(tvs), 1)
    print(f"positions compared : {n}")
    print(f"visit-argmax agree : {argmatch}/{n} ({100*argmatch/max(n,1):.1f}%)")
    print(f"TV distance mean   : {mean_tv:.4f}")
    print(f"TV distance median : {tvs[len(tvs)//2]:.4f}")
    print(f"TV distance p90/max: {tvs[int(0.9*len(tvs))]:.4f} / {tvs[-1]:.4f}")


if __name__ == "__main__":
    main()
