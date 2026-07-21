"""Root-position prevalence probe for deadline/no-turn certificate ideas.

This is deliberately independent of the Rust solver.  It reconstructs only
the static Hexo window facts needed by two candidate certificates:

* the exact two-carrier necessary condition for a pair-complete forcing turn;
* a monotone colour-possibility closure which, when finite, confines every
  future vcf_pair_complete placement to a finite set.

The closure is an over-approximation.  Hitting a cap means "not certified",
never evidence that the true closure is infinite.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


AXES = ((1, 0), (0, 1), (1, -1))


def owner_at(index: int) -> int:
    if index == 0:
        return 0
    return 1 if ((index - 1) // 2) % 2 == 0 else 0


def current_player(nstones: int) -> int:
    if nstones == 0:
        return 0
    return 1 if ((nstones - 1) // 2) % 2 == 0 else 0


def windows_through(cell):
    q, r = cell
    for dq, dr in AXES:
        for offset in range(6):
            yield (q - offset * dq, r - offset * dr, dq, dr)


def window_cells(key):
    q, r, dq, dr = key
    return tuple((q + i * dq, r + i * dr) for i in range(6))


def touched_windows(occupied):
    keys = set()
    for cell in occupied:
        keys.update(windows_through(cell))
    return keys


def alive_profile(stones, attacker):
    defender = 1 - attacker
    bins = Counter()
    max_alive = 0
    for key in touched_windows(stones):
        cells = window_cells(key)
        ca = sum(stones.get(x) == attacker for x in cells)
        cd = sum(stones.get(x) == defender for x in cells)
        if ca and not cd:
            max_alive = max(max_alive, ca)
            if ca <= 5:
                bins[ca] += 1
    a = bins[1] + 3 * bins[3] + 9 * bins[5]
    b = bins[2] + 3 * bins[4]
    return bins, max_alive, b <= 8 and a * a < 3 * (9 - b) * (9 - b)


def static_features(moves):
    stones = {tuple(cell): owner_at(i) for i, cell in enumerate(moves)}
    claimant = current_player(len(moves))
    opponent = 1 - claimant
    c2 = []
    c3 = []
    bins = Counter()
    defender_threats = 0
    max_alive = 0
    for key in touched_windows(stones):
        cells = window_cells(key)
        ca = sum(stones.get(x) == claimant for x in cells)
        cd = sum(stones.get(x) == opponent for x in cells)
        if ca and not cd:
            max_alive = max(max_alive, ca)
            if ca <= 5:
                bins[ca] += 1
            empties = frozenset(x for x in cells if x not in stones)
            if ca == 2:
                c2.append(empties)
            elif ca == 3:
                c3.append(empties)
        if cd >= 4 and not ca:
            defender_threats += 1

    # A post-pair threat is born from a current c3 window hit at least once,
    # or a current c2 window hit twice.  Test whether any two distinct windows
    # can be activated by the same unordered pair.  If not, every pair leaves
    # <=1 threat and can never have tau=2 (the wide forcing gate).
    jointly_activatable = len(c3) >= 2
    if not jointly_activatable and c3:
        e3 = c3[0]
        jointly_activatable = any(bool(e3 & e2) for e2 in c2)
    duplicate_pair = False
    pair_owner = set()
    if not jointly_activatable:
        for empties in c2:
            ordered = sorted(empties)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    pair = (ordered[i], ordered[j])
                    if pair in pair_owner:
                        duplicate_pair = True
                        jointly_activatable = True
                        break
                    pair_owner.add(pair)
                if jointly_activatable:
                    break
            if jointly_activatable:
                break

    a = bins[1] + 3 * bins[3] + 9 * bins[5]
    b = bins[2] + 3 * bins[4]
    phi_lt_1 = b <= 8 and a * a < 3 * (9 - b) * (9 - b)
    phase = "opening" if not moves else ("first" if len(moves) % 2 else "second")
    _, _, opponent_phi_lt_1 = alive_profile(stones, opponent)
    return {
        "claimant": claimant,
        "max_alive": max_alive,
        "n_c2": len(c2),
        "n_c3": len(c3),
        "defender_threats": defender_threats,
        "carrier_eligible": phase == "first",
        "carrierless": phase == "first" and max_alive < 4 and not jointly_activatable,
        "phi_lt_1": phi_lt_1,
        "opponent_phi_lt_1_at_defender_first": phase == "first" and opponent_phi_lt_1,
        "phi_bins": [bins[i] for i in range(1, 6)],
    }


def line_component_lower_bound(moves):
    """Mandatory 1-D part of every forcing region, or infinity.

    Once an A-alive axis window with >=2 A stones is admitted, A-closure
    fills that window.  Consecutive D-free shifts overlap it in five cells,
    so induction fills the whole D0-delimited component of that axis line.
    """
    stones = {tuple(cell): owner_at(i) for i, cell in enumerate(moves)}
    claimant = current_player(len(moves))
    defenders = {x for x, owner in stones.items() if owner != claimant}
    required = set()
    carrier_count = 0
    for key in touched_windows(stones):
        cells = window_cells(key)
        ca = sum(stones.get(x) == claimant for x in cells)
        cd = sum(stones.get(x) != claimant for x in cells if x in stones)
        if ca < 2 or cd:
            continue
        carrier_count += 1
        q0, r0, dq, dr = key
        if (dq, dr) == (1, 0):
            invariant = r0
            ts = [q for q, r in defenders if r == invariant]
            lo_t, hi_t = q0, q0 + 5
            make = lambda t, inv=invariant: (t, inv)
        elif (dq, dr) == (0, 1):
            invariant = q0
            ts = [r for q, r in defenders if q == invariant]
            lo_t, hi_t = r0, r0 + 5
            make = lambda t, inv=invariant: (inv, t)
        else:
            invariant = q0 + r0
            ts = [q for q, r in defenders if q + r == invariant]
            lo_t, hi_t = q0, q0 + 5
            make = lambda t, inv=invariant: (t, inv - t)
        left = max((t for t in ts if t < lo_t), default=None)
        right = min((t for t in ts if t > hi_t), default=None)
        if left is None or right is None:
            return {"carrier_count": carrier_count, "unbounded": True, "empty_lb": None}
        required.update(make(t) for t in range(left + 1, right))
    return {
        "carrier_count": carrier_count,
        "unbounded": False,
        "empty_lb": len(required - set(stones)),
    }


def possibility_closure(moves, cap):
    stones = {tuple(cell): owner_at(i) for i, cell in enumerate(moves)}
    claimant = current_player(len(moves))
    initial_a = {x for x, owner in stones.items() if owner == claimant}
    initial_d = set(stones) - initial_a
    poss_a = set(initial_a)
    poss_d = set(initial_d)

    # The ideal semantics is Z^2.  We stop rather than mistaking i16 overflow
    # for a finite closure if the abstract expansion reaches the carrier edge.
    overflow = False
    iterations = 0
    while True:
        iterations += 1
        keys = touched_windows(poss_a | poss_d)
        add_a = set()
        add_d = set()
        for key in keys:
            cells = window_cells(key)
            if any(abs(q) > 32767 or abs(r) > 32767 for q, r in cells):
                overflow = True
                break
            a_count = sum(x in poss_a for x in cells)
            d_count = sum(x in poss_d for x in cells)
            # Initial stones are permanent.  Possible future opposing stones
            # are intentionally ignored as blockers, making this an overbound.
            if not any(x in initial_d for x in cells):
                if a_count >= 2:
                    add_a.update(x for x in cells if x not in initial_d)
                if a_count >= 4:
                    add_d.update(x for x in cells if x not in initial_a)
            if not any(x in initial_a for x in cells) and d_count >= 4:
                add_a.update(x for x in cells if x not in initial_d)
        if overflow:
            break
        new_a = add_a - poss_a
        new_d = add_d - poss_d
        if not new_a and not new_d:
            break
        poss_a.update(new_a)
        poss_d.update(new_d)
        if len(poss_a | poss_d) > cap:
            return {
                "closed": False,
                "reason": "cap",
                "size_lb": len(poss_a | poss_d),
                "iterations": iterations,
            }
    if overflow:
        return {
            "closed": False,
            "reason": "coord",
            "size_lb": len(poss_a | poss_d),
            "iterations": iterations,
        }
    region = poss_a | poss_d
    return {
        "closed": True,
        "reason": "fixed_point",
        "size": len(region),
        "empty": len(region - set(stones)),
        "a_size": len(poss_a),
        "d_size": len(poss_d),
        "iterations": iterations,
    }


def forcing_region_closure(moves, cap):
    """Tighter one-region invariant for the actual forcing contract.

    R contains every *future* placement before the first implicit-dispatch
    escape.  Current defender stones need not be in R; they only contribute
    to the possible-defender-threat rule below.
    """
    stones = {tuple(cell): owner_at(i) for i, cell in enumerate(moves)}
    claimant = current_player(len(moves))
    initial_a = {x for x, owner in stones.items() if owner == claimant}
    initial_d = set(stones) - initial_a
    region = set(initial_a)
    iterations = 0
    while True:
        iterations += 1
        keys = touched_windows(region | initial_d)
        additions = set()
        for key in keys:
            cells = window_cells(key)
            if any(abs(q) > 32767 or abs(r) > 32767 for q, r in cells):
                return {
                    "closed": False,
                    "reason": "coord",
                    "size_lb": len(region),
                    "iterations": iterations,
                }
            # Every claimant extension (including G1 after the first stone)
            # and every forced hit lies in such a window.
            if not any(x in initial_d for x in cells) and sum(x in region for x in cells) >= 2:
                additions.update(x for x in cells if x not in initial_d)
            # All future defender stones before escape are already in R.  If
            # D0 plus those possible cells could form a count-four threat,
            # include every possible claimant block as well.
            if not any(x in initial_a for x in cells):
                possible_d = sum(x in initial_d or x in region for x in cells)
                if possible_d >= 4:
                    additions.update(x for x in cells if x not in initial_d)
        new = additions - region
        if not new:
            return {
                "closed": True,
                "reason": "fixed_point",
                "size": len(region),
                "empty": len(region - set(stones)),
                "iterations": iterations,
            }
        region.update(new)
        if len(region - set(stones)) > cap:
            return {
                "closed": False,
                "reason": "cap",
                "size_lb": len(region),
                "empty_lb": len(region - set(stones)),
                "iterations": iterations,
            }


def load_jsonl(path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def summarize(name, rows, closure_cap, skip_closure=False):
    total = 0
    carriers = 0
    carrier_eligible = 0
    phi = 0
    opponent_phi = 0
    line_unbounded = 0
    line_lb_gt6 = 0
    line_no_carrier = 0
    closed = 0
    closure_census_refutations = 0
    empties = []
    max_alive = Counter()
    samples = []
    for row in rows:
        total += 1
        feat = static_features(row["moves"])
        carriers += feat["carrierless"]
        carrier_eligible += feat["carrier_eligible"]
        phi += feat["phi_lt_1"]
        opponent_phi += feat["opponent_phi_lt_1_at_defender_first"]
        line = line_component_lower_bound(row["moves"])
        line_unbounded += line["unbounded"]
        line_no_carrier += line["carrier_count"] == 0
        line_lb_gt6 += (not line["unbounded"] and line["empty_lb"] > 6)
        max_alive[feat["max_alive"]] += 1
        if skip_closure:
            continue
        closure = forcing_region_closure(row["moves"], closure_cap)
        if closure["closed"]:
            closed += 1
            empties.append(closure["empty"])
            phase = "opening" if not row["moves"] else ("first" if len(row["moves"]) % 2 else "second")
            table = {
                "first": [10, 10, 9, 6, 2, 1],
                "second": [12, 12, 9, 5, 4, 1],
            }.get(phase)
            if table is not None and feat["max_alive"] <= 5:
                closure_census_refutations += table[feat["max_alive"]] > closure["empty"] + 2
            if len(samples) < 8:
                samples.append((row.get("pos_id") or row.get("id"), closure["empty"]))
    print(json.dumps({
        "cohort": name,
        "n": total,
        "carrierless": carriers,
        "carrier_eligible": carrier_eligible,
        "carrierless_pct_eligible": round(100 * carriers / max(carrier_eligible, 1), 3),
        "phi_lt_1": phi,
        "phi_lt_1_pct": round(100 * phi / max(total, 1), 3),
        "opponent_phi_lt_1_at_defender_first": opponent_phi,
        "line_unbounded": line_unbounded,
        "line_lb_gt6": line_lb_gt6,
        "line_no_carrier": line_no_carrier,
        "closure_fixed": closed,
        "closure_fixed_pct": round(100 * closed / max(total, 1), 3),
        "closure_census_refutations": closure_census_refutations,
        "closure_empty_min": min(empties) if empties else None,
        "closure_empty_max": max(empties) if empties else None,
        "closure_empty_le8": sum(x <= 8 for x in empties),
        "max_alive": dict(sorted(max_alive.items())),
        "closure_samples": samples,
    }, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2000)
    parser.add_argument("--skip-closure", action="store_true")
    parser.add_argument("--sets", nargs="*", default=["selfplay_v1", "human_v1", "puzzle_v3"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    set_dir = root / "scripts" / "tss_harness" / "sets"
    loaded = {}
    for name in args.sets:
        rows = list(load_jsonl(set_dir / f"{name}.jsonl"))
        loaded[name] = rows
        summarize(name, rows, args.cap, args.skip_closure)

    # The 248 production cap-bound grinds are identified by lanec_labels and
    # sourced from the pinned selfplay set.  Keep the measured outcome strata.
    by_id = {r["pos_id"]: r for r in loaded.get("selfplay_v1", [])}
    labels = list(load_jsonl(root / "raws" / "lanec_labels.jsonl"))
    grind_rows = []
    strata = {"deep_win": [], "width_exhaust": [], "cap50k": []}
    for label in labels:
        row = by_id.get(label["pos_id"])
        if row is None:
            continue
        grind_rows.append(row)
        win = label.get("win_pass", {})
        if label.get("status") == "win":
            strata["deep_win"].append(row)
        elif win.get("deep_nodes", 0) >= 50000:
            strata["cap50k"].append(row)
        else:
            strata["width_exhaust"].append(row)
    summarize("grinds_all", grind_rows, args.cap, args.skip_closure)
    for name, rows in strata.items():
        summarize(f"grinds_{name}", rows, args.cap, args.skip_closure)


if __name__ == "__main__":
    main()
