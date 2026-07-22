#!/usr/bin/env python3
"""Third, standalone oracle for RefuteLeafExact/V1 goldens.

Uses only the Python standard library and deliberately imports no repository
package. It independently replays the opening/turn schedule, builds literal
windows, T/G1/S/U, dispositions, quotient classes, canonical preimages, and
SHA-256 values for the frozen roots below.
"""

from __future__ import annotations

import hashlib
import json

AXES = ((1, 0), (0, 1), (1, -1))
DOMAIN = b"HXRFLV1:ROOT-SEMANTIC:V1\0"

ROOTS = {
    "q0_corpus_0hz3hty_prefix3": [(0, 0), (5, 0), (5, -1)],
    "q2_commuting_no_new": [
        (0, 0), (0, 1), (-1, 5), (1, 0), (4, 0),
        (1, 4), (4, 2), (5, 0), (-2, -2),
    ],
    "q4_sole_orientation_no_new": [
        (0, 0), (-1, 0), (6, 0), (1, 0), (2, 0),
        (3, 5), (-1, 6), (4, 0), (5, 0),
    ],
}

# Filled from this independent implementation and frozen for CI drift checks.
EXPECTED = {
    "q0_corpus_0hz3hty_prefix3": {
        "preimage_hex": "485852464c56313a524f4f542d53454d414e5449433a563100010001000100010001000300000000000500ffff0105000000010001030000000000",
        "digest_hex": "1e0ee42712858b73e46fcfe603a6400bf29676ccb5d5921fbdb52225b26d6167",
        "t": 0, "q": 0, "classes": 0,
        "fail": [0, 0, 0, 0],
    },
    "q2_commuting_no_new": {
        "preimage_hex": "485852464c56313a524f4f542d53454d414e5449433a5631000100010001000100010009fefffeff00ffff05000100000000000000010001010000000001000400010400000000040002000105000000000101090000000001",
        "digest_hex": "499d226e46bd418ab44e42819229b09b8ed47f31047856a059445586c73e5b0a",
        "t": 2, "q": 2, "classes": 1,
        "fail": [2, 0, 0, 0],
    },
    "q4_sole_orientation_no_new": {
        "preimage_hex": "485852464c56313a524f4f542d53454d414e5449433a5631000100010001000100010009ffff000001ffff06000100000000000100000000020000000003000500010400000000050000000006000000010101090000000001",
        "digest_hex": "0ef6c6f1d35ff6826d655b3b11a8af537bf1be1da114034a1c7feef2adf2817c",
        "t": 1, "q": 4, "classes": 4,
        "fail": [4, 0, 0, 0],
    },
}

R3_1_SEMANTIC_EXPECTED = {
    "claimant": 1,
    "terminal": None,
    "live_claimant_count_ge_2": [],
    "live_defender_count_ge_4": [
        (0, 0, 0, 5, ((3, 0),)),
    ],
    "claimant_live_through_selected_a": [
        (1, 3, 0, 1, ((3, 0), (3, 1), (3, 2), (3, 3), (3, 4))),
    ],
    "t": ((3, 0),),
    "g1": {(3, 0): ((3, 1), (3, 2), (3, 3), (3, 4))},
    "u": (
        ((3, 0), (3, 1)),
        ((3, 0), (3, 2)),
        ((3, 0), (3, 3)),
        ((3, 0), (3, 4)),
    ),
    "defender_threat_family": (((3, 0),),),
    "defender_tau": 1,
    "own_win_now_claimant_b2": False,
    "forced_loss_claimant_b2": False,
    "occurrence_counts": {
        "no_new": 4,
        "defender_first": 0,
        "loose_0": 0,
        "loose_1": 0,
        "completion": 0,
        "tactical": 0,
        "tight": 0,
    },
    "class_counts": {
        "no_new": 4,
        "defender_first": 0,
        "loose_0": 0,
        "loose_1": 0,
        "completion": 0,
        "tactical": 0,
        "tight": 0,
    },
    "classes": (
        ((((3, 0), (3, 1)),), "no_new"),
        ((((3, 0), (3, 2)),), "no_new"),
        ((((3, 0), (3, 3)),), "no_new"),
        ((((3, 0), (3, 4)),), "no_new"),
    ),
}


def owner_at(index: int) -> int:
    if index == 1:
        return 0
    return 1 if ((index - 2) // 2) % 2 == 0 else 0


def mover_after(count: int) -> int:
    assert count % 2 == 1
    return 1 if ((count - 1) // 2) % 2 == 0 else 0


def uvar(value: int) -> bytes:
    out = bytearray()
    while True:
        low = value & 0x7F
        value >>= 7
        out.append(low if value == 0 else low | 0x80)
        if value == 0:
            return bytes(out)


def i16(value: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=True)


def cells(key):
    axis, q, r = key
    dq, dr = AXES[axis]
    return tuple((q + i * dq, r + i * dr) for i in range(6))


def window_keys(board):
    keys = set()
    for q, r in board:
        for axis, (dq, dr) in enumerate(AXES):
            for offset in range(6):
                keys.add((axis, q - offset * dq, r - offset * dr))
    return sorted(keys)


def distance(a, b):
    dq, dr = a[0] - b[0], a[1] - b[1]
    return max(abs(dq), abs(dr), abs(-dq - dr))


def legal(board, coord):
    return coord not in board and any(distance(coord, stone) <= 8 for stone in board)


def counts(board, key, player):
    return sum(board.get(c) == player for c in cells(key))


def empties(board, key):
    return tuple(c for c in cells(key) if c not in board)


def live(board, key, player):
    return counts(board, key, player) > 0 and counts(board, key, player ^ 1) == 0 and bool(empties(board, key))


def winner(board, placed):
    player = board[placed]
    for axis, (dq, dr) in enumerate(AXES):
        for offset in range(6):
            key = (axis, placed[0] - offset * dq, placed[1] - offset * dr)
            if all(board.get(c) == player for c in cells(key)):
                return player
    return None


def apply(state, coord):
    board, mover, phase, clock, terminal = state
    assert terminal is None and legal(board, coord)
    board = dict(board)
    board[coord] = mover
    clock += 1
    terminal = winner(board, coord)
    if terminal is None:
        if phase[0] == 1:
            phase = (2, coord)
        else:
            phase = (1, None)
            mover ^= 1
    return board, mover, phase, clock, terminal


def replay(history):
    state = ({}, 0, (0, None), 0, None)
    for index, coord in enumerate(history, 1):
        board, mover, phase, clock, terminal = state
        assert terminal is None
        assert mover == owner_at(index)
        if phase[0] == 0:
            assert not board and index == 1 and coord == (0, 0)
            board = {coord: mover}
            state = (board, 1, (1, None), 1, winner(board, coord))
        else:
            state = apply(state, coord)
    return state


def tau(family):
    if not family:
        return 0
    if any(not member for member in family):
        return 3
    union = sorted(set().union(*map(set, family)))
    if any(all(x in member for member in family) for x in union):
        return 1
    if any(all(x in member or y in member for member in family)
           for i, x in enumerate(union) for y in union[i + 1:]):
        return 2
    return 3


def disposition(root, windows, claimant, a, b):
    first = apply(root, a)
    if first[4] == claimant:
        return "completion", True, None
    full = apply(first, b)
    if full[4] == claimant:
        return "completion", False, full
    assert full[4] is None
    rb, _, _, _, _ = root
    fb, _, _, _, _ = full
    family = []
    for key in windows:
        cs = cells(key)
        if live(rb, key, claimant) and counts(rb, key, claimant) >= 2 and (a in cs or b in cs):
            old_empty = empties(rb, key)
            projected = counts(rb, key, claimant) + (a in old_empty) + (b in old_empty)
            if projected >= 4:
                family.append(empties(fb, key))
    if not family:
        return "no_new", False, full
    for key in windows:
        cs = cells(key)
        if live(rb, key, claimant ^ 1) and counts(rb, key, claimant ^ 1) >= 4 and a not in cs and b not in cs:
            return "defender_first", False, full
    value = tau(family)
    return {0: "loose_0", 1: "loose_1", 2: "tight", 3: "tactical"}[value], False, full


def regenerate(history):
    root = replay(history)
    board, claimant, phase, clock, terminal = root
    assert phase == (1, None)
    assert clock == len(history)
    assert claimant == mover_after(len(history))
    assert terminal is None
    windows = window_keys(board)
    t_set = set()
    for key in windows:
        if live(board, key, claimant) and counts(board, key, claimant) >= 2:
            t_set.update(c for c in empties(board, key) if legal(board, c))
        if live(board, key, claimant ^ 1) and counts(board, key, claimant ^ 1) >= 4:
            t_set.update(c for c in empties(board, key) if legal(board, c))
    universe = []
    for a in sorted(t_set):
        second = set(t_set) - {a}
        for key in windows:
            e = empties(board, key)
            if live(board, key, claimant) and counts(board, key, claimant) >= 1 and a in e:
                second.update(c for c in e if c != a and legal(board, c))
        universe.extend((a, b) for b in sorted(second))
    evaluated = {pair: disposition(root, windows, claimant, *pair) for pair in universe}
    fail_names = ("no_new", "defender_first", "loose_0", "loose_1")
    fail = [sum(value[0] == name for value in evaluated.values()) for name in fail_names]
    classes = []
    seen = set()
    for pair in universe:
        if pair in seen:
            continue
        reverse = (pair[1], pair[0])
        value = evaluated[pair]
        rev = evaluated.get(reverse)
        if rev and not value[1] and not rev[1] and value[2] == rev[2]:
            assert value[0] == rev[0]
            seen.add(reverse)
            members = (pair, reverse)
        else:
            members = (pair,)
        seen.add(pair)
        classes.append((members, value[0]))
    occurrence_names = fail_names + ("completion", "tactical", "tight")
    occurrence_counts = {
        name: sum(value[0] == name for value in evaluated.values())
        for name in occurrence_names
    }
    class_counts = {
        name: sum(reason == name for _, reason in classes)
        for name in occurrence_names
    }
    assert len(universe) == sum(occurrence_counts.values())
    assert len(universe) == sum(len(members) for members, _ in classes)
    assert len(classes) == sum(class_counts.values())
    if not any(occurrence_counts[name] for name in ("completion", "tactical", "tight")):
        assert len(universe) == sum(fail)
        assert len(classes) == sum(class_counts[name] for name in fail_names)
    return {
        "root": root,
        "windows": windows,
        "t_set": t_set,
        "universe": universe,
        "evaluated": evaluated,
        "fail": fail,
        "classes": classes,
        "occurrence_counts": occurrence_counts,
        "class_counts": class_counts,
    }


def oracle(history):
    regenerated = regenerate(history)
    board, claimant, _, _, _ = regenerated["root"]
    fail = regenerated["fail"]
    classes = regenerated["classes"]
    stones = sorted((q, r, owner) for (q, r), owner in board.items())
    preimage = bytearray(DOMAIN)
    for value in (1, 1, 1, 1, 1):
        preimage += value.to_bytes(2, "little")
    preimage += uvar(len(stones))
    for q, r, owner in stones:
        preimage += i16(q) + i16(r) + bytes((owner,))
    preimage += bytes((claimant, 1))
    preimage += len(history).to_bytes(4, "little")
    preimage += bytes((0, claimant))
    return {
        "preimage_hex": preimage.hex(),
        "digest_hex": hashlib.sha256(preimage).hexdigest(),
        "t": len(regenerated["t_set"]),
        "q": len(regenerated["universe"]),
        "classes": len(classes),
        "fail": fail,
    }


def own_win_now_claimant_b2(root, windows, claimant):
    board = root[0]
    candidates = sorted({
        coord
        for key in windows
        if live(board, key, claimant)
        for coord in empties(board, key)
        if legal(board, coord)
    })
    for a in candidates:
        first = apply(root, a)
        if first[4] == claimant:
            return True
        for b in candidates:
            if b != a and legal(first[0], b) and apply(first, b)[4] == claimant:
                return True
    return False


def r3_1_semantic_census(history):
    regenerated = regenerate(history)
    root = regenerated["root"]
    board, claimant, _, _, terminal = root
    windows = regenerated["windows"]
    selected_a = (3, 0)

    def window_record(key, player):
        return (key[0], key[1], key[2], counts(board, key, player), empties(board, key))

    live_claimant_count_ge_2 = [
        window_record(key, claimant) for key in windows
        if live(board, key, claimant) and counts(board, key, claimant) >= 2
    ]
    live_defender_count_ge_4 = [
        window_record(key, claimant ^ 1) for key in windows
        if live(board, key, claimant ^ 1) and counts(board, key, claimant ^ 1) >= 4
    ]
    claimant_live_through_selected_a = [
        window_record(key, claimant) for key in windows
        if selected_a in cells(key) and live(board, key, claimant)
    ]
    g1 = {}
    for a in sorted(regenerated["t_set"]):
        promoted = set()
        for key in windows:
            empty = empties(board, key)
            if live(board, key, claimant) and counts(board, key, claimant) >= 1 and a in empty:
                promoted.update(c for c in empty if c != a and legal(board, c))
        g1[a] = tuple(sorted(promoted))
    defender_threat_family = tuple(
        sorted(tuple(sorted(empties(board, key))) for key in windows
               if live(board, key, claimant ^ 1) and counts(board, key, claimant ^ 1) >= 4)
    )
    own_win = own_win_now_claimant_b2(root, windows, claimant)
    census = {
        "claimant": claimant,
        "terminal": terminal,
        "live_claimant_count_ge_2": live_claimant_count_ge_2,
        "live_defender_count_ge_4": live_defender_count_ge_4,
        "claimant_live_through_selected_a": claimant_live_through_selected_a,
        "t": tuple(sorted(regenerated["t_set"])),
        "g1": g1,
        "u": tuple(regenerated["universe"]),
        "defender_threat_family": defender_threat_family,
        "defender_tau": tau(defender_threat_family),
        "own_win_now_claimant_b2": own_win,
        "forced_loss_claimant_b2": not own_win and tau(defender_threat_family) > 2,
        "occurrence_counts": regenerated["occurrence_counts"],
        "class_counts": regenerated["class_counts"],
        "classes": tuple(regenerated["classes"]),
    }
    selected = (((3, 0), (3, 1)),)
    selected_classes = [members for members, reason in census["classes"]
                        if members == selected and reason == "no_new"]
    assert len(selected_classes) == 1 and len(selected_classes[0]) == 1
    assert len(census["u"]) == sum(census["occurrence_counts"].values())
    assert len(census["u"]) == sum(len(members) for members, _ in census["classes"])
    assert len(census["classes"]) == sum(census["class_counts"].values())
    return census


def main():
    actual = {name: oracle(history) for name, history in ROOTS.items()}
    for name, expected in EXPECTED.items():
        if expected["preimage_hex"]:
            assert actual[name] == expected, (name, actual[name], expected)
    census = r3_1_semantic_census(ROOTS["q4_sole_orientation_no_new"])
    assert census == R3_1_SEMANTIC_EXPECTED, (census, R3_1_SEMANTIC_EXPECTED)
    print(json.dumps(actual, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
