"""Enumerate decisive-child transpositions as verifier-gated parent hints.

This script never mints a verdict.  It finds UNKNOWN atlas roots that can reach
an already decisive position in one claimant placement or one complete
two-placement claimant turn.  The Rust atlas harness replays each hint,
re-solves the child, reconstructs the parent certificate, and runs the strict
verifier before emitting anything.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations


def raw_fields(line: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in line.split() if "=" in token)


def parse_moves(value: str) -> tuple[tuple[int, int], ...]:
    if not value or value == "NA":
        return ()
    return tuple(tuple(map(int, pair.split(","))) for pair in value.split(";"))


def transform(coord: tuple[int, int], symmetry: int) -> tuple[int, int]:
    q, r = coord
    if symmetry >= 6:
        r = -q - r
    for _ in range(symmetry % 6):
        q, r = -r, q + r
    return q, r


def owner_at(index: int) -> str:
    if index == 0:
        return "P0"
    return "P1" if ((index - 1) // 2) % 2 == 0 else "P0"


def other(player: str) -> str:
    return "P1" if player == "P0" else "P0"


@dataclass(frozen=True)
class Row:
    id: str
    moves: tuple[tuple[int, int], ...]
    side: str
    phase: str
    claimant: str | None
    status: str

    @property
    def depth(self) -> int:
        return len(self.moves)


def row_from_json(value: dict) -> Row:
    return Row(
        id=value["id"],
        moves=tuple(tuple(pair) for pair in value["moves"]),
        side=value["side"],
        phase=value["phase"],
        claimant=value.get("claimant"),
        status=value["status"],
    )


def row_from_raw(line: str) -> Row:
    value = raw_fields(line)
    return Row(
        id=value["id"],
        moves=parse_moves(value["moves"]),
        side=value["side"],
        phase=value["phase"],
        claimant=value["claimant"],
        status=value["status"],
    )


Stone = tuple[int, int, str]
PositionKey = tuple[str, str, tuple[int, int] | None, tuple[Stone, ...]]


def position_key(row: Row) -> PositionKey:
    first = row.moves[-1] if row.phase == "SecondStone" else None
    stones = tuple(
        sorted((coord[0], coord[1], owner_at(index)) for index, coord in enumerate(row.moves))
    )
    return row.side, row.phase, first, stones


def transformed_child(row: Row, symmetry: int) -> tuple[PositionKey, tuple[Stone, ...]]:
    moves = tuple(transform(coord, symmetry) for coord in row.moves)
    first = moves[-1] if row.phase == "SecondStone" else None
    stones = tuple(
        sorted((coord[0], coord[1], owner_at(index)) for index, coord in enumerate(moves))
    )
    return (row.side, row.phase, first, stones), stones


def without(stones: tuple[Stone, ...], coords: frozenset[tuple[int, int]]) -> tuple[Stone, ...]:
    return tuple(stone for stone in stones if (stone[0], stone[1]) not in coords)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--upgrades", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--parent-min", type=int, default=0)
    parser.add_argument("--parent-max", type=int, default=999)
    parser.add_argument("--include-upgraded-parents", action="store_true")
    args = parser.parse_args()

    with open(args.atlas, encoding="utf-8") as fh:
        atlas = json.load(fh)
    atlas_rows = [row_from_json(value) for value in atlas["rows"]]
    by_id = {row.id: row for row in atlas_rows}
    assert len(by_id) == len(atlas_rows), "duplicate atlas ids"

    decisive = {row.id: row for row in atlas_rows if row.status in ("WIN", "LOSS")}
    with open(args.upgrades, encoding="utf-8-sig") as fh:
        for line in fh:
            if not line.startswith("ATLAS_ROW "):
                continue
            row = row_from_raw(line)
            if row.status not in ("WIN", "LOSS"):
                continue
            old = by_id[row.id]
            assert old.status == "UNKNOWN", f"upgrade is not for UNKNOWN root: {row.id}"
            assert row.moves == old.moves and row.side == old.side and row.phase == old.phase
            previous = decisive.setdefault(row.id, row)
            assert previous.status == row.status and previous.claimant == row.claimant

    unknown_by_key: dict[PositionKey, Row] = {}
    for row in atlas_rows:
        if (
            row.status != "UNKNOWN"
            or (row.id in decisive and not args.include_upgraded_parents)
            or not (args.parent_min <= row.depth <= args.parent_max)
        ):
            continue
        key = position_key(row)
        previous = unknown_by_key.setdefault(key, row)
        assert previous.id == row.id, f"distinct UNKNOWN ids share one exact position: {row.id}"

    # (parent id, line, claimant, child id, delta).  Multiple decisive children
    # may justify the same routing line; retain one deterministic witness.
    hints: dict[tuple[str, tuple[tuple[int, int], ...]], tuple[str, str, int]] = {}
    matched_children: set[str] = set()
    transformed_positions = 0
    for child in decisive.values():
        if child.claimant not in ("P0", "P1"):
            continue
        seen_images: set[PositionKey] = set()
        for symmetry in range(12):
            child_key, stones = transformed_child(child, symmetry)
            if child_key in seen_images:
                continue
            seen_images.add(child_key)
            transformed_positions += 1
            side, phase, first, _ = child_key
            claimant = child.claimant

            def record(parent_key: PositionKey, line: tuple[tuple[int, int], ...]) -> None:
                parent = unknown_by_key.get(parent_key)
                if parent is None or parent.side != claimant or parent.depth + len(line) != child.depth:
                    return
                key = (parent.id, line)
                previous = hints.setdefault(key, (claimant, child.id, len(line)))
                assert previous[0] == claimant and previous[2] == len(line)
                matched_children.add(child.id)

            # Opening is the sole one-stone turn.  Its child is at depth one.
            if child.depth == 1 and phase == "FirstStone" and claimant == "P0":
                claimant_stones = [(q, r) for q, r, owner in stones if owner == claimant]
                if len(claimant_stones) == 1:
                    extra = claimant_stones[0]
                    record(("P0", "Opening", None, ()), (extra,))

            # First placement of a normal two-stone turn: the child retains the
            # exact first coordinate in its phase binding.
            if phase == "SecondStone" and side == claimant and first is not None:
                removed = without(stones, frozenset((first,)))
                record((claimant, "FirstStone", None, removed), (first,))

            # End of a normal turn: a LOSS for the now-moving opponent has the
            # previous mover as claimant.  Enumerate both a one-placement
            # predecessor (already at SecondStone) and a complete-turn
            # predecessor (at FirstStone).
            if phase == "FirstStone" and other(side) == claimant:
                claimant_coords = [(q, r) for q, r, owner in stones if owner == claimant]
                for extra in claimant_coords:
                    removed = without(stones, frozenset((extra,)))
                    for q, r, owner in removed:
                        if owner == claimant:
                            record(
                                (claimant, "SecondStone", (q, r), removed),
                                (extra,),
                            )
                for a, b in combinations(claimant_coords, 2):
                    removed = without(stones, frozenset((a, b)))
                    parent_key = (claimant, "FirstStone", None, removed)
                    record(parent_key, (a, b))
                    record(parent_key, (b, a))

    ordered = sorted(
        ((parent, line, *data) for (parent, line), data in hints.items()),
        key=lambda item: (item[0], item[1], item[3]),
    )
    by_delta = {1: 0, 2: 0}
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "ATLAS_TRANSPOSE_SETUP schema=1 "
            f"decisive_children={len(decisive)} unknown_parents={len(unknown_by_key)} "
            f"transformed_positions={transformed_positions}\n"
        )
        for parent, line, claimant, child, delta in ordered:
            by_delta[delta] += 1
            line_text = ";".join(f"{q},{r}" for q, r in line)
            fh.write(
                f"ATLAS_TRANSPOSE_HINT parent={parent} child={child} "
                f"claimant={claimant} delta={delta} line={line_text}\n"
            )
        fh.write(
            "ATLAS_TRANSPOSE_DONE "
            f"parents={len({item[0] for item in ordered})} lines={len(ordered)} "
            f"delta1={by_delta[1]} delta2={by_delta[2]} "
            f"matched_children={len(matched_children)}\n"
        )
    print(
        json.dumps(
            {
                "decisive_children": len(decisive),
                "unknown_parents": len(unknown_by_key),
                "transformed_positions": transformed_positions,
                "hint_parents": len({item[0] for item in ordered}),
                "hint_lines": len(ordered),
                "delta1": by_delta[1],
                "delta2": by_delta[2],
                "matched_children": len(matched_children),
                "out": args.out,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
