"""Validate and merge the 16 additive corpus-11 expansion shards."""

import glob
import json
import os
import statistics


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARD_DIR = os.path.join(BASE, "corpus11_expand_shards_run2")
ATLAS = os.path.join(BASE, "atlas-web", "data", "atlas.json")
OUT = os.path.join(BASE, "OPENING_ATLAS_CORPUS11_EXPAND_RAW.txt")


def fields(line, prefix):
    out = {}
    for token in line[len(prefix) :].split():
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
    return out


def parse_moves(value):
    if value in ("", "NA"):
        return []
    return [tuple(map(int, pair.split(","))) for pair in value.split(";")]


def owner(placement):
    if placement == 0:
        return "P0"
    return "P0" if ((placement - 1) // 2) % 2 == 1 else "P1"


def has_six(board, player):
    stones = {coord for coord, occupant in board.items() if occupant == player}
    for q, r in stones:
        for dq, dr in ((1, 0), (0, 1), (1, -1)):
            if all((q + step * dq, r + step * dr) in stones for step in range(6)):
                return True
    return False


shards = sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.txt")))
assert len(shards) == 16, f"expected 16 shards, found {len(shards)}"
setups = []
dones = []
rows = []
for path in shards:
    lines = open(path, encoding="utf-8-sig").read().splitlines()
    assert any(line.startswith("test result: ok") for line in lines), path
    shard_setups = [
        fields(line[line.index("ATLAS_SETUP ") :], "ATLAS_SETUP ")
        for line in lines
        if "ATLAS_SETUP " in line
    ]
    shard_dones = [fields(line, "ATLAS_DONE ") for line in lines if line.startswith("ATLAS_DONE ")]
    assert len(shard_setups) == len(shard_dones) == 1, path
    setups.extend(shard_setups)
    dones.extend(shard_dones)
    rows.extend(
        (line, fields(line, "ATLAS_ROW "))
        for line in lines
        if line.startswith("ATLAS_ROW ")
    )

assert all(
    setup["games"] == "8698"
    and setup["first_n"] == "11"
    and setup["candidates"] == "9968"
    and setup["skipped_existing"] == "37840"
    and setup["width"] == "vcf_pair_complete"
    and setup["node_ladder"] == "[100000]"
    and setup["unbounded_horizon"] == "true"
    for setup in setups
)
assert sum(int(done["attempted"]) for done in dones) == 9968
assert sum(int(done["residual"]) for done in dones) == 0
assert len(rows) == 9968

with open(ATLAS, encoding="utf-8") as stream:
    old_ids = {row["id"] for row in json.load(stream)["rows"]}
ids = [row["id"] for _line, row in rows]
assert len(ids) == len(set(ids)), "duplicate expansion ids"
assert not old_ids.intersection(ids), "expansion overlaps frozen atlas"

for _line, row in rows:
    assert (row["status"] == "UNKNOWN") == (row["certified"] == "0")
    if row["certified"] == "1":
        assert row["status"] in ("WIN", "LOSS")
        # Symmetry zero is the canonical certificate asserted by the harness.
        assert int(row["d6_mask"], 16) & 1
        assert int(row["d6_verified"]) >= 1
    if row["status"] != "WIN":
        continue
    root = parse_moves(row["moves"])
    line = parse_moves(row["win_line"])
    assert len(line) == int(row["win_line_len"])
    board = {coord: owner(index) for index, coord in enumerate(root)}
    assert len(board) == len(root)
    for offset, coord in enumerate(line):
        assert coord not in board
        board[coord] = owner(len(root) + offset)
        terminal = has_six(board, "P0") or has_six(board, "P1")
        if terminal:
            assert offset + 1 == len(line)
            assert has_six(board, row["claimant"])
            assert row["win_line_terminal"] == "1"
    assert row["win_line_terminal"] == "1"
    assert has_six(board, row["claimant"])

rows.sort(key=lambda item: (int(item[1]["source_prefix"]), item[1]["id"]))
setup = setups[0].copy()
setup["shard_index"] = "ALL"
setup["shard_total"] = setup["candidates"]
setup_line = "ATLAS_SETUP " + " ".join(f"{key}={value}" for key, value in setup.items())
wall_max = max(float(done["wall_ms"]) for done in dones)
wall_sum = sum(float(done["wall_ms"]) for done in dones)
with open(OUT, "w", encoding="utf-8", newline="\n") as stream:
    stream.write(setup_line + "\n")
    for line, _row in rows:
        stream.write(line + "\n")
    stream.write(
        "ATLAS_DONE_AGGREGATE shards=16 attempted=9968 residual=0 "
        f"rows=9968 wall_ms_max={wall_max:.3f} wall_ms_sum={wall_sum:.3f}\n"
    )

statuses = {
    status: sum(row["status"] == status for _line, row in rows)
    for status in ("WIN", "LOSS", "UNKNOWN")
}
nodes = [int(row["nodes"]) for _line, row in rows]
deepest = max(rows, key=lambda item: int(item[1]["nodes"]))[1]
print(
    json.dumps(
        {
            "new_rows": len(rows),
            "status": statuses,
            "certified": statuses["WIN"] + statuses["LOSS"],
            "terminal_wins": sum(
                row["status"] == "WIN" and row["win_line_terminal"] == "1"
                for _line, row in rows
            ),
            "nodes_total": sum(nodes),
            "nodes_median": statistics.median(nodes),
            "nodes_max": max(nodes),
            "max_node_id": deepest["id"],
            "wall_ms_max": wall_max,
            "wall_ms_sum": wall_sum,
            "output": OUT,
        },
        indent=2,
    )
)
