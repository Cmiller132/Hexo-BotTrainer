import os
import re


BASE = r"E:/Hexo-BotTrainer-hexgt/.claude/worktrees/opening-atlas"
BASE_RAW = os.path.join(BASE, "OPENING_ATLAS_CORPUS9_DEEP_RAW.txt")
UPGRADE_RAWS = [
    os.path.join(BASE, "squeeze9_hinted_lift.txt"),
    os.path.join(BASE, "squeeze9_hinted_lift_depth7.txt"),
]
OUT = os.path.join(BASE, "OPENING_ATLAS_CORPUS9_SQUEEZE_RAW.txt")


def read_text(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


def field(line, key):
    match = re.search(rf"(?:^| ){re.escape(key)}=([^\s]+)", line)
    return match.group(1) if match else None


base_text = read_text(BASE_RAW)
base_rows = [line for line in base_text.splitlines() if line.startswith("ATLAS_ROW ")]
assert len(base_rows) == 24_672, len(base_rows)
base_by_id = {field(line, "id"): line for line in base_rows}
assert len(base_by_id) == len(base_rows)
base_wins = sum(field(line, "status") == "WIN" for line in base_rows)
assert base_wins == 958, base_wins

upgrades = {}
for path in UPGRADE_RAWS:
    for line in read_text(path).splitlines():
        if not line.startswith("ATLAS_ROW ") or field(line, "status") != "WIN":
            continue
        assert field(line, "certified") == "1"
        assert field(line, "win_line") not in (None, "NA", "")
        rid = field(line, "id")
        assert rid in base_by_id, rid
        assert field(base_by_id[rid], "status") == "UNKNOWN", rid
        # The depth-7 hint run is sourced from the first-11 raw but its parent
        # is still an exact member of the first-9 corpus result.
        line = re.sub(r" source=corpus7:depth7 ", " source=corpus9:depth7 ", line)
        if rid in upgrades:
            assert upgrades[rid] == line, rid
        upgrades[rid] = line

assert len(upgrades) == 7, len(upgrades)
merged_rows = [upgrades.get(field(line, "id"), line) for line in base_rows]
assert len({field(line, "id") for line in merged_rows}) == 24_672
assert sum(field(line, "status") == "WIN" for line in merged_rows) == 965
assert sum(field(line, "status") == "LOSS" for line in merged_rows) == 0
for line in merged_rows:
    if field(line, "status") == "WIN":
        assert field(line, "certified") == "1"
        assert field(line, "win_line") not in (None, "NA", "")

setup = next(line for line in base_text.splitlines() if line.startswith("ATLAS_SETUP "))
setup += " squeeze_upgrades=7 squeeze_profile=verified_same_claimant_child_lift"
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(setup + "\n")
    fh.write("\n".join(merged_rows) + "\n")
    fh.write(
        "ATLAS_DONE_AGGREGATE rows=24672 wins=965 losses=0 unknown=23707 "
        "baseline_wins=958 squeeze_upgrades=7 verifier_rejects_merged=0\n"
    )

print(
    {
        "rows": len(merged_rows),
        "baseline_wins": base_wins,
        "new_wins": len(upgrades),
        "final_wins": sum(field(line, "status") == "WIN" for line in merged_rows),
        "upgrade_ids": sorted(upgrades),
        "out": OUT,
    }
)
