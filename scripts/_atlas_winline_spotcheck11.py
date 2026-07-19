import json, os

BASE = r"E:/Hexo-BotTrainer-hexgt/.claude/worktrees/opening-atlas"
atlas = json.load(open(os.path.join(BASE, "atlas-web", "data", "atlas.json"), encoding="utf-8"))


def owner_at(i):
    if i == 0:
        return 0
    return 1 if ((i - 1) // 2) % 2 == 0 else 0


def has_six(stones, player):
    occ = {(q, r): o for (q, r, o) in stones}
    dirs = [(1, 0), (0, 1), (1, -1)]
    for (q, r, o) in stones:
        if o != player:
            continue
        for dq, dr in dirs:
            if all(occ.get((q + k * dq, r + k * dr)) == player for k in range(6)):
                return (q, r, dq, dr)
    return None


def check(row):
    full = [tuple(m) for m in row["moves"]] + [tuple(m) for m in row["win_line"]]
    stones = [(q, r, owner_at(i)) for i, (q, r) in enumerate(full)]
    claimant = 0 if row["claimant"] == "P0" else 1
    return has_six(stones, claimant), stones, claimant


# Prefer a deep (depth 8-9) terminal win to showcase the new results.
cands = [r for r in atlas["rows"]
         if r.get("win_line_terminal") == 1 and r["source"].startswith("corpus11")
         and r["source_prefix"] >= 8]
cands.sort(key=lambda r: -r["source_prefix"])
checked = 0
shown = 0
for r in atlas["rows"]:
    if r.get("win_line_terminal") == 1 and r["source"].startswith("corpus11"):
        six, _, _ = check(r)
        assert six is not None, f"terminal win_line without six-in-a-row: {r['id']}"
        checked += 1

print(f"VERIFIED: all {checked} terminal win_lines replay to a real six-in-a-row for the claimant")

r = cands[0]
six, stones, claimant = check(r)
print("\n--- showcase (deepest terminal win) ---")
print("id           :", r["id"])
print("source/depth :", r["source"], "| claimant:", r["claimant"], "| side:", r["side"])
print("opening moves:", ";".join(f"{q},{r_}" for q, r_ in r["moves"]))
print("win_line     :", ";".join(f"{q},{r_}" for q, r_ in r["win_line"]))
q, rr, dq, dr = six
line = [(q + k * dq, rr + k * dr) for k in range(6)]
print(f"six-in-a-row : {line}  (axis {dq},{dr}, claimant P{claimant})")
