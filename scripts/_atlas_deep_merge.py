import glob, os, re

BASE = r"E:/Hexo-BotTrainer-hexgt/.claude/worktrees/opening-atlas"
SHARDS = sorted(glob.glob(os.path.join(BASE, "deep_shards", "shard_*.txt")))
OUT = os.path.join(BASE, "OPENING_ATLAS_CORPUS7_DEEP_RAW.txt")


def read_text(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


rows = []
setup = None
attempted = 0
residual = 0
for sp in SHARDS:
    for line in read_text(sp).splitlines():
        line = line.rstrip("\r")
        if line.startswith("ATLAS_ROW "):
            rows.append(line)
        elif setup is None and "ATLAS_SETUP" in line:
            setup = line[line.index("ATLAS_SETUP"):]
        elif "ATLAS_DONE" in line:
            m = re.search(r"attempted=(\d+) residual=(\d+)", line)
            if m:
                attempted += int(m.group(1))
                residual += int(m.group(2))

# Write merged raw (UTF-8): one header, then all rows, then an aggregate DONE.
with open(OUT, "w", encoding="utf-8") as f:
    if setup:
        f.write(setup + "\n")
    for r in rows:
        f.write(r + "\n")
    f.write(f"ATLAS_DONE_AGGREGATE shards={len(SHARDS)} attempted={attempted} residual={residual} rows={len(rows)}\n")


def field(line, key):
    m = re.search(rf" {key}=([^\s]+)", line)
    return m.group(1) if m else None


status = {"WIN": 0, "LOSS": 0, "UNKNOWN": 0}
certified = 0
by_depth = {}
nodes = []
maxrow = None
for r in rows:
    s = field(r, "status")
    status[s] = status.get(s, 0) + 1
    if field(r, "certified") == "1":
        certified += 1
    d = int(field(r, "source_prefix"))
    by_depth[d] = by_depth.get(d, 0) + 1
    n = int(field(r, "nodes"))
    nodes.append(n)
    if maxrow is None or n > int(field(maxrow, "nodes")):
        maxrow = r

nodes.sort()
n = len(nodes)
print("shards:", len(SHARDS), "attempted:", attempted, "residual:", residual, "rows:", len(rows))
print("status:", status, "certified:", certified)
print("by_depth:", {k: by_depth[k] for k in sorted(by_depth)})
print("nodes min/median/max:", nodes[0], nodes[n // 2], nodes[-1], "sum:", sum(nodes))
print("nodes>100:", sum(1 for x in nodes if x > 100), "nodes>10000:", sum(1 for x in nodes if x > 10000),
      "nodes>=2000000:", sum(1 for x in nodes if x >= 2000000))
print("deepest-search row:")
print("  ", maxrow)
