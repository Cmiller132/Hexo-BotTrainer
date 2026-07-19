import glob, os, re

BASE = r"E:/Hexo-BotTrainer-hexgt/.claude/worktrees/opening-atlas"
SHARDS = sorted(glob.glob(os.path.join(BASE, "corpus9_shards", "shard_*.txt")))
OUT = os.path.join(BASE, "OPENING_ATLAS_CORPUS9_DEEP_RAW.txt")


def read_text(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


rows, setup = [], None
attempted = residual = 0
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
                attempted += int(m.group(1)); residual += int(m.group(2))

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
cert = 0
by_depth_win = {}
by_depth_all = {}
wl_terminal = 0
wl_total = 0
nodes = []
maxrow = None
for r in rows:
    s = field(r, "status"); status[s] = status.get(s, 0) + 1
    d = int(field(r, "source_prefix")); by_depth_all[d] = by_depth_all.get(d, 0) + 1
    n = int(field(r, "nodes")); nodes.append(n)
    if maxrow is None or n > int(field(maxrow, "nodes")):
        maxrow = r
    if field(r, "certified") == "1":
        cert += 1
        if s == "WIN":
            by_depth_win[d] = by_depth_win.get(d, 0) + 1
        wl_total += 1
        if field(r, "win_line_terminal") == "1":
            wl_terminal += 1

nodes.sort()
print("shards", len(SHARDS), "attempted", attempted, "residual", residual, "rows", len(rows))
print("status", status, "certified", cert)
print("positions_by_depth", {k: by_depth_all[k] for k in sorted(by_depth_all)})
print("certified_WIN_by_depth", {k: by_depth_win[k] for k in sorted(by_depth_win)})
print("win_line_terminal", wl_terminal, "/", wl_total,
      f"({100*wl_terminal/max(wl_total,1):.1f}% reach literal six-in-a-row)")
print("nodes min/median/max", nodes[0], nodes[len(nodes)//2], nodes[-1])
print("deepest-search row:", maxrow[:200])
