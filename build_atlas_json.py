import json, os

BASE = r"E:/Hexo-BotTrainer-hexgt/.claude/worktrees/opening-atlas"
RAW_PASS1 = os.path.join(BASE, "OPENING_ATLAS_PASS1_RAW.txt")
# Deep 16-core corpus first-7-ply layer (unbounded horizon, 2M node cap,
# 640 MiB TT/worker). Falls back to the earlier shallow corpus7 raw if absent.
RAW_CORPUS7 = os.path.join(BASE, "OPENING_ATLAS_CORPUS7_DEEP_RAW.txt")
if not os.path.exists(RAW_CORPUS7):
    RAW_CORPUS7 = os.path.join(BASE, "OPENING_ATLAS_CORPUS7_RAW.txt")
OUT_DIR = os.path.join(BASE, "atlas-web", "data")
OUT = os.path.join(OUT_DIR, "atlas.json")
OUT_JSONP = os.path.join(OUT_DIR, "atlas.jsonp.js")
COMMIT = "db96d1b136021212ef32e1f1fdf747bc2262e1c7"

INT_FIELDS = {"source_prefix","placements","orbit","cap","horizon","nodes",
              "expansions","tt_bytes","peak_tt_bytes","certified","cert_nodes",
              "cert_edges","cert_commutations","cert_zones","d6_verified"}
FLOAT_FIELDS = {"ms"}
# derived_horizon is int-or-NA; handled specially

def parse_moves(val):
    if val is None or val == "":
        return []
    out = []
    for pair in val.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        q, r = pair.split(",")
        out.append([int(q), int(r)])
    return out

def read_text(path):
    """Raw logs are UTF-16 (PowerShell '>' redirect); fall back to UTF-8-sig."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("utf-16")

def parse_raw(path):
    if not os.path.exists(path):
        return []
    rows = []
    for line in read_text(path).splitlines():
        line = line.rstrip("\r")
        if not line.startswith("ATLAS_ROW "):
            continue
        body = line[len("ATLAS_ROW "):]
        # moves is last; may be empty. Split on spaces (no value contains a space).
        rec = {}
        for tok in body.split(" "):
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            rec[k] = v
        row = {}
        for k, v in rec.items():
            if k == "moves":
                row["moves"] = parse_moves(v)
            elif k == "derived_horizon":
                row[k] = None if v == "NA" else int(v)
            elif k == "cert_fnv1a64_debug_v1":
                row[k] = None if v == "NA" else v
            elif k == "claimant":
                row[k] = None if v == "NA" else v
            elif k in INT_FIELDS:
                row[k] = int(v)
            elif k in FLOAT_FIELDS:
                row[k] = float(v)
            else:
                row[k] = v  # id, source, side, phase, status, d6_mask
        if "moves" not in row:
            row["moves"] = []
        rows.append(row)
    return rows

# ---- Load + merge (pass1 rows first; corpus7 rows only for NEW ids) ----
pass1_rows = parse_raw(RAW_PASS1)
corpus7_rows = parse_raw(RAW_CORPUS7)

seen = set()
rows = []
for r in pass1_rows:
    if r["id"] in seen:
        continue
    seen.add(r["id"])
    rows.append(r)
corpus7_new = 0
corpus7_dupe = 0
for r in corpus7_rows:
    if r["id"] in seen:
        corpus7_dupe += 1
        continue
    seen.add(r["id"])
    rows.append(r)
    corpus7_new += 1

# ---- Verification ----
total = len(rows)
win = sum(1 for r in rows if r["status"] == "WIN")
loss = sum(1 for r in rows if r["status"] == "LOSS")
unknown = sum(1 for r in rows if r["status"] == "UNKNOWN")
certified = sum(1 for r in rows if r["certified"] == 1)
other = total - win - loss - unknown
assert other == 0, f"unexpected statuses: {other}"
assert win + loss + unknown == total, "counts do not add up"
for r in rows:
    if r["certified"] == 1:
        assert r["status"] in ("WIN","LOSS"), f"certified non-decisive: {r['id']}"
for r in rows:
    for m in r["moves"]:
        assert len(m) == 2 and all(isinstance(x,int) for x in m), f"bad move in {r['id']}"

# Per-source-kind and per-depth breakdown for the corpus7 layer.
corpus7_by_depth = {}
for r in rows:
    if r["source"].startswith("corpus7:"):
        d = r["source_prefix"]
        corpus7_by_depth[d] = corpus7_by_depth.get(d, 0) + 1

census = {"ply2_raw": 216, "ply2_d6": 24, "ply3_raw": 42768, "ply3_d6": 3684}

sharp_examples = [
    {
        "kind": "verdict_flip",
        "game": "004759ff34cefdc2",
        "corpus_winner": "P1",
        "description": "Adjacent certificate-backed verdict flip: the proven winner changes from P0 to P1 after a single placement, so (14,-3) is a certified losing blunder in this exact opening.",
        "flip_move": [14, -3],
        "source_ply": 44,
        "before": {"prefix": 44, "side": "P0", "phase": "SecondStone",
                   "verdict": "CERTIFIED P0 WIN", "nodes": 2, "derived_horizon": 49},
        "after": {"prefix": 45, "side": "P1", "phase": "FirstStone",
                  "verdict": "CERTIFIED P1 WIN", "nodes": 1, "derived_horizon": 47},
    },
    {
        "kind": "compact_win",
        "source": "xsnfyll",
        "description": "Compact 13-stone P1 win, certified in only 82 nodes at the 10k rung.",
        "stones": 13, "side": "P1", "verdict": "CERTIFIED WIN", "nodes": 82, "rung": 10000,
        "moves": parse_moves("0,0;-1,0;1,-2;-2,0;1,0;0,-2;1,-3;0,-3;2,-5;2,-4;1,-4;3,-4;3,-2"),
    },
    {
        "kind": "certified_loss",
        "sources": ["8is963b", "dy3dg99"],
        "description": "Genuine dual results: both are P0-to-move CERTIFIED LOSS roots, each resolved in one solver node (not merely NO/UNKNOWN controls).",
        "side": "P0", "verdict": "CERTIFIED LOSS", "nodes": 1,
    },
    {
        "kind": "deepest_new_proof",
        "id": "oa-558f79a590c31b6a",
        "game": "002f5360162bac9b",
        "prefix": 48,
        "description": "Deepest new proof by node count: P0 to move at SecondStone, CERTIFIED WIN in 6,619 nodes (18 certificate nodes, derived T=57); preceding prefix 47 is also a P0 win but needs only 148 nodes.",
        "side": "P0", "phase": "SecondStone", "verdict": "CERTIFIED WIN",
        "nodes": 6619, "cert_nodes": 18, "derived_horizon": 57,
    },
]

doc = {
    "schema": 1,
    "generated_from": COMMIT,
    "census": census,
    "summary": {"total": total, "win": win, "loss": loss,
                "unknown": unknown, "certified": certified},
    "corpus7": {
        "new_rows": corpus7_new,
        "duplicate_of_pass1": corpus7_dupe,
        "by_depth": {str(k): corpus7_by_depth[k] for k in sorted(corpus7_by_depth)},
    },
    "sharp_examples": sharp_examples,
    "rows": rows,
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)
with open(OUT_JSONP, "w", encoding="utf-8") as f:
    f.write("window.__ATLAS__ = ")
    json.dump(doc, f, ensure_ascii=False, indent=2)
    f.write(";\n")

print(json.dumps({
    "pass1_rows": len(pass1_rows),
    "corpus7_rows_parsed": len(corpus7_rows),
    "corpus7_new": corpus7_new,
    "corpus7_dupe_of_pass1": corpus7_dupe,
    "total": total, "win": win, "loss": loss, "unknown": unknown,
    "certified": certified,
    "certified_win": sum(1 for r in rows if r["certified"]==1 and r["status"]=="WIN"),
    "certified_loss": sum(1 for r in rows if r["certified"]==1 and r["status"]=="LOSS"),
    "corpus7_by_depth": {str(k): corpus7_by_depth[k] for k in sorted(corpus7_by_depth)},
    "out": OUT, "out_jsonp": OUT_JSONP,
}, indent=2))
