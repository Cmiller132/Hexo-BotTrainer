import json, os

BASE = r"E:/Hexo-BotTrainer-hexgt/.claude/worktrees/opening-atlas"
RAW_PASS1 = os.path.join(BASE, "OPENING_ATLAS_PASS1_RAW.txt")
# Corpus first-7-ply layers, applied in order. pass1 is authoritative for its
# ids; corpus layers stack with a NEVER-DOWNGRADE rule (a certified WIN/LOSS is
# never replaced by an UNKNOWN), so the deep vcf layer backfills any position
# the wider round3 layer left UNKNOWN or never reached, and round3 upgrades any
# newly-proven win.
# The 9-ply deep run (vcf + unbounded horizon, 8M cap) supersedes the 7-ply
# deep set and carries the win_line PV on every certified win. The 7-ply raw is
# kept only as a fallback if the 9-ply raw is absent.
RAW_DEEP9 = os.path.join(BASE, "OPENING_ATLAS_CORPUS9_DEEP_RAW.txt")
RAW_DEEP7 = os.path.join(BASE, "OPENING_ATLAS_CORPUS7_DEEP_RAW.txt")
CORPUS_LAYERS = [p for p in (RAW_DEEP9,) if os.path.exists(p)] or \
                [p for p in (RAW_DEEP7,) if os.path.exists(p)]
OUT_DIR = os.path.join(BASE, "atlas-web", "data")
OUT = os.path.join(OUT_DIR, "atlas.json")
OUT_JSONP = os.path.join(OUT_DIR, "atlas.jsonp.js")
COMMIT = "db96d1b136021212ef32e1f1fdf747bc2262e1c7"

INT_FIELDS = {"source_prefix","placements","orbit","cap","horizon","nodes",
              "expansions","tt_bytes","peak_tt_bytes","certified","cert_nodes",
              "cert_edges","cert_commutations","cert_zones","d6_verified","win_line_len"}
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
            elif k == "win_line":
                row["win_line"] = [] if v == "NA" else parse_moves(v)
            elif k == "win_line_terminal":
                row[k] = None if v == "NA" else int(v)
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
        # win_line schema (absent in the legacy pass1 raw).
        row.setdefault("win_line", [])
        row.setdefault("win_line_len", len(row["win_line"]))
        row.setdefault("win_line_terminal", None)
        rows.append(row)
    return rows

# ---- Load + layered merge ----
# pass1 ids are authoritative and never overwritten. Corpus layers stack in
# order; within corpus ids, an incoming row replaces the current one iff it is
# certified OR the current one is not certified (never downgrade a verdict).
pass1_rows = parse_raw(RAW_PASS1)
pass1_ids = {r["id"] for r in pass1_rows}
corpus = {}                       # id -> row (best corpus verdict so far)
corpus_order = []                 # preserve first-seen order for stable output
layer_new = {}                    # layer path -> new wins contributed
for path in CORPUS_LAYERS:
    new_wins = 0
    for r in parse_raw(path):
        rid = r["id"]
        if rid in pass1_ids:
            continue
        cur = corpus.get(rid)
        if cur is None:
            corpus[rid] = r
            corpus_order.append(rid)
        elif r["certified"] == 1 or cur["certified"] == 0:
            if r["certified"] == 1 and cur["certified"] == 0:
                new_wins += 1
            corpus[rid] = r
    layer_new[os.path.basename(path)] = new_wins

rows = list(pass1_rows) + [corpus[rid] for rid in corpus_order]
corpus7_new = len(corpus_order)
corpus7_dupe = sum(1 for r in (parse_raw(CORPUS_LAYERS[-1]) if CORPUS_LAYERS else [])
                   if r["id"] in pass1_ids)

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

# win_line: emitted for every certified WIN sourced from a win_line-aware raw
# (the corpus first-N layer). Legacy pass1 certified rows predate the field.
win_line_wins = [r for r in rows if r["certified"] == 1 and r["status"] == "WIN"
                 and r["source"].startswith("corpus")]
for r in win_line_wins:
    assert len(r["win_line"]) > 0, f"certified corpus win without win_line: {r['id']}"
    for m in r["win_line"]:
        assert len(m) == 2 and all(isinstance(x,int) for x in m), f"bad win_line move in {r['id']}"
win_line_terminal = sum(1 for r in win_line_wins if r.get("win_line_terminal") == 1)

# Per-depth breakdown for the corpus first-N layer.
corpus7_by_depth = {}
for r in rows:
    if r["source"].startswith("corpus"):
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
        "layers": [os.path.basename(p) for p in CORPUS_LAYERS],
        "new_wins_by_layer": layer_new,
        "win_line_wins": len(win_line_wins),
        "win_line_terminal": win_line_terminal,
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
    "corpus_layers": [os.path.basename(p) for p in CORPUS_LAYERS],
    "new_wins_by_layer": layer_new,
    "corpus7_new": corpus7_new,
    "corpus7_dupe_of_pass1": corpus7_dupe,
    "total": total, "win": win, "loss": loss, "unknown": unknown,
    "certified": certified,
    "certified_win": sum(1 for r in rows if r["certified"]==1 and r["status"]=="WIN"),
    "certified_loss": sum(1 for r in rows if r["certified"]==1 and r["status"]=="LOSS"),
    "win_line_wins": len(win_line_wins),
    "win_line_terminal": win_line_terminal,
    "corpus7_by_depth": {str(k): corpus7_by_depth[k] for k in sorted(corpus7_by_depth)},
    "out": OUT, "out_jsonp": OUT_JSONP,
}, indent=2))
