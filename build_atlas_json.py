import json, os, gzip, re, hashlib

BASE = r"E:/Hexo-BotTrainer-hexgt/.claude/worktrees/opening-atlas"
RAW_PASS1 = os.path.join(BASE, "OPENING_ATLAS_PASS1_RAW.txt")
# Corpus first-7-ply layers, applied in order. pass1 is authoritative for its
# ids; corpus layers stack with a NEVER-DOWNGRADE rule (a certified WIN/LOSS is
# never replaced by an UNKNOWN), so the deep vcf layer backfills any position
# the wider round3 layer left UNKNOWN or never reached, and round3 upgrades any
# newly-proven win.
# The deepest available corpus first-N run (vcf + unbounded horizon, 8M cap)
# supersedes the shallower sets and carries the win_line PV on every certified
# win. Later (deeper) raws take precedence; shallower ones are fallbacks only.
RAW_DEEP11 = os.path.join(BASE, "OPENING_ATLAS_CORPUS11_DEEP_RAW.txt")
RAW_DEEP9 = os.path.join(BASE, "OPENING_ATLAS_CORPUS9_DEEP_RAW.txt")
RAW_DEEP7 = os.path.join(BASE, "OPENING_ATLAS_CORPUS7_DEEP_RAW.txt")
CORPUS_LAYERS = next(([p] for p in (RAW_DEEP11, RAW_DEEP9, RAW_DEEP7) if os.path.exists(p)), [])
OUT_DIR = os.path.join(BASE, "atlas-web", "data")
OUT = os.path.join(OUT_DIR, "atlas.json")           # full doc, kept for selfcheck.mjs
OUT_JSONP = os.path.join(OUT_DIR, "atlas.jsonp.js")  # legacy shim — removed on build
IDX_BASE = os.path.join(OUT_DIR, "atlas-index")      # light browse index (loaded up front)
DET_BASE = os.path.join(OUT_DIR, "atlas-details")    # per-id detail store (lazy, one fetch)
# Served frequencies: a SLIM (counts-only) gzipped derivative of the full
# frequencies.json emitted by build_frequencies.mjs. The site only reads
# .counts / .total_games, so by_depth (~half the file) is dropped from the wire
# and the doc is pre-gzipped like the other split docs. frequencies.json itself
# is left untouched (kept for provenance / external analysis).
FREQ_SRC = os.path.join(OUT_DIR, "frequencies.json")
FREQ_WEB_BASE = os.path.join(OUT_DIR, "frequencies-web")
# UI code the cache-bust token must cover (any edit here must re-fetch, not run a
# stale cached module/stylesheet). Keyed alongside the generated data in the hash.
CODE_FILES = [os.path.join(BASE, "atlas-web", n)
              for n in ("atlas.js", "board.js", "d6.js", "mini-board.js", "style.css")]
INDEX_HTML = os.path.join(BASE, "atlas-web", "index.html")
COMMIT = "db96d1b136021212ef32e1f1fdf747bc2262e1c7"

# The light index carries only the fields the browse list / sort / search /
# sharp card / build-lookup / mini-board / canonicalId touch synchronously.
# win_line_terminal is the ONE detail-tier field promoted into the index: it lets
# boot() pick a first showcase win whose forced line actually completes a six
# (terminal==1) WITHOUT first paying the lazy details fetch. It is emitted only
# for the rows that carry it (certified wins) — see _index_row below — so the
# 35k UNKNOWN rows never gain a null field.
INDEX_FIELDS = ["id", "moves", "status", "side", "claimant",
                "placements", "orbit", "source", "certified", "phase",
                "win_line_terminal"]
# The lazy detail store holds every field renderDetails()/setupScrub() read that
# is NOT already in the index. expansions/ms/tt_bytes/peak_tt_bytes are dropped
# from both split files (the UI never reads them); they remain in atlas.json.
DETAIL_FIELDS = ["win_line", "win_line_len", "win_line_terminal",
                 "cap", "horizon", "derived_horizon", "nodes",
                 "cert_nodes", "cert_edges", "cert_commutations", "cert_zones",
                 "cert_fnv1a64_debug_v1", "d6_verified", "d6_mask"]

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

# ---- Split index / details docs ----
def _index_row(r):
    d = {}
    for k in INDEX_FIELDS:
        if k not in r:
            continue
        # keep the index lean: only wins carry win_line_terminal (0/1); drop the
        # None it holds on every UNKNOWN/LOSS row so it costs nothing there.
        if k == "win_line_terminal" and r[k] is None:
            continue
        d[k] = r[k]
    return d
index_rows = [_index_row(r) for r in rows]
index_doc = {
    "schema": doc["schema"],
    "generated_from": doc["generated_from"],
    "census": doc["census"],
    "summary": doc["summary"],
    "corpus7": doc["corpus7"],
    "sharp_examples": doc["sharp_examples"],
    "rows": index_rows,
}
details = {r["id"]: {k: r[k] for k in DETAIL_FIELDS if k in r} for r in rows}

def _compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def write_split(base, obj, global_name):
    """Emit <base>.json (plain), <base>.json.gz (served fast path, inflated in
    JS via DecompressionStream), and <base>.jsonp.js (window global — the
    file:// fallback). All compact. Returns (raw_bytes, gz_bytes)."""
    txt = _compact(obj)
    raw = txt.encode("utf-8")
    gz = gzip.compress(raw, 9)
    with open(base + ".json", "wb") as f:
        f.write(raw)
    with open(base + ".json.gz", "wb") as f:
        f.write(gz)
    with open(base + ".jsonp.js", "w", encoding="utf-8") as f:
        f.write("window." + global_name + " = " + txt + ";\n")
    return len(raw), len(gz)

os.makedirs(OUT_DIR, exist_ok=True)
# Full doc: kept ONLY for selfcheck.mjs (id round-trip + counts). Compact now
# (was indent=2); selfcheck is whitespace-agnostic. The browser never fetches it.
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
# Old monolithic shim is superseded by the two split shims — remove it so a
# stale pre-win_line copy can never be served under the old filename.
if os.path.exists(OUT_JSONP):
    os.remove(OUT_JSONP)

idx_raw, idx_gz = write_split(IDX_BASE, index_doc, "__ATLAS_INDEX__")
det_raw, det_gz = write_split(DET_BASE, details, "__ATLAS_DETAILS__")
atlas_bytes = os.path.getsize(OUT)

# Slim served frequencies (counts only), pre-gzipped. Derived from the full
# frequencies.json if present; the site degrades gracefully when it is absent.
freq_web_raw = freq_web_gz = 0
if os.path.exists(FREQ_SRC):
    with open(FREQ_SRC, encoding="utf-8") as f:
        _fq = json.load(f)
    freq_web = {
        "schema": _fq.get("schema", 1),
        "total_games": _fq.get("total_games"),
        "generated_from": _fq.get("generated_from"),
        "counts": _fq.get("counts", {}),
    }
    freq_web_raw, freq_web_gz = write_split(FREQ_WEB_BASE, freq_web, "__ATLAS_FREQ__")

# Content-derived cache-bust token: sha1 over the served data (index + details +
# slim frequencies) AND the UI code modules. ANY change to code or data flips the
# token, so the browser can never re-run a stale atlas.js/board.js/etc. (COMMIT
# stays the *provenance* stamp — the solver commit the certificates were minted
# against — shown as "minted from" in the census; it is NOT a cache key.)
def _sha1_files(paths):
    h = hashlib.sha1()
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()
VERSION = _sha1_files(
    [IDX_BASE + ".json", DET_BASE + ".json"] +
    ([FREQ_WEB_BASE + ".json"] if freq_web_raw else []) +
    CODE_FILES)

# Stamp the cache-busting version token into the served atlas.js URL so a
# rebuilt site can never re-run a stale (pre-win_line) atlas.js module. The
# dynamically-imported board/d6/mini-board modules inherit the same token from
# atlas.js's own URL, so one stamp busts the whole module graph.
_html = open(INDEX_HTML, encoding="utf-8").read()
_new_html = re.sub(r'src="atlas\.js(?:\?v=[^"]*)?"', f'src="atlas.js?v={VERSION}"', _html)
_new_html = re.sub(r'href="style\.css(?:\?v=[^"]*)?"', f'href="style.css?v={VERSION}"', _new_html)
if _new_html != _html:
    open(INDEX_HTML, "w", encoding="utf-8").write(_new_html)

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
    "out_atlas_json_bytes": atlas_bytes,
    "index_raw_bytes": idx_raw, "index_gz_bytes": idx_gz,
    "details_raw_bytes": det_raw, "details_gz_bytes": det_gz,
    "freq_web_raw_bytes": freq_web_raw, "freq_web_gz_bytes": freq_web_gz,
    "initial_wire_gz_bytes": idx_gz + freq_web_gz,   # up-front, before first select
    "provenance_commit": COMMIT,
    "cache_bust_version": VERSION,
    "files": [OUT, IDX_BASE + ".json(.gz|p.js)", DET_BASE + ".json(.gz|p.js)",
              FREQ_WEB_BASE + ".json(.gz|p.js)"],
}, indent=2))
