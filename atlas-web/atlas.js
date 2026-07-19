/* atlas.js — Hexo Opening Atlas Explorer.
 *
 * Loads data/atlas.json, renders a browsable/filterable list, replays positions
 * on the showcase hex board, and reproduces the D6-canonical FNV-1a-64 id so a
 * user-built opening can be looked up against the certified atlas — honestly.
 *
 * The canonicalization here is byte-for-byte the identity the certificates were
 * minted against (root_position_key -> Rust {:?} Debug -> FNV-1a-64), validated
 * to reproduce every stored row id (see selfcheck.mjs).
 */
/* Code + data are cache-busted with a single version token, read from this
 * module's own URL (index.html stamps atlas.js?v=<commit> at build time). The
 * board/d6/mini-board modules are loaded DYNAMICALLY below so they inherit the
 * token — the stale-atlas.js bug (old code bounding the slider to the opening)
 * can never resurface once the token changes. Under file:// the query is
 * dropped (local files ignore it; a query can break file resolution). */
const VER = (() => {
  try { return new URL(import.meta.url).searchParams.get("v") || ""; }
  catch (_) { return ""; }
})();
const FILE = location.protocol === "file:";
const VERQ = (!FILE && VER) ? ("?v=" + encodeURIComponent(VER)) : "";
const vq = base => base + VERQ;                 // base carries no existing query

// Bound at boot by loadModules() — every reference lives inside a function that
// runs only after boot() awaits the dynamic imports below.
let createBoard, findWin, findThreats, bestClaimantWindow, ownerAt, deriveBinding, canonicalId, miniBoardSVG;
async function loadModules() {
  const [B, D, M] = await Promise.all([
    import("./board.js" + VERQ),
    import("./d6.js" + VERQ),
    import("./mini-board.js" + VERQ),
  ]);
  createBoard = B.createBoard; findWin = B.findWin; findThreats = B.findThreats;
  bestClaimantWindow = B.bestClaimantWindow;
  ownerAt = D.ownerAt; deriveBinding = D.deriveBinding; canonicalId = D.canonicalId;
  miniBoardSVG = M.miniBoardSVG;
}

const $ = id => document.getElementById(id);

/* ------------------------------------------------------------------ *
 * Data loading — split into a light browse index (up front) and a lazy
 * per-id detail store (one fetch on first select). Each doc has three forms:
 * pre-gzipped .json.gz (served fast path, inflated via DecompressionStream),
 * plain .json (fetch fallback), and a window-global .jsonp.js (file:// shim).
 * ------------------------------------------------------------------ */
async function loadDoc(base, globalName) {
  // 1. gzip fast path (a dumb static server won't compress; we ship pre-gzipped)
  if (!FILE && typeof DecompressionStream !== "undefined") {
    try {
      const res = await fetch(vq(base + ".json.gz"), { cache: "no-store" });
      if (res.ok && res.body) {
        const txt = await new Response(
          res.body.pipeThrough(new DecompressionStream("gzip"))).text();
        return JSON.parse(txt);
      }
    } catch (_) { /* fall through */ }
  }
  // 2. plain json
  if (!FILE) {
    try {
      const res = await fetch(vq(base + ".json"), { cache: "no-store" });
      if (res.ok) return await res.json();
    } catch (_) { /* file:// or blocked — fall through to the shim */ }
  }
  // 3. file:// — a bare fetch of a local file is blocked; load the JSONP shim.
  return await new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = vq(base + ".jsonp.js");
    s.onload = () => window[globalName]
      ? resolve(window[globalName])
      : reject(new Error("shim empty (" + globalName + ")"));
    s.onerror = () => reject(new Error(
      "could not load " + base + " (need a static server, or " + base + ".jsonp.js)"));
    document.head.appendChild(s);
  });
}

async function loadIndex() { return loadDoc("data/atlas-index", "__ATLAS_INDEX__"); }

/* Lazy detail store: fetched once per session on the first row select, then
 * memoized. Returns a Map(id -> detailObj). */
let _detailsPromise = null, DETAILS = null;
function ensureDetails() {
  if (!_detailsPromise)
    _detailsPromise = loadDoc("data/atlas-details", "__ATLAS_DETAILS__")
      .then(obj => { DETAILS = new Map(Object.entries(obj)); return DETAILS; });
  return _detailsPromise;
}
// Merge a row's lazy detail fields onto it (idempotent). renderDetails/setupScrub
// read these; before the merge they're absent and both degrade gracefully.
function mergeDetails(row) {
  if (row._merged) return;
  const d = DETAILS && DETAILS.get(row.id);
  if (d) { Object.assign(row, d); row._merged = true; }
}

/* Human-game usage counts (D6-collapsed), keyed by canonical atlas id.
 * Served as the slim, pre-gzipped frequencies-web doc (counts only; the full
 * frequencies.json with its unused by_depth map never hits the wire). Uses the
 * same gzip / plain / jsonp loadDoc path as the atlas docs, so it is compact,
 * cache-busted, AND works under file://. Optional: degrades to "no counts"
 * (badge hidden, freq sort inert) if the doc is absent. */
async function loadFrequencies() {
  try {
    return await loadDoc("data/frequencies-web", "__ATLAS_FREQ__");
  } catch (_) { /* missing — degrade gracefully */ return null; }
}

/* ------------------------------------------------------------------ *
 * App
 * ------------------------------------------------------------------ */
let ATLAS = null;
let INDEX = null;            // Map(id -> row)
let board = null;
let mode = "view";          // "view" | "build"
let buildMoves = [];        // [[q,r],...] placed in build mode
let selectedId = null;
let lastPlaceT = 0;
let staged = null;

/* ---- move-history slider state (scrub the selected opening) ---- */
let scrubRow = null;         // row whose placements the slider scrubs
let openingN = 0;            // placements in the opening itself
let totalN = 0;              // opening + forced-win continuation
let fullMoves = [];          // row.moves concat the (validated) win_line
let winLineArr = [];         // validated forced-win continuation, or []
let claimantIdx = null;      // 0/1 index of the proven winner, or null
let scrubK = 0;              // current slider position (stones shown)
let playTimer = null;        // auto-advance interval id, or null when paused
const PLAY_MS = 700;         // autoplay step interval

/* ---- mini-board icons — built on demand for the visible window only ---- */
const MINI_PX = 44;
const FREQ_FIELD = "freq";        // corpus usage count attached per row from frequencies.json
const _miniCache = new Map();     // row.id -> svg string (insertion-ordered LRU)
const MINI_CACHE_MAX = 500;       // cap so scrolling 35k unknown rows can't retain 35k SVGs
function miniSVGFor(row) {
  let svg = _miniCache.get(row.id);
  if (svg) return svg;
  svg = miniBoardSVG(row.moves, MINI_PX);
  _miniCache.set(row.id, svg);
  if (_miniCache.size > MINI_CACHE_MAX)
    _miniCache.delete(_miniCache.keys().next().value);   // evict oldest
  return svg;
}

function toStones(moves) {
  return moves.map(([q, r], i) => ({ q, r, color: ownerAt(i) }));
}

function toast(msg, err) {
  const wrap = $("toastWrap");
  const t = document.createElement("div");
  t.className = "toast" + (err ? " err" : "");
  t.textContent = msg;
  wrap.appendChild(t);
  setTimeout(() => t.remove(), 2200);
}

/* ---- summary / census ---- */
function fillSummary() {
  const s = ATLAS.summary, c = ATLAS.census;
  $("totPos").textContent = s.total.toLocaleString();
  $("totWin").textContent = s.win.toLocaleString();
  $("totLoss").textContent = s.loss.toLocaleString();
  $("totUnknown").textContent = s.unknown.toLocaleString();
  $("totCert").textContent = s.certified.toLocaleString();
  $("cPly2raw").textContent = c.ply2_raw.toLocaleString();
  $("cPly2d6").textContent = c.ply2_d6.toLocaleString();
  $("cPly3raw").textContent = c.ply3_raw.toLocaleString();
  $("cPly3d6").textContent = c.ply3_d6.toLocaleString();
  $("cGen").textContent = (ATLAS.generated_from || "").slice(0, 12);

  // Vocabulary-note figures — derived from the same data the tiles use, so they
  // can never drift from the totals above (was hardcoded "7-ply / 226 / 269").
  const wins = ATLAS.rows.filter(r => r.status === "WIN");
  const humanWins = wins.filter(r => r.source.split(":")[0] === "human").length;
  const corpusWins = wins.length - humanWins;
  const byDepth = (ATLAS.corpus7 && ATLAS.corpus7.by_depth) || {};
  const depths = Object.keys(byDepth).map(Number).filter(n => !isNaN(n)).sort((a, b) => a - b);
  const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
  set("vnDepthRange", depths.length ? `${depths[0]}–${depths[depths.length - 1]}` : "—");
  set("vnCorpusWins", corpusWins.toLocaleString());
  set("vnHumanWins", humanWins.toLocaleString());
  set("vnLoss", s.loss.toLocaleString());
  set("vnCertTotal", s.certified.toLocaleString());
}

/* ---- sharp example card ---- */
function fillSharp() {
  const flip = (ATLAS.sharp_examples || []).find(e => e.kind === "verdict_flip");
  if (!flip) { $("sharpCard").style.display = "none"; return; }
  // colour the verdict pill by the player the text names, straight from data,
  // so a future flip direction keeps colour matched to text.
  const vClass = v => "sc-v " + (/\bP1\b/.test(v) ? "win-p1" : "win-p0");
  $("scGame").textContent = flip.game;
  $("scBefore").textContent = flip.before.verdict;
  $("scBefore").className = vClass(flip.before.verdict);
  $("scBeforeSub").textContent = `prefix ${flip.before.prefix} · ${flip.before.side} · ${flip.before.phase}`;
  $("scAfter").textContent = flip.after.verdict;
  $("scAfter").className = vClass(flip.after.verdict);
  $("scAfterSub").textContent = `prefix ${flip.after.prefix} · ${flip.after.side} · ${flip.after.phase}`;
  $("scMove").textContent = `plays (${flip.flip_move[0]},${flip.flip_move[1]})`;
  $("scDesc").textContent = flip.description;

  // Locate the "before" row (same game, prefix = source_ply) to load on click.
  const beforeRow = ATLAS.rows.find(r =>
    r.source.includes(flip.game) && r.placements === flip.before.prefix);
  const load = () => {
    if (beforeRow) { setMode("view"); selectRow(beforeRow.id, true); toast("loaded verdict-flip · before"); }
    else toast("before-position row not found", true);
  };
  $("sharpCard").addEventListener("click", load);
  $("sharpCard").addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); load(); } });
}

/* ---- list rendering ---- */
function verdictClass(row) {
  if (row.status === "WIN") return "win";
  if (row.status === "LOSS") return "loss";
  return "unknown";
}
// A certified DECISIVE row (WIN or LOSS) carries the claimant's forced-win line:
// for a WIN the side-to-move is the claimant; for a LOSS the claimant is the
// side-to-move's OPPONENT, and the same win_line is the full solution by which
// that opponent forces the win against the (losing) side-to-move. Both render
// their solution identically through the scrubber; UNKNOWN has none.
function isDecisive(row) { return row.status === "WIN" || row.status === "LOSS"; }
function sourceLabel(row) {
  // "human:<hash>:winner=-1" -> "human <hash>"; "shallow:empty" -> "shallow empty"
  const parts = row.source.split(":");
  if (parts[0] === "human") return `human ${(parts[1] || "").slice(0, 8)}`;
  if (parts[0] === "shallow") return `shallow ${parts[1] || ""}`;
  return row.source.slice(0, 22);
}

function buildSourceFilter() {
  const kinds = new Set();
  for (const r of ATLAS.rows) kinds.add(r.source.split(":")[0]);
  const sel = $("fSource");
  for (const k of [...kinds].sort()) {
    const o = document.createElement("option");
    o.value = k; o.textContent = k + " sources";
    sel.appendChild(o);
  }
}

function currentFilter() {
  return {
    q: $("fSearch").value.trim().toLowerCase(),
    verdict: $("fVerdict").value,
    source: $("fSource").value,
  };
}

function currentSort() {
  const el = $("fSort");
  return (el && el.value) || "freq";
}

function rowMatches(row, f) {
  if (f.verdict === "win" && row.status !== "WIN") return false;
  if (f.verdict === "loss" && row.status !== "LOSS") return false;
  if (f.verdict === "unknown" && row.status !== "UNKNOWN") return false;
  if (f.verdict === "certified" && row.certified !== 1) return false;
  if (f.source && row.source.split(":")[0] !== f.source) return false;
  if (f.q && !row._hay.includes(f.q)) return false;   // _hay precomputed once at boot
  return true;
}

// Sort orders for the browse list. Default "freq" ranks by human-game usage
// (most-played openings first); ties break to the deeper/canonical row so the
// depth-1 origin (6902) precedes the empty root (6902), matching the census.
const _freq = r => (r[FREQ_FIELD] || 0);
function sortRows(rows, mode) {
  mode = mode || "freq";
  const rank = r => r.status === "WIN" ? 0 : r.status === "LOSS" ? 1 : 2;
  const cmp = {
    freq: (a, b) => (_freq(b) - _freq(a)) || (b.placements - a.placements) ||
      (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
    verdict: (a, b) => (rank(a) - rank(b)) || (b.placements - a.placements) ||
      (_freq(b) - _freq(a)),
    depth: (a, b) => (b.placements - a.placements) || (_freq(b) - _freq(a)) ||
      (rank(a) - rank(b)),
  }[mode] || cmp_freq_fallback;
  return rows.slice().sort(cmp);
}
function cmp_freq_fallback(a, b) { return (_freq(b) - _freq(a)) || (b.placements - a.placements); }

/* ---- virtualized browse list -----------------------------------------------
 * Rows are a uniform height, so the list renders only the rows intersecting the
 * scroll viewport into a small recycled node pool absolutely positioned inside a
 * full-height spacer. Filter/sort/search recompute the id array (VIEW) once; a
 * rAF-throttled scroll repaints the window. This keeps the all/unknown tiles
 * (35k+ rows) from ever building 35k DOM nodes. */
const ROW_PITCH = 60;             // 58px row (44px mini + 6px×2 pad + 1px×2 border) + 2px gap
const OVERSCAN = 6;               // rows rendered beyond the viewport each side
let VIEW = [];                    // filtered+sorted rows currently browsable
let _pool = [];                   // reusable .arow buttons
let _spacer = null;               // full-height inner scroll spacer
let _win = { a: -1, b: -1 };      // last rendered [first, last) window
let _scrollRAF = 0;

function ensureListDOM() {
  if (_spacer) return;
  const list = $("atlasList");
  list.textContent = "";
  list.style.display = "block";       // was flex column; we position rows ourselves
  list.style.position = "relative";
  _spacer = document.createElement("div");
  _spacer.style.position = "relative";
  _spacer.style.width = "100%";
  list.appendChild(_spacer);
  list.addEventListener("scroll", () => {
    if (_scrollRAF) return;
    _scrollRAF = requestAnimationFrame(() => { _scrollRAF = 0; renderWindow(); });
  }, { passive: true });
}

function buildRowNode() {
  const b = document.createElement("button");
  b.className = "arow";
  b.style.position = "absolute";
  b.style.left = "0";
  b.style.right = "0";
  b.innerHTML =
    `<span class="mini-slot"></span>` +
    `<span class="a-main"><span class="a-label"></span><span class="a-sub"></span></span>` +
    `<span class="a-right"><span class="a-badge"></span><span class="a-freq"></span></span>`;
  b._slot = b.querySelector(".mini-slot");
  b._label = b.querySelector(".a-label");
  b._sub = b.querySelector(".a-sub");
  b._badge = b.querySelector(".a-badge");
  b._freq = b.querySelector(".a-freq");
  return b;
}

function fillRowNode(b, row, top) {
  b.style.top = top + "px";
  b.dataset.id = row.id;
  b.classList.toggle("sel", row.id === selectedId);
  b._label.textContent = sourceLabel(row);
  b._sub.textContent = `${row.placements}st · ×${row.orbit}`;
  // colour the WIN badge by the CLAIMANT (p0 blue / p1 red) so a "P1 WIN" badge
  // isn't tinted blue while P1's stones are red; LOSS/UNKNOWN keep their class.
  b._badge.className = "a-badge " + verdictClass(row) +
    (row.status === "WIN" && row.claimant === "P1" ? " c-p1" : "");
  // primary browse fact = who wins; UNKNOWN just states the verdict
  b._badge.textContent = row.status === "WIN" ? `${row.claimant} WIN`
    : row.status === "LOSS" ? `${row.side} LOSS`
    : row.status;
  const freq = row[FREQ_FIELD];
  b._freq.textContent = freq ? `used ${freq.toLocaleString()}×` : "";
  if (b._slot._id !== row.id) {          // only rebuild the SVG when the row changes
    b._slot.innerHTML = miniSVGFor(row);
    b._slot._id = row.id;
  }
}

function renderWindow() {
  if (!_spacer) return;
  const list = $("atlasList");
  const vh = list.clientHeight || 400;
  const first = Math.max(0, Math.floor(list.scrollTop / ROW_PITCH) - OVERSCAN);
  const last = Math.min(VIEW.length, Math.ceil((list.scrollTop + vh) / ROW_PITCH) + OVERSCAN);
  if (first === _win.a && last === _win.b) return;
  _win = { a: first, b: last };
  const need = last - first;
  while (_pool.length < need) { const b = buildRowNode(); _pool.push(b); _spacer.appendChild(b); }
  for (let i = 0; i < _pool.length; i++) {
    const b = _pool[i], idx = first + i;
    if (idx < last) { b.style.display = ""; fillRowNode(b, VIEW[idx], idx * ROW_PITCH); }
    else { b.style.display = "none"; b.dataset.id = ""; }
  }
}

function renderList() {
  const f = currentFilter();
  VIEW = sortRows(ATLAS.rows.filter(r => rowMatches(r, f)), currentSort());
  $("filterCount").textContent = `${VIEW.length} of ${ATLAS.rows.length} shown`;
  ensureListDOM();
  _spacer.style.height = (VIEW.length * ROW_PITCH) + "px";
  $("atlasList").scrollTop = 0;
  _win = { a: -1, b: -1 };
  renderWindow();
}

// Only the small visible pool exists in the DOM; toggle sel across it.
function markSelection() {
  for (const b of _pool) {
    if (b.style.display === "none") continue;
    b.classList.toggle("sel", b.dataset.id === selectedId);
  }
}

/* ---- selecting / rendering a stored position ---- */
function setSideFrame(side) {
  const f = $("boardFrame");
  f.classList.remove("side-p0", "side-p1");
  if (side === "P0" || side === 0) f.classList.add("side-p0");
  else if (side === "P1" || side === 1) f.classList.add("side-p1");
}

/* Empty completion cells of the claimant's leading four(s) in the CURRENT scrub
 * position (fullMoves = opening + validated win_line). Cheap (<100 stones) and
 * purely descriptive — the WIN is proven by the certificate, not by this four;
 * this only marks where the claimant is strong. Returns [] when no board-level
 * four exists (a deep win). */
function currentThreatGaps() {
  if (claimantIdx === null || !findThreats) return [];
  return findThreats(toStones(fullMoves), claimantIdx);
}

/* Presentation state of a certified DECISIVE row (WIN or LOSS) — the claimant's
 * forced-win line — keyed off fields already in the data (win_line_terminal +
 * presence of win_line) plus a live threat scan, so it is regeneration-proof and
 * never asserts a six the board doesn't show:
 *   "terminal"  win_line replays to a real six (win_line_terminal === 1)
 *   "threat"    win_line stops before the six at a proven-won, four-holding node
 *   "recorded-absent" no line recorded, but a board-level four is visible
 *   "proof-only"      no line recorded AND no board four — a deep proof cert
 *   "loading"   details still in flight                                       */
function winState(row) {
  if (!isDecisive(row)) return null;
  if (row.win_line_terminal === 1) return "terminal";
  if (winLineArr.length) return "threat";
  if (!row._merged) return "loading";
  return currentThreatGaps().length ? "recorded-absent" : "proof-only";
}

function certShapeNote(row) {
  return (row._merged && typeof row.cert_nodes === "number")
    ? ` (${row.cert_nodes} proof node${row.cert_nodes === 1 ? "" : "s"} / ${row.cert_edges} edge${row.cert_edges === 1 ? "" : "s"})`
    : "";
}

// readout strip — name the OUTCOME honestly (claimant is always the winner) AND
// be explicit about whether the shown board is the six or a proven-won position
// short of it, so a non-terminal certified win never reads as a bald "FORCED
// WIN" on a board with no six.
function renderReadout(row) {
  let verdict;
  if (row.status === "UNKNOWN")
    verdict = "UNKNOWN — no certificate within pass bounds";
  else if (row.status === "WIN")
    verdict = `CERTIFIED WIN — ${row.claimant} (side to move) forces the win`;
  else // LOSS: side-to-move is lost; the claimant is the proven winner
    verdict = `CERTIFIED LOSS — ${row.side} is lost; ${row.claimant} forces the win`;
  $("roK").textContent = verdict;

  let sub = `<b>${sourceLabel(row)}</b> · ${row.placements} stones · ${row.side} to move · ${row.phase}`;
  if (isDecisive(row)) {
    const c = row.claimant;
    // WIN and LOSS share the SAME solution readout (the claimant's forced win);
    // a LOSS just leads with the losing side's framing. `lost` is "" for a WIN,
    // so every existing WIN string is byte-for-byte unchanged.
    const lost = row.status === "LOSS" ? `${row.side} is lost; ` : "";
    switch (winState(row)) {
      case "terminal":
        sub += ` · <span class="ro-win">${lost}${c} forces the win — the line replays all the way to six-in-a-row &rarr;</span>`;
        break;
      case "threat":
        sub += ` · <span class="ro-win">${lost}${c}'s win is certified. The recorded line runs ${winLineArr.length} placements to the proven-won position shown — it stops before the sixth stone, so ${c}'s leading six-in-a-row window is outlined with its completion cells as ghosts; the certificate proves the win continues from here &rarr;</span>`;
        break;
      case "recorded-absent":
        sub += ` · <span class="ro-win">${lost}forced win certified for ${c}${certShapeNote(row)} · the exact line was not recorded, but ${c} already holds a leading six-window on the board (outlined, completion cells ghosted) — the win is proven by the certificate, not by the shown window alone</span>`;
        break;
      case "proof-only":
        sub += ` · <span class="ro-proof">${lost}forced win certified for ${c} by proof structure${certShapeNote(row)} · the forced six is deep — no line was recorded and no board-level four is visible, so only the certificate proves it (not board-obvious)</span>`;
        break;
      default: // loading
        sub += ` · <span class="ro-pending">forced win certified for ${c}; loading the winning line&hellip;</span>`;
    }
  }
  $("roT").innerHTML = sub;
}

function markDetailPending() {
  const body = $("detailBody");
  body.className = "mod-status";
  body.textContent = "loading certificate…";
}

/* Select a row. Detail fields (win_line + the whole certificate readout, plus
 * cap/horizon/nodes) live in the lazy store, so we paint the opening + a pending
 * state immediately, then await the (once-per-session) details fetch and re-run
 * the scrub — this is what extends the slider through the forced win. The
 * selectedId guard drops a stale fetch if a newer selection superseded us. */
async function selectRow(id, reframe) {
  const row = INDEX.get(id);
  if (!row) return;
  selectedId = id;
  markSelection();

  board.setLegal(null);            // read-only in browse
  setupScrub(row);                 // opening-only for now (win_line still absent → graceful)
  renderScrub(totalN);
  if (reframe) board.resetView();
  setSideFrame(row.side);
  renderReadout(row);
  markDetailPending();

  await ensureDetails();
  if (selectedId !== id) return;   // a newer selection won the race
  mergeDetails(row);               // win_line + cert fields now on the row
  setupScrub(row);                 // re-run: slider now extends through the forced win
  renderScrub(totalN);
  renderReadout(row);              // "forces the win in N placements →"
  renderDetails(row);
}

function renderDetails(row) {
  const body = $("detailBody");
  const cell = (k, v, cls, title) =>
    `<dt${title ? ` title="${title}"` : ""}>${k}</dt><dd class="${cls || ""}">${v}</dd>`;
  const na = v => (v === null || v === undefined) ? "—" : v;
  const sideCls = row.side === "P0" ? "p0" : "p1";
  const claimCls = row.claimant === "P0" ? "p0" : row.claimant === "P1" ? "p1" : "muted";

  let html = `<dl class="detail-grid">`;
  html += cell("id", row.id);
  html += cell("status", row.status,
    row.status === "WIN" ? (row.claimant === "P1" ? "p1" : "p0")
      : row.status === "LOSS" ? "p1" : "muted");
  html += cell("winner (proven)", na(row.claimant), claimCls,
    "the player the certificate proves can force the win");
  html += cell("side to move", row.side, sideCls);
  html += cell("phase", row.phase);
  html += cell("source", row.source);
  html += cell("placements", row.placements);
  html += cell("orbit size", "×" + row.orbit, "",
    `this canonical opening represents ${row.orbit} D6-symmetric variant${row.orbit === 1 ? "" : "s"} collapsed into one`);
  html += cell("cap rung", row.cap.toLocaleString(), "", "node-budget tier the search ran under");
  html += cell("horizon", row.horizon, "", "search depth budget (plies)");
  html += cell("derived horizon", na(row.derived_horizon), "muted", "depth the finished proof actually needed");
  html += cell("search nodes", row.nodes.toLocaleString(), "",
    "positions explored to find the proof (larger than the compact certificate)");
  html += `</dl>`;

  // certificate shape
  if (row.certified === 1) {
    const zeroZones = !row.cert_zones;
    html += `<div class="cert-shape">` +
      `<span class="cert-chip" title="positions in the proof itself">cert nodes <b>${row.cert_nodes}</b></span>` +
      `<span class="cert-chip" title="proof moves (edges of the proof tree)">edges <b>${row.cert_edges}</b></span>` +
      `<span class="cert-chip" title="order-independent move swaps folded out of the proof">commutations <b>${row.cert_commutations}</b></span>` +
      `<span class="cert-chip${zeroZones ? " muted" : ""}" title="defender move-set zone reductions used">zones <b>${zeroZones ? "—" : row.cert_zones}</b></span>` +
      `</div>`;
    // D6 audit mask
    const full = row.d6_mask === "0xfff";
    html += `<div class="d6-note${full ? "" : " seam"}" title="raw remap-audit bitmask · ${row.d6_mask}">D6 remap audit · verified ${row.d6_verified}/12 images` +
      (full ? " (all images accepted)."
            : ". Fewer than 12 images pass the <em>remap audit</em> — this is a certificate-remapping seam, NOT a weaker verdict. The strict WIN/LOSS is minted for the exact canonical (symmetry-0) representative.") +
      `</div>`;
    if (row.cert_fnv1a64_debug_v1)
      html += `<div class="d6-note">cert digest · ${row.cert_fnv1a64_debug_v1}</div>`;
  } else {
    html += `<div class="d6-note">Not certified: the solver attempted this root but no certificate finished ` +
      `within the pass bounds (+12 horizon, ≤100k rung). UNKNOWN is not a draw or balance claim.</div>`;
  }
  body.className = "";
  body.innerHTML = html;
}

/* ------------------------------------------------------------------ *
 * Move-history slider — scrub the opening AND its forced-win line
 *
 * Reuses the board renderer: renderScrub(k) draws exactly the first k stones of
 * `fullMoves` (= row.moves + the validated win_line continuation), owners by
 * Hexo turn order via toStones/ownerAt, and rings the k-th. Past the opening the
 * continuation stones are tagged (attacker vs. forced reply) and the true
 * terminal (k === totalN) frames the winning six. win_line is feature-detected:
 * when absent the slider is bounded to the opening exactly as before.
 * ------------------------------------------------------------------ */

/* Validate a per-row win_line: array of integer [q,r] not colliding with the
 * opening or each other. Any malformation => [] (fall back to opening-only).
 * The board is unbounded (virtualized/infinite) and stored openings already
 * reach axial distance ~48 (win_line extends up to ~17 further), so the distance
 * test is only a coarse garbage guard against non-coordinate junk — NOT a board
 * bound. It must sit FAR above any real coordinate, or a legitimate forced-win
 * line that extends outward on a future regeneration would be silently dropped,
 * collapsing a terminal win to an opening-only board with no six and no error.
 * 256 clears the current max (~65) by ~4x while still catching garbage. */
const WINLINE_SANITY_DIST = 256;
function sanitizeWinLine(row, wl) {
  if (!Array.isArray(wl) || !wl.length) return [];
  const occ = new Set(row.moves.map(([q, r]) => q + "," + r));
  const out = [];
  for (const mv of wl) {
    if (!Array.isArray(mv) || mv.length < 2) return [];
    const q = mv[0], r = mv[1];
    if (!Number.isInteger(q) || !Number.isInteger(r)) return [];
    if (Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r)) > WINLINE_SANITY_DIST) return [];
    const k = q + "," + r;
    if (occ.has(k)) return [];
    occ.add(k);
    out.push([q, r]);
  }
  return out;
}

function setupScrub(row) {
  stopPlay();
  scrubRow = row;
  winLineArr = (isDecisive(row) && Array.isArray(row.win_line))
    ? sanitizeWinLine(row, row.win_line) : [];
  openingN = row.moves.length;
  fullMoves = row.moves.concat(winLineArr);
  totalN = fullMoves.length;
  claimantIdx = row.claimant === "P0" ? 0 : row.claimant === "P1" ? 1 : null;
  scrubK = totalN;
  const rng = $("msRange");
  rng.min = "0";
  rng.max = String(totalN);
  rng.step = "1";
  rng.value = String(totalN);
  rng.disabled = totalN === 0;     // empty root: nothing to scrub
  $("moveScrub").hidden = false;
  updateScrubTrack();
  const leg = $("msLegend");
  if (leg) leg.hidden = winLineArr.length === 0;
}

// Two-tone the slider track: opening in --line, forced-win portion in win-blue.
function updateScrubTrack() {
  const rng = $("msRange");
  if (totalN > 0 && winLineArr.length && openingN < totalN) {
    const pct = (openingN / totalN * 100).toFixed(1);
    rng.style.background =
      `linear-gradient(90deg, var(--line) 0 ${pct}%, var(--p0) ${pct}% 100%)`;
  } else {
    rng.style.background = "";
  }
}

function renderScrub(k) {
  const row = scrubRow;
  if (!row) return;
  const N = totalN;
  k = Math.max(0, Math.min(N, k));
  scrubK = k;
  const subset = fullMoves.slice(0, k);
  const stones = toStones(subset);
  const atTerminal = k === N;
  // draw the winning six only at the true terminal of a real forced line (the
  // claimant's line — identical for a WIN and for a LOSS, whose line is the
  // opponent's full solution against the losing side-to-move)
  const winCells = (isDecisive(row) && atTerminal && winLineArr.length)
    ? findWin(stones) : null;
  board.setStones(stones, winCells, openingN, claimantIdx);
  // At the terminal of a certified win that DOESN'T show a six (non-terminal
  // line, or an empty-line human win), outline the claimant's strongest 6-window
  // as a whole with ghost completion cells, so the board shows the FULL six
  // window forming rather than a placed six. Never drawn mid-scrub or when a real
  // six already exists. Null when no board-level four exists (a deep proof).
  const projWin = (isDecisive(row) && atTerminal && !winCells && claimantIdx !== null)
    ? bestClaimantWindow(stones, claimantIdx) : null;
  board.setProjectedWin(projWin, claimantIdx);
  if (!atTerminal && k >= 1) {
    const last = subset[k - 1];
    board.setScrubHighlight({ q: last[0], r: last[1], color: ownerAt(k - 1) });
  } else {
    board.setScrubHighlight(null);   // terminal or empty board: no scrub ring
  }
  const rng = $("msRange");
  if (rng.value !== String(k)) rng.value = String(k);
  // The scrub tag names the claimant's forced win during the solution portion
  // (same for WIN and LOSS — a LOSS shows the opponent's forced win); the opening
  // portion falls back to the row's own verdict ("atlas · loss" / "atlas · win").
  let tag;
  if (isDecisive(row) && atTerminal)
    tag = winCells ? "atlas · forced win · six-in-a-row"
      : projWin ? "atlas · forced win · projected six (proven-won)"
      : "atlas · forced win · proof-certified";
  else if (k > openingN) tag = "atlas · forced win";
  else tag = `atlas · ${row.status.toLowerCase()}`;
  $("atlasTag").textContent = tag;
  updateScrubRead(k);
  updateScrubControls();
}

function updateScrubRead(k) {
  const el = $("msRead");
  let text;
  if (k === 0) {
    text = "start · empty board";
  } else {
    const m = fullMoves[k - 1];
    const owner = ownerAt(k - 1);
    if (k <= openingN) {
      text = `opening · move ${k} / ${openingN} — P${owner} at (${m[0]},${m[1]})`;
    } else {
      const j = k - openingN;
      const role = (claimantIdx !== null && owner === claimantIdx) ? "attacker" : "forced reply";
      text = `forced win · move ${j} / ${winLineArr.length} — P${owner} (${role}) at (${m[0]},${m[1]})`;
    }
  }
  el.textContent = text;
  $("msRange").setAttribute("aria-valuetext", text);   // mirror for screen readers
}

function updateScrubControls() {
  const playing = playTimer !== null;
  $("msPrev").disabled = scrubK <= 0;
  $("msNext").disabled = scrubK >= totalN;
  $("msPlay").disabled = totalN === 0;
  const play = $("msPlay");
  play.innerHTML = playing ? "&#9208;" : "&#9654;";           // ⏸ / ▶
  play.setAttribute("aria-label", playing ? "Pause move history" : "Play move history");
}

function stepScrub(d) { renderScrub(scrubK + d); }

function startPlay() {
  if (totalN === 0) return;
  if (scrubK >= totalN) renderScrub(0);          // at the end: replay from empty
  playTimer = setInterval(() => {
    renderScrub(scrubK + 1);
    if (scrubK >= totalN) stopPlay();            // stop at the terminal (the six)
  }, PLAY_MS);
  updateScrubControls();
}
function stopPlay() {
  if (playTimer !== null) { clearInterval(playTimer); playTimer = null; }
  updateScrubControls();
}
function togglePlay() { (playTimer !== null) ? stopPlay() : startPlay(); }

function wireScrub() {
  $("msRange").addEventListener("input", e => { stopPlay(); renderScrub(+e.target.value); });
  $("msPrev").addEventListener("click", () => { stopPlay(); stepScrub(-1); });
  $("msNext").addEventListener("click", () => { stopPlay(); stepScrub(1); });
  $("msPlay").addEventListener("click", togglePlay);
  // arrow keys step when the slider or the board area holds focus
  $("boardCol").addEventListener("keydown", e => {
    if ($("moveScrub").hidden || e.target === $("msRange")) return;  // range handles its own arrows
    if (e.key === "ArrowLeft") { e.preventDefault(); stopPlay(); stepScrub(-1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); stopPlay(); stepScrub(1); }
  });
}

/* ------------------------------------------------------------------ *
 * Build / test mode
 * ------------------------------------------------------------------ */
function setMode(m) {
  mode = m;
  for (const btn of $("modeSeg").querySelectorAll("button")) {
    const on = btn.dataset.mode === m;
    btn.classList.toggle("sel", on);
    btn.setAttribute("aria-checked", on ? "true" : "false");
  }
  const building = m === "build";
  $("testBtn").disabled = !building || buildMoves.length === 0;
  $("undoBtn").disabled = !building || buildMoves.length === 0;
  $("clearBtn").disabled = !building || buildMoves.length === 0;
  $("modeHint").style.opacity = building ? "1" : ".55";

  if (building) {
    stopPlay();
    $("moveScrub").hidden = true;   // scrubber is a browse-mode affordance
    scrubRow = null;
    selectedId = null;
    markSelection();
    $("atlasTag").textContent = "atlas · build";
    renderBuild(true);
  } else {
    board.setLegal(null);
    $("verdictBig").textContent = "—";
    $("verdictBig").className = "value-big";
    $("verdictCap").textContent = "";
    $("testStatus").textContent = "";
    $("testStatus").className = "mod-status";
  }
}

// Empty on-board cells (dist <= 8) not already occupied — used to gate the ghost.
function legalCells() {
  const occ = new Set(buildMoves.map(([q, r]) => q + "," + r));
  const cells = [];
  for (let q = -8; q <= 8; q++)
    for (let r = -8; r <= 8; r++) {
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r)) > 8) continue;
      if (occ.has(q + "," + r)) continue;
      cells.push({ q, r });
    }
  return cells;
}

function renderBuild(reframe) {
  const stones = toStones(buildMoves);
  board.setStones(stones, null);
  board.setLegal(legalCells());
  if (reframe) board.resetView();

  const { side, phase, first } = deriveBinding(buildMoves);
  const sideStr = side === 0 ? "P0" : "P1";
  setSideFrame(sideStr);

  $("mgStones").textContent = buildMoves.length;
  const tm = $("mgToMove");
  tm.textContent = sideStr;
  tm.className = "n " + (side === 0 ? "is-p0" : "is-p1");
  $("mgPhase").textContent = phase + (phase === "SecondStone" && first ? ` (first ${first[0]},${first[1]})` : "");

  const { id, debug } = canonicalId(buildMoves);
  $("canonId").textContent = id;
  $("canonLine").title = debug;

  $("moveList").textContent = buildMoves.length
    ? buildMoves.map(([q, r], i) => `${i + 1}:${ownerAt(i) === 0 ? "P0" : "P1"}(${q},${r})`).join("  ")
    : "";

  const has = buildMoves.length > 0;
  $("testBtn").disabled = !has;
  $("undoBtn").disabled = !has;
  $("clearBtn").disabled = !has;

  // reset the verdict readout when the position changes
  $("verdictBig").textContent = "—";
  $("verdictBig").className = "value-big";
  $("verdictCap").textContent = "";
  $("testStatus").textContent = has ? "position changed — press Look up verdict" : "";
  $("testStatus").className = "mod-status";
}

function placeStone(q, r) {
  // reject off-board or occupied
  if (Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r)) > 8) { toast("off board", true); return; }
  if (buildMoves.some(([bq, br]) => bq === q && br === r)) return;
  // Hexo forces P0's opening stone to the origin.
  if (buildMoves.length === 0 && !(q === 0 && r === 0)) {
    toast("P0 must open at the origin (0,0)", true);
    return;
  }
  lastPlaceT = Date.now();
  buildMoves.push([q, r]);
  renderBuild(false);
}

function undo() {
  if (!buildMoves.length) return;
  buildMoves.pop();
  renderBuild(false);
}
function clearBuild() {
  buildMoves = [];
  staged = null;
  board.clearStage();
  $("placeChip").classList.remove("show");
  renderBuild(false);
  $("canonId").textContent = "—";
  $("moveList").textContent = "";
}

function clearStage() {
  staged = null;
  board.clearStage();
  $("placeChip").classList.remove("show");
}

async function lookupVerdict() {
  const { id } = canonicalId(buildMoves);
  const row = INDEX.get(id);
  const big = $("verdictBig"), cap = $("verdictCap"), st = $("testStatus");

  if (!row) {
    big.textContent = "UNKNOWN";
    big.className = "value-big";
    cap.textContent = "not in atlas";
    st.textContent = "No certified verdict for this position (not an atlas entry). Not a draw or balance claim.";
    st.className = "mod-status";
    return;
  }
  // Determine the verdict readout, then LOAD the canonical row onto the board in
  // view mode. buildMoves may be a D6 rotation of the certified representative;
  // showing row.moves (+ its forced-win scrubber) instead of the raw build makes
  // the board, the details panel, and the highlighted list row all agree — and,
  // for a WIN, actually surfaces the forced-win line (the whole point).
  let bigT, bigC, capT, stT;
  if (row.status === "WIN" || row.status === "LOSS") {
    bigT = "CERTIFIED " + row.status;
    bigC = "value-big " + (row.status === "WIN" ? "pos" : "neg");
    capT = row.status === "WIN"
      ? `${row.claimant} forces the win`
      : `${row.side} is lost · ${row.claimant} wins`;
    stT = `Strict TssVerifier-accepted · source ${sourceLabel(row)}. Loaded onto the board (canonical D6 representative).`;
  } else {
    bigT = "UNKNOWN";
    bigC = "value-big";
    capT = "in atlas · no certificate";
    stT = "This root is in the atlas but was left UNKNOWN within pass bounds. Not a draw or balance claim.";
  }
  setMode("view");                 // switch to browse (clears the build readout)
  selectRow(row.id, true);         // paints the canonical board + forced-win scrub + details
  big.textContent = bigT;          // re-assert the lookup verdict (view mode reset it)
  big.className = bigC;
  cap.textContent = capT;
  st.textContent = stT;
  st.className = "mod-status ok";
}

/* ------------------------------------------------------------------ *
 * Board wiring
 * ------------------------------------------------------------------ */
function initBoard() {
  board = createBoard($("atlasBoard"), {
    onCellClick(q, r, ptrType) {
      if (mode !== "build") return;
      if (ptrType === "touch") {
        if (staged && staged.q === q && staged.r === r) { clearStage(); placeStone(q, r); }
        else { staged = { q, r }; board.stage(q, r); $("placeChip").classList.add("show"); }
      } else {
        clearStage();
        placeStone(q, r);
      }
    },
    onHover(q, r) {
      $("cursorPos").textContent = q === null || q === undefined ? "—" : q + "," + r;
    },
    ghostAllowed: () => mode === "build",
    onPanStart: () => board.hideHoverGhost(),
    canReset: () => Date.now() - lastPlaceT >= 400,
  });

  // touch: tap the floating chip to confirm the staged cell
  $("placeChip").addEventListener("click", () => {
    if (staged) { const s = staged; clearStage(); placeStone(s.q, s.r); }
  });
}

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */
function selfCheck() {
  // Empty root must hash to its stored id, if present.
  const empty = canonicalId([]);
  const emptyRow = ATLAS.rows.find(r => r.moves.length === 0);
  let matched = 0, tested = 0;
  const sample = ATLAS.rows.filter(r => r.certified === 1).slice(0, 6);
  for (const r of sample) { tested++; if (canonicalId(r.moves).id === r.id) matched++; }
  const okEmpty = !emptyRow || empty.id === emptyRow.id;
  console.log(`[atlas self-check] empty-root id ${okEmpty ? "OK" : "MISMATCH"} (${empty.id}); ` +
    `certified id parity ${matched}/${tested}`);
  if (!okEmpty || matched !== tested)
    toast("self-check parity warning — see console", true);
}

async function boot() {
  try {
    await loadModules();
    ATLAS = await loadIndex();
  } catch (e) {
    document.querySelector(".atlas-main").insertAdjacentHTML("afterbegin",
      `<div class="vocab-note" style="border-left-color:#5c2f2c;color:var(--p1-soft)">Could not load atlas data: ${e.message}. ` +
      `Serve this folder over HTTP (e.g. <b>python -m http.server</b>) or generate data/atlas-index.jsonp.js.</div>`);
    return;
  }
  INDEX = new Map(ATLAS.rows.map(r => [r.id, r]));

  // Precompute the search haystack once (was rebuilt per keystroke over 38k rows).
  for (const r of ATLAS.rows)
    r._hay = (r.id + " " + r.source + " " + r.placements + " " +
      r.moves.map(m => m.join(",")).join(" ")).toLowerCase();

  // Attach human-game usage counts (D6-collapsed) onto each row, keyed by id.
  const FREQ = await loadFrequencies();
  const counts = (FREQ && FREQ.counts) || {};
  for (const r of ATLAS.rows) {
    const c = counts[r.id];
    if (c != null) r[FREQ_FIELD] = c;                // absent (deep/off-scope) => badge hidden
  }

  initBoard();
  wireScrub();
  fillSummary();
  fillSharp();
  buildSourceFilter();
  // land the browse list on the proven openings, not a wall of UNKNOWN
  $("fVerdict").value = "certified";
  renderList();

  // one debounced search + one delegated click for the whole (37.9k-row) list;
  // the list is virtualized, so a filter/search repaints only the visible window
  let searchTimer = null;
  $("fSearch").addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(renderList, 150);
  });
  $("fVerdict").addEventListener("change", renderList);
  $("fSource").addEventListener("change", renderList);
  $("fSort").addEventListener("change", renderList);
  $("atlasList").addEventListener("click", e => {
    const row = e.target.closest(".arow");
    if (row && row.dataset.id) selectRow(row.dataset.id, true);
  });

  // totals tiles double as one-click verdict filters
  $("atlasTotals").addEventListener("click", e => {
    const cell = e.target.closest("[data-filter]");
    if (!cell) return;
    setMode("view");
    $("fVerdict").value = cell.dataset.filter;
    renderList();
  });
  $("atlasTotals").addEventListener("keydown", e => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const cell = e.target.closest("[data-filter]");
    if (!cell) return;
    e.preventDefault();
    setMode("view");
    $("fVerdict").value = cell.dataset.filter;
    renderList();
  });

  for (const btn of $("modeSeg").querySelectorAll("button"))
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  $("testBtn").addEventListener("click", lookupVerdict);
  $("undoBtn").addEventListener("click", undo);
  $("clearBtn").addEventListener("click", clearBuild);

  // land on a shallow, frequently-played certified WIN so the first frame is a
  // proven line that scrubs to six-in-a-row (not the 78-stone endgame). PREFER a
  // win whose forced line actually completes a six (win_line_terminal === 1,
  // promoted into the index) — many PV lines end at a proven-won node without the
  // six placed, and opening on one of those looks like "the win isn't shown".
  const wins = ATLAS.rows.filter(r => r.status === "WIN");
  const sixWins = wins.filter(r => r.win_line_terminal === 1);
  const first = (sixWins.length ? sixWins : wins).slice().sort((a, b) =>
    (a.placements - b.placements) || (_freq(b) - _freq(a)) ||
    (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))[0] ||
    sortRows(ATLAS.rows, "freq")[0];
  if (first) selectRow(first.id, true);

  selfCheck();
}

boot();
