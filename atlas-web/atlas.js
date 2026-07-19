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
import { createBoard, findWin } from "./board.js";
import { ownerAt, deriveBinding, canonicalId } from "./d6.js";
import { miniBoardSVG } from "./mini-board.js";

const $ = id => document.getElementById(id);

/* ------------------------------------------------------------------ *
 * Data loading (fetch on a server; JSONP shim for file://)
 * ------------------------------------------------------------------ */
async function loadAtlas() {
  try {
    const res = await fetch("data/atlas.json", { cache: "no-store" });
    if (res.ok) return await res.json();
    throw new Error("http " + res.status);
  } catch (_) {
    // file:// — fetch of a local file is blocked; load the JSONP shim instead.
    return await new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "data/atlas.jsonp.js";
      s.onload = () => window.__ATLAS__ ? resolve(window.__ATLAS__) : reject(new Error("shim empty"));
      s.onerror = () => reject(new Error("could not load atlas data (need a static server, or data/atlas.jsonp.js)"));
      document.head.appendChild(s);
    });
  }
}

/* Human-game usage counts (D6-collapsed), keyed by canonical atlas id.
 * Optional: degrades to "no counts" (badge hidden, freq sort inert) if absent
 * or under file:// where a bare fetch is blocked. */
async function loadFrequencies() {
  try {
    const res = await fetch("data/frequencies.json", { cache: "no-store" });
    if (res.ok) return await res.json();
  } catch (_) { /* file:// or missing — degrade gracefully */ }
  return null;
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
let TOTAL_GAMES = null;      // corpus denominator (from frequencies.json)

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

/* ---- lazy mini-board icons (IntersectionObserver, smooth at ~12k rows) ---- */
const MINI_PX = 44;
const FREQ_FIELD = "freq";      // corpus usage count attached per row from frequencies.json
const _miniCache = new Map();   // row.id -> svg string (survives re-filters)
let _miniObserver = null;

function ensureMiniObserver() {
  if (_miniObserver) return _miniObserver;
  _miniObserver = new IntersectionObserver(entries => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const slot = e.target;
      _miniObserver.unobserve(slot);
      if (slot.firstChild) continue;                 // already filled
      const row = INDEX.get(slot.dataset.id);
      if (!row) continue;
      let svg = _miniCache.get(row.id);
      if (!svg) { svg = miniBoardSVG(row.moves, MINI_PX); _miniCache.set(row.id, svg); }
      slot.innerHTML = svg;
    }
  }, { root: $("atlasList"), rootMargin: "300px 0px" });  // prefetch a screen ahead
  return _miniObserver;
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
  if (f.q) {
    const hay = (row.id + " " + row.source + " " + row.placements + " " +
      row.moves.map(m => m.join(",")).join(" ")).toLowerCase();
    if (!hay.includes(f.q)) return false;
  }
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

function renderList() {
  const f = currentFilter();
  const rows = sortRows(ATLAS.rows.filter(r => rowMatches(r, f)), currentSort());
  const list = $("atlasList");
  const ob = ensureMiniObserver();
  ob.disconnect();                                   // drop stale targets from prior render
  list.textContent = "";
  $("filterCount").textContent = `${rows.length} of ${ATLAS.rows.length} shown`;
  for (const row of rows) {
    const v = verdictClass(row);
    const freq = row[FREQ_FIELD];
    // primary browse fact = who wins; UNKNOWN just states the verdict
    const badge = row.status === "WIN" ? `${row.claimant} WIN`
      : row.status === "LOSS" ? `${row.side} LOSS`
      : row.status;
    const b = document.createElement("button");
    b.className = "arow" + (row.id === selectedId ? " sel" : "");
    b.dataset.id = row.id;
    b.innerHTML =
      `<span class="mini-slot" data-id="${row.id}"></span>` +
      `<span class="a-main">` +
        `<span class="a-label">${sourceLabel(row)}</span>` +
        `<span class="a-sub">${row.placements}st · ×${row.orbit}</span>` +
      `</span>` +
      `<span class="a-right">` +
        `<span class="a-badge ${v}">${badge}</span>` +
        (freq ? `<span class="a-freq">used ${freq.toLocaleString()}×</span>` : ``) +
      `</span>`;
    list.appendChild(b);
    ob.observe(b.firstElementChild);                 // the .mini-slot
  }
}

function markSelection() {
  for (const el of $("atlasList").querySelectorAll(".arow"))
    el.classList.toggle("sel", el.dataset.id === selectedId);
}

/* ---- selecting / rendering a stored position ---- */
function setSideFrame(side) {
  const f = $("boardFrame");
  f.classList.remove("side-p0", "side-p1");
  if (side === "P0" || side === 0) f.classList.add("side-p0");
  else if (side === "P1" || side === 1) f.classList.add("side-p1");
}

function selectRow(id, reframe) {
  const row = INDEX.get(id);
  if (!row) return;
  selectedId = id;
  markSelection();

  board.setLegal(null);            // read-only in browse
  setupScrub(row);                 // move-history slider: opening + forced win
  renderScrub(totalN);             // draw the full line (terminal frames the six)
  if (reframe) board.resetView();
  setSideFrame(row.side);

  // readout strip — name the OUTCOME honestly (claimant is always the winner)
  let verdict;
  if (row.status === "UNKNOWN")
    verdict = "UNKNOWN — no certificate within pass bounds";
  else if (row.status === "WIN")
    verdict = `CERTIFIED WIN — ${row.claimant} (side to move) forces the win`;
  else // LOSS: side-to-move is lost; the claimant is the proven winner
    verdict = `CERTIFIED LOSS — ${row.side} is lost; ${row.claimant} forces the win`;
  $("roK").textContent = verdict;

  let sub = `<b>${sourceLabel(row)}</b> · ${row.placements} stones · ${row.side} to move · ${row.phase}`;
  if (row.status === "WIN") {
    sub += winLineArr.length
      ? ` · <span class="ro-win">${row.claimant} forces the win in ${winLineArr.length} placements &rarr;</span>`
      : ` · <span class="ro-pending">forced win certified for ${row.claimant}; winning line: computing (not in this snapshot)</span>`;
  }
  $("roT").innerHTML = sub;

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
  html += cell("status", row.status, verdictClass(row) === "win" ? "p0" : verdictClass(row) === "loss" ? "p1" : "muted");
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
 * reach axial distance ~32, so the distance test is only a coarse garbage guard
 * — NOT a board bound. It must sit well above real coordinates or a legitimate
 * forced-win line that extends outward would be silently dropped. */
const WINLINE_SANITY_DIST = 64;   // 2x the widest stored opening coordinate
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
  winLineArr = (row.status === "WIN" && Array.isArray(row.win_line))
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
  // draw the winning six only at the true terminal of a real forced line
  const winCells = (row.status === "WIN" && atTerminal && winLineArr.length)
    ? findWin(stones) : null;
  board.setStones(stones, winCells, openingN, claimantIdx);
  if (!atTerminal && k >= 1) {
    const last = subset[k - 1];
    board.setScrubHighlight({ q: last[0], r: last[1], color: ownerAt(k - 1) });
  } else {
    board.setScrubHighlight(null);   // terminal or empty board: no scrub ring
  }
  const rng = $("msRange");
  if (rng.value !== String(k)) rng.value = String(k);
  $("atlasTag").textContent = (k > openingN)
    ? "atlas · forced win"
    : `atlas · ${row.status.toLowerCase()}`;
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

function lookupVerdict() {
  const { id, binding } = canonicalId(buildMoves);
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
  if (row.status === "WIN" || row.status === "LOSS") {
    big.textContent = "CERTIFIED " + row.status;
    big.className = "value-big " + (row.status === "WIN" ? "pos" : "neg");
    cap.textContent = row.status === "WIN"
      ? `${row.claimant} forces the win`
      : `${row.side} is lost · ${row.claimant} wins`;
    st.textContent = `Strict TssVerifier-accepted · source ${sourceLabel(row)}. Selecting its atlas row.`;
    st.className = "mod-status ok";
    // also surface the matching row's details + select it in the list
    selectedId = row.id;
    markSelection();
    renderDetails(row);
  } else {
    big.textContent = "UNKNOWN";
    big.className = "value-big";
    cap.textContent = "in atlas · no certificate";
    st.textContent = "This root is in the atlas but was left UNKNOWN within pass bounds. Not a draw or balance claim.";
    st.className = "mod-status";
    selectedId = row.id;
    markSelection();
    renderDetails(row);
  }
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
    ATLAS = await loadAtlas();
  } catch (e) {
    document.querySelector(".atlas-main").insertAdjacentHTML("afterbegin",
      `<div class="vocab-note" style="border-left-color:#5c2f2c;color:var(--p1-soft)">Could not load atlas data: ${e.message}. ` +
      `Serve this folder over HTTP (e.g. <b>python -m http.server</b>) or generate data/atlas.jsonp.js.</div>`);
    return;
  }
  INDEX = new Map(ATLAS.rows.map(r => [r.id, r]));

  // Attach human-game usage counts (D6-collapsed) onto each row, keyed by id.
  const FREQ = await loadFrequencies();
  const counts = (FREQ && FREQ.counts) || {};
  TOTAL_GAMES = FREQ ? FREQ.total_games : null;
  for (const r of ATLAS.rows) {
    const c = counts[r.id];
    if (c != null) r[FREQ_FIELD] = c;                // absent (deep/off-scope) => badge hidden
  }

  initBoard();
  wireScrub();
  fillSummary();
  fillSharp();
  buildSourceFilter();
  // land the browse list on the 269 proven openings, not a wall of UNKNOWN
  $("fVerdict").value = "certified";
  renderList();

  // one debounced search + one delegated click for the whole (up to ~12.8k) list
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
  // proven line that scrubs to six-in-a-row (not the 78-stone endgame).
  const wins = ATLAS.rows.filter(r => r.status === "WIN");
  const first = wins.slice().sort((a, b) =>
    (a.placements - b.placements) || (_freq(b) - _freq(a)) ||
    (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))[0] ||
    sortRows(ATLAS.rows, "freq")[0];
  if (first) selectRow(first.id, true);

  selfCheck();
}

boot();
