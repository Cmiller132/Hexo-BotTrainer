const HEX = 19;
const SQRT3 = Math.sqrt(3);
let state = null;
let tacticsOn = false;
let selectedWindowId = null;
let selectedCellKey = null;
let tacticFilters = { mode: "windows", player: "both", axis: "all", inspect: false };
let pendingRequest = false;
let requestSeq = 0;

const svg = document.getElementById("boardSvg");
const tip = document.getElementById("tip");

document.getElementById("newBtn").addEventListener("click", () => post("/api/new", {}));
document.getElementById("fitBtn").addEventListener("click", render);
document.getElementById("tacticsBtn").addEventListener("click", () => {
  tacticsOn = !tacticsOn;
  if (!tacticsOn) {
    selectedWindowId = null;
    selectedCellKey = null;
  }
  render();
});
document.querySelectorAll("#modeSeg button").forEach(button => {
  button.addEventListener("click", () => { tacticFilters.mode = button.dataset.mode; render(); });
});
document.querySelectorAll("#playerSeg button").forEach(button => {
  button.addEventListener("click", () => { tacticFilters.player = button.dataset.player; render(); });
});
document.querySelectorAll("#axisSeg button").forEach(button => {
  button.addEventListener("click", () => { tacticFilters.axis = button.dataset.axis; render(); });
});
document.getElementById("inspectBtn").addEventListener("click", () => {
  tacticFilters.inspect = !tacticFilters.inspect;
  if (!tacticFilters.inspect) {
    selectedWindowId = null;
    selectedCellKey = null;
  }
  render();
});
window.addEventListener("resize", () => { if (state) render(); });

async function loadState() {
  const res = await fetch("/api/state");
  state = await res.json();
  render();
}

async function post(url, payload) {
  if (pendingRequest) return;
  const seq = ++requestSeq;
  setPending(true);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (seq !== requestSeq) return;
    if (!res.ok) {
      state = data.state;
      document.getElementById("statusText").textContent = data.error || "Illegal move";
    } else {
      state = data;
    }
  } catch (error) {
    if (seq === requestSeq) document.getElementById("statusText").textContent = "Request failed";
  } finally {
    if (seq === requestSeq) {
      setPending(false);
      render();
    }
  }
}

function setPending(value) {
  pendingRequest = value;
  document.body.classList.toggle("pending", value);
  document.querySelectorAll("button").forEach(button => { button.disabled = value; });
}

function playerLabel(player) {
  return player === "player0" ? "Player 0" : "Player 1";
}

function phaseLabel(phase) {
  if (phase === "opening") return "Opening";
  if (phase === "first_stone") return "First stone";
  return "Second stone";
}

function center(q, r) {
  return { x: HEX * SQRT3 * (q + r / 2), y: HEX * 1.5 * r };
}

function path(cx, cy, size) {
  let d = "";
  for (let i = 0; i < 6; i++) {
    const angle = Math.PI / 180 * (60 * i - 30);
    const x = cx + size * Math.cos(angle);
    const y = cy + size * Math.sin(angle);
    d += (i === 0 ? "M" : "L") + x.toFixed(2) + "," + y.toFixed(2);
  }
  return d + "Z";
}

function render() {
  if (!state) return;
  renderControls();
  const board = buildBoardModel();
  renderBoard(board);
  renderStatus();
  renderMoves();
  renderTacticsPanel(board.tacticMaps);
}

function renderControls() {
  document.body.classList.toggle("tactics-on", tacticsOn);
  document.body.classList.toggle("pending", pendingRequest);
  document.getElementById("tacticsBtn").classList.toggle("active", tacticsOn);
  document.querySelectorAll("#modeSeg button").forEach(button => button.classList.toggle("active", button.dataset.mode === tacticFilters.mode));
  document.querySelectorAll("#playerSeg button").forEach(button => button.classList.toggle("active", button.dataset.player === tacticFilters.player));
  document.querySelectorAll("#axisSeg button").forEach(button => button.classList.toggle("active", button.dataset.axis === tacticFilters.axis));
  document.getElementById("inspectBtn").classList.toggle("active", tacticFilters.inspect);
  document.querySelectorAll("button").forEach(button => { button.disabled = pendingRequest; });
}

function buildBoardModel() {
  const occupied = new Map(state.placements.map(p => [`${p.q},${p.r}`, p]));
  const legal = new Map(state.legal.map(c => [`${c.q},${c.r}`, c]));
  const tacticMaps = buildTacticMaps();
  const cells = new Map([...legal, ...occupied].map(([key, val]) => [key, val]));
  let minX = -HEX, maxX = HEX, minY = -HEX, maxY = HEX;
  const data = [];

  for (const [key, cell] of cells) {
    const c = center(cell.q, cell.r);
    minX = Math.min(minX, c.x - HEX * 1.4);
    maxX = Math.max(maxX, c.x + HEX * 1.4);
    minY = Math.min(minY, c.y - HEX * 1.4);
    maxY = Math.max(maxY, c.y + HEX * 1.4);
    data.push({ key, q: cell.q, r: cell.r, x: c.x, y: c.y, placement: occupied.get(key), legal: legal.has(key) });
  }

  return { data, minX, maxX, minY, maxY, tacticMaps };
}

function renderBoard(board) {
  const pad = 24;
  svg.setAttribute("viewBox", `${board.minX - pad} ${board.minY - pad} ${board.maxX - board.minX + pad * 2} ${board.maxY - board.minY + pad * 2}`);
  board.data.sort((a, b) => (a.placement ? 1 : 0) - (b.placement ? 1 : 0));

  let html = "";
  for (const h of board.data) {
    const isStone = Boolean(h.placement);
    const fill = isStone ? (h.placement.player === "player0" ? "var(--p0)" : "var(--p1)") : "#111722";
    const stroke = isStone ? "#5d6b7c" : "#30363d";
    const opacity = isStone ? "1" : "0.72";
    const roles = board.tacticMaps.cellRoles.get(h.key) || new Set();
    const tacticClasses = tacticsOn ? Array.from(roles).map(role => role + "-cell").join(" ") : "";
    const selectedClass = selectedCellKey === h.key ? "selected-cell" : "";
    const cls = (h.legal && !isStone ? "cell legal" : "cell") + " " + tacticClasses + " " + selectedClass;
    html += `<path class="${cls}" d="${path(h.x, h.y, HEX - 1)}" fill="${fill}" stroke="${stroke}" stroke-width="1" opacity="${opacity}" data-q="${h.q}" data-r="${h.r}"></path>`;
    if (tacticsOn && !isStone) html += renderHeatOverlay(h, board.tacticMaps);
    if (tacticsOn && !isStone) html += renderThreatOverlay(h, board.tacticMaps);
    if (tacticsOn) html += renderCellBadge(h, roles);
    if (isStone) html += `<text class="stone-label" x="${h.x}" y="${h.y}">${h.placement.index}</text>`;
  }
  svg.innerHTML = html;
  bindBoardEvents();
}

function bindBoardEvents() {
  svg.querySelectorAll(".cell").forEach(el => {
    el.addEventListener("click", () => {
      if (pendingRequest) return;
      if (tacticsOn && tacticFilters.inspect) {
        selectedCellKey = `${el.dataset.q},${el.dataset.r}`;
        selectedWindowId = null;
        render();
      } else if (el.classList.contains("legal")) {
        post("/api/move", { q: Number(el.dataset.q), r: Number(el.dataset.r) });
      }
    });
    el.addEventListener("mousemove", showTip);
    el.addEventListener("mouseleave", hideTip);
  });
}

function renderStatus() {
  document.getElementById("playerVal").textContent = state.winner ? playerLabel(state.winner) + " wins" : playerLabel(state.current_player);
  document.getElementById("phaseVal").textContent = state.winner ? "Complete" : phaseLabel(state.phase);
  document.getElementById("stonesVal").textContent = state.placements.length;
  document.getElementById("legalVal").textContent = state.legal_count;
  document.getElementById("statusText").textContent = state.winner ? `${playerLabel(state.winner)} wins by six in line` : `${playerLabel(state.current_player)} to place`;
}

function renderMoves() {
  const moves = document.getElementById("moves");
  moves.innerHTML = state.placements.map(p => {
    const cls = p.player === "player0" ? "p0" : "p1";
    return `<div class="move ${cls}">${String(p.index).padStart(2, "0")} ${p.player === "player0" ? "P0" : "P1"} (${p.q}, ${p.r})</div>`;
  }).join("");
  moves.parentElement.scrollTop = moves.parentElement.scrollHeight;
}

function renderCellBadge(h, roles) {
  if (!roles.size) return "";
  const label = roles.has("win") ? "W" : roles.has("block") ? "!" : "";
  return label ? `<text class="cell-badge" x="${h.x}" y="${h.y + 1}">${label}</text>` : "";
}

function renderHeatOverlay(h, tacticMaps) {
  const heat = tacticMaps.cellHeat.get(h.key);
  if (!heat) return "";
  const shape = path(h.x, h.y, HEX - 3);
  return ["player0", "player1"].map(player => {
    const count = heat[player] || 0;
    if (!count) return "";
    const cls = player === "player1" ? "p1" : "p0";
    const opacity = Math.min(0.72, 0.08 + count * 0.045);
    return `<path class="heat-cell ${cls}" d="${shape}" opacity="${opacity.toFixed(3)}"></path>`;
  }).join("");
}

function renderThreatOverlay(h, tacticMaps) {
  const count = tacticMaps.threatHeat.get(h.key) || 0;
  if (!count) return "";
  const opacity = Math.min(0.7, 0.16 + count * 0.08);
  return `<path class="threat-heat" d="${path(h.x, h.y, HEX - 5)}" opacity="${opacity.toFixed(3)}"></path>`;
}

function renderTacticsPanel(tacticMaps) {
  const panel = document.getElementById("tacticsPanel");
  if (!tacticsOn) {
    panel.innerHTML = "";
    panel.classList.remove("has-selection");
    return;
  }
  const tactics = state.tactics || {};
  const selectedWindow = findWindow(selectedWindowId);
  const selectedCell = selectedCellKey ? cellDebug(selectedCellKey) : null;
  panel.classList.toggle("has-selection", Boolean(selectedWindow || selectedCell));
  const summary = tactics.summary || {};
  const body = selectedWindow ? renderWindowInspector(selectedWindow) : selectedCell ? renderCellInspector(selectedCell) : renderTacticsOverview(tacticMaps);
  panel.innerHTML = `
    <div class="tactics-head">
      <div class="tactics-title">Tactics Overlay</div>
      <div class="metric-row">
        <span><strong>${tactics.window_count || 0}</strong>w</span>
        <span><strong>${summary.threats || 0}</strong>t</span>
        <span><strong>${(tactics.immediate_wins || []).length}</strong>win</span>
        <span><strong>${(tactics.must_blocks || []).length}</strong>block</span>
      </div>
    </div>
    <div class="tactics-body">${body}</div>
  `;
  bindTacticsPanel();
}

function renderTacticsOverview(tacticMaps) {
  const tactics = state.tactics || {};
  return `
    <div class="tactics-section">
      <div class="metric-grid">
        ${metric("Windows", tacticMaps.windows.length)}
        ${metric("Coverage", tacticMaps.coverage)}
        ${metric("P0 Max", tacticMaps.maxHeat.player0)}
        ${metric("P1 Max", tacticMaps.maxHeat.player1)}
      </div>
    </div>
    <div class="tactics-section">
      <div class="tactics-title">Forcing</div>
      <div class="metric-grid">
        ${metric("Wins", (tactics.immediate_wins || []).length)}
        ${metric("Blocks", (tactics.must_blocks || []).length)}
        ${metric("Threats", (tactics.summary || {}).threats || 0)}
        ${metric("Blocked", (tactics.summary || {}).blocked || 0)}
      </div>
    </div>
  `;
}

function renderCellInspector(info) {
  return `
    <div class="tactics-section">
      <div class="fact-main"><span><span class="pill threat">cell</span> (${info.q}, ${info.r})</span><span>${info.legal ? "legal" : info.owner ? playerShort(info.owner) : "empty"}</span></div>
      <div class="fact-sub">${info.owner ? `Stone ${info.index} by ${playerShort(info.owner)}` : info.legal ? "Legal move" : "Not currently playable"}</div>
      ${info.legal ? `<button id="playSelectedBtn" data-q="${info.q}" data-r="${info.r}">Play selected</button>` : ""}
    </div>
    ${renderFactSection("Wins From This Cell", info.wins, "win")}
    ${renderFactSection("Blocks From This Cell", info.blocks, "block")}
    ${renderWindowList(info.windows.slice(0, 18), "Containing Windows")}
  `;
}

function renderWindowInspector(w) {
  const relatedWins = factsForWindow((state.tactics || {}).immediate_wins || [], w.id);
  const relatedBlocks = factsForWindow((state.tactics || {}).must_blocks || [], w.id);
  return `
    <div class="tactics-section">
      <div class="fact-main">
        <span>${playerPill(w.player || w.active_player)} ${escapeText(w.id)}</span>
        <span>${w.own_count || 0}/6</span>
      </div>
      <div class="fact-sub">${escapeText(w.axis)} axis · ${escapeText(w.severity)} · ${w.is_blocked ? "blocked" : w.blockable_now ? "blockable now" : "not blockable now"}</div>
    </div>
    <div class="tactics-section">
      <div class="tactics-title">Cells</div>
      <div class="cell-strip">${(w.cells || []).map(c => renderSlot(c, w)).join("")}</div>
    </div>
    <div class="tactics-section">
      <div class="tactics-title">Masks</div>
      ${maskRow("P0", w.mask && w.mask.player0)}
      ${maskRow("P1", w.mask && w.mask.player1)}
      ${maskRow("Occupied", w.mask && w.mask.occupied)}
      ${maskRow("Empty", w.mask && w.mask.empty)}
    </div>
    <div class="tactics-section">
      <div class="tactics-title">Derived Facts</div>
      <div class="detail-grid">
        ${flag("active", w.is_active)}
        ${flag("blocked", w.is_blocked)}
        ${flag("threat", w.is_threat)}
        ${flag("win", w.is_win)}
        ${flag("blockable", w.blockable_now)}
        ${flag("player", playerShort(w.player || w.active_player))}
      </div>
    </div>
    ${renderFactSection("Related Wins", relatedWins, "win")}
    ${renderFactSection("Related Blocks", relatedBlocks, "block")}
    <div class="tactics-section">
      <div class="tactics-title">Raw Window</div>
      <div class="detail">${escapeText(JSON.stringify(w, null, 2))}</div>
    </div>
  `;
}

function metric(label, value) {
  return `<div class="metric"><strong>${escapeText(value)}</strong>${label}</div>`;
}

function renderFactSection(title, facts, kind) {
  const filtered = facts.filter(f => tacticFilters.player === "both" || f.player === tacticFilters.player);
  return `
    <div class="tactics-section">
      <div class="tactics-title">${title}</div>
      <div class="fact-list">
        ${filtered.length ? filtered.map(f => `<div class="fact" data-cell-key="${f.q},${f.r}">
          <div class="fact-main"><span><span class="pill ${kind}">${kind}</span> ${playerShort(f.player)} (${f.q}, ${f.r})</span><span>${(f.window_ids || []).length}w</span></div>
          <div class="fact-sub">${idList(f.window_ids)}</div>
        </div>`).join("") : `<div class="fact-sub">None</div>`}
      </div>
    </div>
  `;
}

function renderWindowList(windows, title = "Visible Windows") {
  return `
    <div class="tactics-section">
      <div class="tactics-title">${title}</div>
      <div class="fact-list">
        ${windows.length ? windows.map(renderWindowRow).join("") : `<div class="fact-sub">No matching windows</div>`}
      </div>
    </div>
  `;
}

function renderWindowRow(w) {
  const selected = selectedWindowId === w.id ? "selected" : "";
  return `<div class="fact ${selected}" data-window-id="${escapeAttr(w.id)}">
    <div class="fact-main"><span>${playerPill(w.player || w.active_player)} ${escapeText(w.id)}</span><span>${w.own_count || 0}/6</span></div>
    <div class="fact-sub">${escapeText(w.severity)} · empty ${coordList(w.empty_cells)} · playable ${coordList(w.blockable_cells)}</div>
  </div>`;
}

function bindTacticsPanel() {
  document.querySelectorAll("[data-window-id]").forEach(el => {
    el.addEventListener("click", () => {
      selectedWindowId = el.dataset.windowId;
      selectedCellKey = null;
      render();
    });
  });
  document.querySelectorAll("[data-cell-key]").forEach(el => {
    el.addEventListener("click", () => {
      selectedCellKey = el.dataset.cellKey;
      selectedWindowId = null;
      render();
    });
  });
  const play = document.getElementById("playSelectedBtn");
  if (play) play.addEventListener("click", () => post("/api/move", { q: Number(play.dataset.q), r: Number(play.dataset.r) }));
}

function buildTacticMaps() {
  const cellRoles = new Map();
  const cellHeat = new Map();
  const threatHeat = new Map();
  const tactics = state.tactics || {};
  if (!tacticsOn) return { cellRoles, cellHeat, threatHeat, coverage: 0, maxHeat: { player0: 0, player1: 0 }, windows: [] };
  const activeWindows = activeOverlayWindows();
  for (const w of activeWindows) {
    for (const cell of w.empty_cells || []) addHeat(cellHeat, cell, w.active_player || w.player);
  }
  for (const fact of tactics.immediate_wins || []) {
    addRole(cellRoles, fact, "win");
  }
  for (const fact of tactics.must_blocks || []) {
    addRole(cellRoles, fact, "block");
  }
  for (const threat of tactics.threats || []) {
    for (const cell of threat.empty_cells || []) {
      addThreatHeat(threatHeat, cell);
    }
  }
  const windows = visibleWindows();
  for (const w of windows) {
    if (w.id === selectedWindowId) {
      for (const cell of w.cells || []) addRole(cellRoles, cell, "selected");
    }
  }
  return { cellRoles, cellHeat, threatHeat, coverage: cellHeat.size, maxHeat: heatMax(cellHeat), windows };
}

function activeOverlayWindows() {
  const tactics = state.tactics || {};
  return (tactics.windows || []).filter(w => w.is_active && windowMatchesFilters(w));
}

function visibleWindows() {
  if (!tacticsOn) return [];
  const tactics = state.tactics || {};
  const windows = [];
  for (const w of tactics.windows || []) {
    if (!windowMatchesFilters(w)) continue;
    if (tacticFilters.mode === "forcing" && !(w.is_win || Number(w.own_count || 0) >= 5)) continue;
    if (tacticFilters.mode === "threats" && !w.is_threat) continue;
    if (tacticFilters.mode === "windows" && !w.is_active) continue;
    if (tacticFilters.mode === "all" && !(w.is_active || w.is_blocked || w.is_win)) continue;
    windows.push(w);
  }
  const selected = (tactics.windows || []).find(w => w.id === selectedWindowId);
  if (selected && !windows.find(w => w.id === selected.id)) windows.push(selected);
  return [...new Map(windows.map(w => [w.id, w])).values()];
}

function addRole(map, coord, role) {
  const key = `${coord.q},${coord.r}`;
  if (!map.has(key)) map.set(key, new Set());
  map.get(key).add(role);
}

function addHeat(map, coord, player) {
  if (!player) return;
  const key = `${coord.q},${coord.r}`;
  if (!map.has(key)) map.set(key, { player0: 0, player1: 0 });
  map.get(key)[player] += 1;
}

function addThreatHeat(map, coord) {
  const key = `${coord.q},${coord.r}`;
  map.set(key, (map.get(key) || 0) + 1);
}

function heatMax(map) {
  const max = { player0: 0, player1: 0 };
  for (const heat of map.values()) {
    max.player0 = Math.max(max.player0, heat.player0 || 0);
    max.player1 = Math.max(max.player1, heat.player1 || 0);
  }
  return max;
}

function windowMatchesFilters(w) {
  if (tacticFilters.player !== "both" && w.player !== tacticFilters.player && w.active_player !== tacticFilters.player && w.threat_player !== tacticFilters.player) return false;
  if (tacticFilters.axis !== "all" && w.axis !== tacticFilters.axis) return false;
  return true;
}

function findWindow(id) {
  return (state.tactics && (state.tactics.windows || []).find(w => w.id === id)) || null;
}

function cellDebug(key) {
  const [q, r] = key.split(",").map(Number);
  const placements = state.placements || [];
  const owner = placements.find(p => p.q === q && p.r === r);
  const legal = (state.legal || []).some(c => c.q === q && c.r === r);
  const tactics = state.tactics || {};
  return {
    q,
    r,
    legal,
    owner: owner && owner.player,
    index: owner && owner.index,
    wins: (tactics.immediate_wins || []).filter(f => f.q === q && f.r === r),
    blocks: (tactics.must_blocks || []).filter(f => f.q === q && f.r === r),
    windows: (tactics.windows || []).filter(w => (w.cells || []).some(c => c.q === q && c.r === r)).filter(windowMatchesFilters),
  };
}

function factsForWindow(facts, windowId) {
  return facts.filter(f => (f.window_ids || []).includes(windowId));
}

function renderSlot(cell, w) {
  const ownerClass = cell.owner === "player1" ? "p1" : cell.owner === "player0" ? "p0" : "empty";
  const blockable = (w.blockable_cells || []).some(c => c.q === cell.q && c.r === cell.r);
  return `<div class="slot ${ownerClass} ${blockable ? "blockable" : ""}" data-cell-key="${cell.q},${cell.r}">
    <div>${cell.index}</div>
    <div>${cell.owner ? playerShort(cell.owner) : "--"}</div>
    <div>(${cell.q},${cell.r})</div>
  </div>`;
}

function maskRow(label, value) {
  return `<div class="mask-row"><span class="label">${label}</span><span class="bits">${maskBits(value)}</span></div>`;
}

function maskBits(value) {
  const mask = Number(value || 0);
  return Array.from({ length: 6 }, (_, i) => (mask & (1 << i)) ? "1" : "0").join(" ");
}

function flag(label, value) {
  return `<div class="fact-sub"><span class="label">${label}</span> ${escapeText(value)}</div>`;
}

function playerPill(player) {
  const cls = player === "player1" ? "p1" : player === "player0" ? "p0" : "blocked";
  return `<span class="pill ${cls}">${playerShort(player)}</span>`;
}

function coordList(coords) {
  return (coords || []).map(c => `(${c.q},${c.r})`).join(" ") || "-";
}

function idList(ids) {
  return (ids || []).map(escapeText).join(" ");
}

function playerShort(player) {
  if (player === "player0") return "P0";
  if (player === "player1") return "P1";
  return "--";
}

function escapeText(text) {
  return String(text).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function escapeAttr(text) {
  return escapeText(text);
}

function showTip(event) {
  tip.style.display = "block";
  tip.style.left = event.offsetX + 12 + "px";
  tip.style.top = event.offsetY + 12 + "px";
  const key = `${event.target.dataset.q},${event.target.dataset.r}`;
  const info = tacticsOn ? cellDebug(key) : null;
  if (!info) {
    tip.textContent = `${event.target.dataset.q}, ${event.target.dataset.r}`;
    return;
  }
  const parts = [`(${info.q}, ${info.r})`];
  if (info.legal) parts.push("legal");
  if (info.owner) parts.push(playerShort(info.owner));
  if (info.wins.length) parts.push(`${info.wins.length} win`);
  if (info.blocks.length) parts.push(`${info.blocks.length} block`);
  const threats = info.windows.filter(w => w.is_threat).length;
  if (threats) parts.push(`${threats} threat windows`);
  tip.textContent = parts.join(" · ");
}

function hideTip() {
  tip.style.display = "none";
}

loadState();
