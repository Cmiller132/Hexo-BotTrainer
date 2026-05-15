const HEX = 19;
const SQRT3 = Math.sqrt(3);
const FIT_MOVE_COUNT = 8;
const FIT_LEGAL_RADIUS = 5;

let state = null;
let tacticsOn = false;
let selectedWindowId = null;
let selectedCellKey = null;
let tacticFilters = { mode: "windows", player: "both", axis: "all", inspect: false };
let pendingRequest = false;
let requestSeq = 0;
let replayIndex = null;
let replayTimer = null;
let boardBaseView = null;
let boardView = null;
let boardViewDirty = false;
let boardDrag = null;
let suppressBoardClick = false;

const svg = document.getElementById("boardSvg");
const boardArea = document.getElementById("boardArea");
const tip = document.getElementById("tip");
const cellHud = document.getElementById("cellHud");

document.getElementById("newBtn").addEventListener("click", () => {
  clearBoardView();
  resetReplay();
  post("/api/new", {});
});
document.getElementById("fitBtn").addEventListener("click", fitBoard);
document.getElementById("zoomInBtn").addEventListener("click", () => zoomBoardAtCenter(0.82));
document.getElementById("zoomOutBtn").addEventListener("click", () => zoomBoardAtCenter(1.22));
document.getElementById("tacticsBtn").addEventListener("click", () => {
  tacticsOn = !tacticsOn;
  if (!tacticsOn) clearTacticSelection();
  render();
});
document.querySelectorAll("#modeSeg button").forEach(button => {
  button.addEventListener("click", () => { tacticFilters.mode = button.dataset.mode; clearTacticSelection(); render(); });
});
document.querySelectorAll("#playerSeg button").forEach(button => {
  button.addEventListener("click", () => { tacticFilters.player = button.dataset.player; clearTacticSelection(); render(); });
});
document.querySelectorAll("#axisSeg button").forEach(button => {
  button.addEventListener("click", () => { tacticFilters.axis = button.dataset.axis; clearTacticSelection(); render(); });
});
document.getElementById("inspectBtn").addEventListener("click", () => {
  tacticFilters.inspect = !tacticFilters.inspect;
  if (!tacticFilters.inspect) clearTacticSelection();
  render();
});
document.getElementById("replayStartBtn").addEventListener("click", () => setReplayIndex(0));
document.getElementById("replayPrevBtn").addEventListener("click", () => setReplayIndex(viewedPlacementCount() - 1));
document.getElementById("replayPlayBtn").addEventListener("click", toggleReplayPlay);
document.getElementById("replayNextBtn").addEventListener("click", () => setReplayIndex(viewedPlacementCount() + 1));
document.getElementById("replayLiveBtn").addEventListener("click", () => setReplayIndex(totalPlacements()));
document.getElementById("replaySlider").addEventListener("input", event => setReplayIndex(Number(event.target.value)));
window.addEventListener("resize", () => { if (state) render(); });
boardArea.addEventListener("click", handleBoardClick);
bindBoardViewEvents();

async function loadState() {
  const res = await fetch("/api/state");
  state = await res.json();
  clearBoardView();
  resetReplay();
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
      resetReplay();
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
  if (value) stopReplay();
  document.body.classList.toggle("pending", value);
  document.querySelectorAll("button").forEach(button => { button.disabled = value; });
}

function render() {
  if (!state) return;
  renderControls();
  const board = buildBoardModel();
  renderBoard(board);
  renderStatus();
  renderMoves();
  renderTacticsPanel(board.tacticMaps);
  renderReplay();
}

function renderControls() {
  document.body.classList.toggle("tactics-on", tacticsOn);
  document.body.classList.toggle("pending", pendingRequest);
  document.body.classList.toggle("replay-mode", !isLiveView());
  document.getElementById("tacticsBtn").classList.toggle("active", tacticsOn);
  document.querySelectorAll("#modeSeg button").forEach(button => button.classList.toggle("active", button.dataset.mode === tacticFilters.mode));
  document.querySelectorAll("#playerSeg button").forEach(button => button.classList.toggle("active", button.dataset.player === tacticFilters.player));
  document.querySelectorAll("#axisSeg button").forEach(button => button.classList.toggle("active", button.dataset.axis === tacticFilters.axis));
  document.getElementById("inspectBtn").classList.toggle("active", tacticFilters.inspect);
  document.querySelectorAll("button").forEach(button => { button.disabled = pendingRequest; });
  document.querySelectorAll(".replay-buttons button").forEach(button => { button.disabled = pendingRequest || totalPlacements() === 0; });
  document.getElementById("replaySlider").disabled = pendingRequest || totalPlacements() === 0;
}

function buildBoardModel() {
  const shownPlacements = visiblePlacements();
  const occupied = new Map(shownPlacements.map(p => [`${p.q},${p.r}`, p]));
  const liveLegal = new Map((state.legal || []).map(c => [`${c.q},${c.r}`, c]));
  const legal = isLiveView() ? liveLegal : new Map();
  const tacticMaps = buildTacticMaps();
  const cells = new Map();
  for (const [key, cell] of liveLegal) cells.set(key, cell);
  for (const placement of state.placements || []) cells.set(`${placement.q},${placement.r}`, placement);

  let minX = -HEX;
  let maxX = HEX;
  let minY = -HEX;
  let maxY = HEX;
  let focusMinX = Infinity;
  let focusMaxX = -Infinity;
  let focusMinY = Infinity;
  let focusMaxY = -Infinity;
  const data = [];

  for (const [key, cell] of cells) {
    const c = center(cell.q, cell.r);
    minX = Math.min(minX, c.x - HEX * 1.4);
    maxX = Math.max(maxX, c.x + HEX * 1.4);
    minY = Math.min(minY, c.y - HEX * 1.4);
    maxY = Math.max(maxY, c.y + HEX * 1.4);
    if (occupied.has(key)) {
      focusMinX = Math.min(focusMinX, c.x);
      focusMaxX = Math.max(focusMaxX, c.x);
      focusMinY = Math.min(focusMinY, c.y);
      focusMaxY = Math.max(focusMaxY, c.y);
    }
    data.push({ key, q: cell.q, r: cell.r, x: c.x, y: c.y, placement: occupied.get(key), legal: legal.has(key) });
  }

  const hasFocus = Number.isFinite(focusMinX);
  const focusPad = HEX * 7;
  const focus = hasFocus ? {
    minX: Math.max(minX, focusMinX - focusPad),
    maxX: Math.min(maxX, focusMaxX + focusPad),
    minY: Math.max(minY, focusMinY - focusPad),
    maxY: Math.min(maxY, focusMaxY + focusPad),
  } : null;

  const boardBounds = { minX, maxX, minY, maxY };
  const camera = buildCameraBox(shownPlacements, liveLegal, boardBounds);

  return { data, minX, maxX, minY, maxY, focus, camera, tacticMaps };
}

function renderBoard(board) {
  const compact = window.innerWidth < 1200;
  const box = board.camera || (compact && board.focus ? board.focus : board);
  const pad = compact ? 44 : 32;
  syncBoardView(viewForBox(box, pad));
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  board.data.sort((a, b) => (a.placement ? 1 : 0) - (b.placement ? 1 : 0));

  let html = "";
  const drawTactics = tacticsOn && isLiveView();
  for (const h of board.data) {
    const isStone = Boolean(h.placement);
    const fill = isStone ? playerColor(h.placement.player) : "#101924";
    const stroke = isStone ? "#708296" : "#2c3d50";
    const opacity = isStone ? "1" : h.legal ? "0.86" : "0.62";
    const roles = board.tacticMaps.cellRoles.get(h.key) || new Set();
    const tacticClasses = drawTactics ? Array.from(roles).map(role => role + "-cell").join(" ") : "";
    const selectedClass = selectedCellKey === h.key ? "selected-cell" : "";
    const cls = (h.legal && !isStone ? "cell legal" : "cell") + " " + tacticClasses + " " + selectedClass;
    html += `<path class="${cls}" d="${path(h.x, h.y, HEX - 1)}" fill="${fill}" stroke="${stroke}" stroke-width="1" opacity="${opacity}" data-q="${h.q}" data-r="${h.r}"></path>`;
    if (drawTactics && !isStone) html += renderHeatOverlay(h, board.tacticMaps);
    if (drawTactics && !isStone) html += renderThreatOverlay(h, board.tacticMaps);
    if (drawTactics) html += renderCellBadge(h, roles);
    if (isStone) html += `<text class="stone-label" x="${h.x}" y="${h.y}">${h.placement.index}</text>`;
  }
  svg.innerHTML = html;
  bindBoardEvents();
}

function buildCameraBox(shownPlacements, liveLegal, boardBounds) {
  const coords = [];
  const selectedWindow = selectedWindowId ? findWindow(selectedWindowId) : null;
  if (selectedWindow) {
    coords.push(...(selectedWindow.cells || []));
  } else if (selectedCellKey) {
    const selected = cellInfo(selectedCellKey);
    if (Number.isFinite(selected.q) && Number.isFinite(selected.r)) coords.push(selected);
  }

  const recent = shownPlacements.slice(-FIT_MOVE_COUNT);
  coords.push(...recent);

  const anchor = coords.length ? coords[coords.length - 1] : shownPlacements[shownPlacements.length - 1];
  if (anchor) {
    for (const cell of liveLegal.values()) {
      if (axialDistance(anchor, cell) <= FIT_LEGAL_RADIUS) coords.push(cell);
    }
  }

  if (!coords.length) return boardBounds;

  const focused = boxForCoords(coords, HEX * 8);
  const maxSpan = HEX * (window.innerWidth < 700 ? 34 : 48);
  if (focused.maxX - focused.minX <= maxSpan && focused.maxY - focused.minY <= maxSpan) return focused;

  const c = center(anchor.q, anchor.r);
  return {
    minX: c.x - maxSpan / 2,
    maxX: c.x + maxSpan / 2,
    minY: c.y - maxSpan / 2,
    maxY: c.y + maxSpan / 2,
  };
}

function boxForCoords(coords, pad) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const coord of coords) {
    const c = center(coord.q, coord.r);
    minX = Math.min(minX, c.x - pad);
    maxX = Math.max(maxX, c.x + pad);
    minY = Math.min(minY, c.y - pad);
    maxY = Math.max(maxY, c.y + pad);
  }
  return { minX, maxX, minY, maxY };
}

function axialDistance(a, b) {
  const dq = a.q - b.q;
  const dr = a.r - b.r;
  return Math.max(Math.abs(dq), Math.abs(dr), Math.abs(dq + dr));
}

function bindBoardEvents() {
  svg.querySelectorAll(".cell").forEach(el => {
    el.addEventListener("mousemove", showTip);
    el.addEventListener("mouseleave", hideTip);
  });
}

function handleBoardClick(event) {
  if (event.target.closest(".board-view-controls") || event.target.closest(".legend")) return;
  if (suppressBoardClick || pendingRequest || !isLiveView()) return;
  const el = cellElementFromClick(event);
  if (!el) return;
  if (tacticsOn && tacticFilters.inspect) {
    selectedCellKey = `${el.dataset.q},${el.dataset.r}`;
    selectedWindowId = null;
    render();
  } else if (el.classList.contains("legal")) {
    post("/api/move", { q: Number(el.dataset.q), r: Number(el.dataset.r) });
  }
}

function cellElementFromClick(event) {
  let el = event.target.closest(".cell");
  if (!el) {
    const hit = document.elementFromPoint(event.clientX, event.clientY);
    el = hit && hit.closest(".cell");
  }
  return el && svg.contains(el) ? el : null;
}

function bindBoardViewEvents() {
  boardArea.addEventListener("wheel", event => {
    if (!boardView || event.target.closest(".board-view-controls")) return;
    event.preventDefault();
    const factor = event.deltaY < 0 ? 0.88 : 1.14;
    zoomBoard(factor, clientToBoardPoint(event.clientX, event.clientY));
  }, { passive: false });

  boardArea.addEventListener("pointerdown", event => {
    if (!boardView || pendingRequest || (event.pointerType === "mouse" && event.button !== 0)) return;
    if (event.target.closest(".board-view-controls") || event.target.closest(".legend")) return;
    const rect = svg.getBoundingClientRect();
    boardDrag = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      scaleX: boardView.width / Math.max(1, rect.width),
      scaleY: boardView.height / Math.max(1, rect.height),
      view: { ...boardView },
      moved: false,
    };
    boardArea.setPointerCapture(event.pointerId);
    boardArea.classList.add("dragging");
    hideTip();
  });

  boardArea.addEventListener("pointermove", event => {
    if (!boardDrag || event.pointerId !== boardDrag.pointerId) return;
    event.preventDefault();
    const dx = (event.clientX - boardDrag.clientX) * boardDrag.scaleX;
    const dy = (event.clientY - boardDrag.clientY) * boardDrag.scaleY;
    if (Math.hypot(event.clientX - boardDrag.clientX, event.clientY - boardDrag.clientY) > 4) boardDrag.moved = true;
    boardView = {
      ...boardDrag.view,
      x: boardDrag.view.x - dx,
      y: boardDrag.view.y - dy,
    };
    boardViewDirty = true;
    applyBoardView();
  });

  boardArea.addEventListener("pointerup", finishBoardDrag);
  boardArea.addEventListener("pointercancel", finishBoardDrag);
}

function finishBoardDrag(event) {
  if (!boardDrag || event.pointerId !== boardDrag.pointerId) return;
  if (boardDrag.moved) {
    suppressBoardClick = true;
    window.setTimeout(() => { suppressBoardClick = false; }, 80);
  }
  if (boardArea.hasPointerCapture(event.pointerId)) boardArea.releasePointerCapture(event.pointerId);
  boardDrag = null;
  boardArea.classList.remove("dragging");
}

function viewForBox(box, pad) {
  return {
    x: box.minX - pad,
    y: box.minY - pad,
    width: box.maxX - box.minX + pad * 2,
    height: box.maxY - box.minY + pad * 2,
  };
}

function syncBoardView(nextBase) {
  boardBaseView = nextBase;
  if (!boardView || !boardViewDirty) boardView = { ...nextBase };
  applyBoardView();
}

function applyBoardView() {
  if (!boardView) return;
  svg.setAttribute("viewBox", `${boardView.x} ${boardView.y} ${boardView.width} ${boardView.height}`);
}

function fitBoard() {
  boardViewDirty = false;
  if (boardBaseView) boardView = { ...boardBaseView };
  render();
}

function clearBoardView() {
  boardBaseView = null;
  boardView = null;
  boardViewDirty = false;
}

function zoomBoardAtCenter(factor) {
  if (!boardView) return;
  zoomBoard(factor, {
    x: boardView.x + boardView.width / 2,
    y: boardView.y + boardView.height / 2,
  });
}

function zoomBoard(factor, anchor) {
  if (!boardView) return;
  const base = boardBaseView || boardView;
  const nextWidth = clamp(boardView.width * factor, base.width * 0.14, base.width * 4.2);
  const scale = nextWidth / boardView.width;
  const nextHeight = boardView.height * scale;
  const point = anchor || {
    x: boardView.x + boardView.width / 2,
    y: boardView.y + boardView.height / 2,
  };
  boardView = {
    x: point.x - (point.x - boardView.x) * scale,
    y: point.y - (point.y - boardView.y) * scale,
    width: nextWidth,
    height: nextHeight,
  };
  boardViewDirty = true;
  applyBoardView();
}

function clientToBoardPoint(clientX, clientY) {
  const matrix = svg.getScreenCTM();
  if (!matrix || !boardView) {
    return {
      x: boardView ? boardView.x + boardView.width / 2 : 0,
      y: boardView ? boardView.y + boardView.height / 2 : 0,
    };
  }
  const point = svg.createSVGPoint();
  point.x = clientX;
  point.y = clientY;
  return point.matrixTransform(matrix.inverse());
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function renderStatus() {
  const total = totalPlacements();
  const viewed = viewedPlacementCount();
  document.getElementById("playerVal").textContent = state.winner ? playerLabel(state.winner) + " wins" : playerLabel(state.current_player);
  document.getElementById("phaseVal").textContent = state.winner ? "Complete" : phaseLabel(state.phase);
  document.getElementById("stonesVal").textContent = total;
  document.getElementById("legalVal").textContent = state.legal_count;
  document.getElementById("viewVal").textContent = isLiveView() ? "Live" : `${viewed} / ${total}`;
  if (!isLiveView()) {
    document.getElementById("statusText").textContent = `Reviewing move ${viewed} / ${total}`;
  } else {
    document.getElementById("statusText").textContent = state.winner ? `${playerLabel(state.winner)} wins by six in line` : `${playerLabel(state.current_player)} to place`;
  }
}

function renderMoves() {
  renderMoveHistory();
  bindMoveSelectors();
}

function renderMoveHistory() {
  const history = document.getElementById("moveHistory");
  const placements = state.placements || [];
  const selected = viewedPlacementCount();
  if (!placements.length) {
    history.innerHTML = `<div class="empty-list">No moves yet</div>`;
    return;
  }
  history.innerHTML = placements.map(p => {
    const cls = p.player === "player0" ? "p0" : "p1";
    const selectedClass = selected === p.index ? "selected" : "";
    return `<button class="history-chip ${cls} ${selectedClass}" data-move-index="${p.index}">
      <span class="chip-index">${p.index}</span>
      <span class="chip-dot"></span>
      <span class="chip-text">${playerShort(p.player)} (${p.q}, ${p.r})</span>
    </button>`;
  }).join("");
  const selectedChip = history.querySelector(".history-chip.selected");
  if (selectedChip) {
    const centered = selectedChip.offsetLeft - history.clientWidth / 2 + selectedChip.clientWidth / 2;
    history.scrollLeft = Math.max(0, centered);
  }
}

function bindMoveSelectors() {
  document.querySelectorAll("[data-move-index]").forEach(el => {
    el.addEventListener("click", () => setReplayIndex(Number(el.dataset.moveIndex)));
  });
}

function renderReplay() {
  const total = totalPlacements();
  const viewed = viewedPlacementCount();
  const slider = document.getElementById("replaySlider");
  slider.max = String(total);
  slider.value = String(viewed);
  document.getElementById("replayLabel").textContent = `${viewed} / ${total}`;
  document.getElementById("replaySub").textContent = replaySubtitle(viewed);
  document.getElementById("replayMidTick").textContent = String(Math.floor(total / 2));
  document.getElementById("replayMaxTick").textContent = String(total);
  document.getElementById("replayPlayBtn").textContent = replayTimer ? "Pause" : "Play";
}

function renderCellBadge(h, roles) {
  if (!roles.size) return "";
  const label = roles.has("win") ? "W" : roles.has("block") ? "B" : "";
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
    const opacity = Math.min(0.74, 0.08 + count * 0.048);
    return `<path class="heat-cell ${cls}" d="${shape}" opacity="${opacity.toFixed(3)}"></path>`;
  }).join("");
}

function renderThreatOverlay(h, tacticMaps) {
  const count = tacticMaps.threatHeat.get(h.key) || 0;
  if (!count) return "";
  const opacity = Math.min(0.74, 0.18 + count * 0.075);
  return `<path class="threat-heat" d="${path(h.x, h.y, HEX - 5)}" opacity="${opacity.toFixed(3)}"></path>`;
}

function renderTacticsPanel(tacticMaps) {
  const panel = document.getElementById("tacticsPanel");
  const tactics = state.tactics || {};
  const summary = tactics.summary || {};
  const selectedWindow = tacticsOn && isLiveView() ? findWindow(selectedWindowId) : null;
  const selectedCell = tacticsOn && isLiveView() && selectedCellKey ? cellDebug(selectedCellKey) : null;
  panel.classList.toggle("has-selection", Boolean(selectedWindow || selectedCell));

  let body = `<div class="fact-sub">Overlay off</div>`;
  if (tacticsOn && !isLiveView()) {
    body = `<div class="fact-sub">Replay view</div>`;
  } else if (tacticsOn) {
    body = selectedWindow ? renderWindowInspector(selectedWindow) : selectedCell ? renderCellInspector(selectedCell) : renderTacticsOverview(tacticMaps);
  }

  panel.innerHTML = `
    <div class="tactics-head">
      <div class="metric-row">
        <span><strong>${tacticMaps.windows.length}</strong>Windows</span>
        <span><strong>${tacticMaps.coverage}</strong>Coverage</span>
        <span><strong>${(tactics.immediate_wins || []).length}</strong>Wins</span>
        <span><strong>${(tactics.must_blocks || []).length}</strong>Blocks</span>
      </div>
    </div>
    <div class="tactics-body">
      <div class="metric-grid stats-grid">
        ${metric("P0 Max", tacticMaps.maxHeat.player0)}
        ${metric("P1 Max", tacticMaps.maxHeat.player1)}
        ${metric("Threats", summary.threats || 0)}
        ${metric("Blocked", summary.blocked || 0)}
      </div>
      ${body}
    </div>
  `;
  bindTacticsPanel();
}

function renderTacticsOverview(tacticMaps) {
  const tactics = state.tactics || {};
  return `
    <div class="tactics-section">
      <div class="tactics-title">Forcing</div>
      <div class="metric-grid">
        ${metric("Forcing Wins", (tactics.immediate_wins || []).length)}
        ${metric("Must Blocks", (tactics.must_blocks || []).length)}
      </div>
    </div>
    ${renderFactSection("Immediate Wins", tactics.immediate_wins || [], "win")}
    ${renderFactSection("Must Blocks", tactics.must_blocks || [], "block")}
    ${renderWindowList(tacticMaps.windows.slice(0, 40), "Visible Windows")}
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
    ${renderWindowList(info.windows.slice(0, 30), "Containing Windows")}
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
      <div class="fact-sub">${escapeText(w.axis)} axis - ${escapeText(w.severity)} - ${w.is_blocked ? "blocked" : w.blockable_now ? "blockable now" : "not blockable now"}</div>
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
    <div class="fact-sub">${escapeText(w.severity)} - empty ${coordList(w.empty_cells)} - playable ${coordList(w.blockable_cells)}</div>
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
  if (!tacticsOn || !isLiveView()) return emptyTacticMaps(cellRoles, cellHeat, threatHeat);

  const windows = visibleWindows();
  const overlayWindows = windows.filter(w => w.is_active);
  for (const w of overlayWindows) {
    for (const cell of w.empty_cells || []) addHeat(cellHeat, cell, w.active_player || w.player);
    if (w.is_threat) {
      for (const cell of w.empty_cells || []) addThreatHeat(threatHeat, cell);
    }
  }
  for (const fact of (state.tactics || {}).immediate_wins || []) addRole(cellRoles, fact, "win");
  for (const fact of (state.tactics || {}).must_blocks || []) addRole(cellRoles, fact, "block");
  for (const w of windows) {
    if (w.id === selectedWindowId) {
      for (const cell of w.cells || []) addRole(cellRoles, cell, "selected");
    }
  }
  return { cellRoles, cellHeat, threatHeat, coverage: cellHeat.size, maxHeat: heatMax(cellHeat), windows };
}

function emptyTacticMaps(cellRoles = new Map(), cellHeat = new Map(), threatHeat = new Map()) {
  return { cellRoles, cellHeat, threatHeat, coverage: 0, maxHeat: { player0: 0, player1: 0 }, windows: [] };
}

function visibleWindows() {
  if (!tacticsOn || !isLiveView()) return [];
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
  const info = cellInfo(key);
  const tactics = state.tactics || {};
  return {
    ...info,
    wins: (tactics.immediate_wins || []).filter(f => f.q === info.q && f.r === info.r),
    blocks: (tactics.must_blocks || []).filter(f => f.q === info.q && f.r === info.r),
    windows: (tactics.windows || []).filter(w => (w.cells || []).some(c => c.q === info.q && c.r === info.r)).filter(windowMatchesFilters),
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

function playerLabel(player) {
  return player === "player0" ? "Player 0" : "Player 1";
}

function playerShort(player) {
  if (player === "player0") return "P0";
  if (player === "player1") return "P1";
  return "--";
}

function playerColor(player) {
  return player === "player0" ? "var(--p0)" : "var(--p1)";
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

function visiblePlacements() {
  return (state.placements || []).slice(0, viewedPlacementCount());
}

function totalPlacements() {
  return state ? (state.placements || []).length : 0;
}

function viewedPlacementCount() {
  const total = totalPlacements();
  if (replayIndex === null) return total;
  return Math.max(0, Math.min(replayIndex, total));
}

function isLiveView() {
  return replayIndex === null || viewedPlacementCount() === totalPlacements();
}

function setReplayIndex(index) {
  stopReplay();
  const total = totalPlacements();
  replayIndex = Math.max(0, Math.min(index, total));
  if (replayIndex === total) replayIndex = null;
  clearTacticSelection();
  render();
}

function resetReplay() {
  stopReplay();
  replayIndex = null;
  clearTacticSelection();
}

function toggleReplayPlay() {
  const total = totalPlacements();
  if (!total) return;
  if (replayTimer) {
    stopReplay(true);
    return;
  }
  if (viewedPlacementCount() >= total) replayIndex = 0;
  replayTimer = window.setInterval(() => {
    const next = viewedPlacementCount() + 1;
    if (next >= total) {
      replayIndex = null;
      stopReplay();
    } else {
      replayIndex = next;
    }
    clearTacticSelection();
    render();
  }, 520);
  render();
}

function stopReplay(renderAfter = false) {
  if (replayTimer) {
    window.clearInterval(replayTimer);
    replayTimer = null;
    if (renderAfter) render();
  }
}

function replaySubtitle(viewed) {
  if (!viewed) return "Opening";
  const placement = (state.placements || [])[viewed - 1];
  if (!placement) return "Live";
  return `${phaseLabel(placement.phase)} - ${playerShort(placement.player)} (${placement.q}, ${placement.r})`;
}

function clearTacticSelection() {
  selectedWindowId = null;
  selectedCellKey = null;
}

function cellInfo(key) {
  const [q, r] = key.split(",").map(Number);
  const owner = visiblePlacements().find(p => p.q === q && p.r === r);
  const legal = isLiveView() && (state.legal || []).some(c => c.q === q && c.r === r);
  return {
    q,
    r,
    legal,
    owner: owner && owner.player,
    index: owner && owner.index,
  };
}

function showTip(event) {
  if (boardDrag) {
    hideTip();
    return;
  }
  tip.style.display = "block";
  tip.style.left = event.offsetX + 12 + "px";
  tip.style.top = event.offsetY + 12 + "px";
  const key = `${event.target.dataset.q},${event.target.dataset.r}`;
  const info = tacticsOn && isLiveView() ? cellDebug(key) : cellInfo(key);
  updateHud(info);
  const parts = [`(${info.q}, ${info.r})`, cellStateLabel(info)];
  if (info.wins && info.wins.length) parts.push(`${info.wins.length} win`);
  if (info.blocks && info.blocks.length) parts.push(`${info.blocks.length} block`);
  const threats = info.windows ? info.windows.filter(w => w.is_threat).length : 0;
  if (threats) parts.push(`${threats} threat windows`);
  tip.textContent = parts.join(" - ");
}

function hideTip() {
  tip.style.display = "none";
}

function updateHud(info) {
  if (!info) return;
  cellHud.innerHTML = `
    <div><span>Q:</span> <strong>${info.q}</strong> <span>R:</span> <strong>${info.r}</strong></div>
    <div>Cell: ${escapeText(cellStateLabel(info))}</div>
  `;
}

function cellStateLabel(info) {
  if (info.owner) return `${playerShort(info.owner)} stone ${info.index}`;
  if (info.legal) return "legal";
  return "empty";
}

function escapeText(text) {
  return String(text).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function escapeAttr(text) {
  return escapeText(text);
}

loadState();
