const HEX = 19;
const SQRT3 = Math.sqrt(3);
const FIT_MOVE_COUNT = 8;
const FIT_LEGAL_RADIUS = 5;

let state = null;
let tacticsOn = false;
let selectedWindowId = null;
let selectedCellKey = null;
let tacticFilters = { mode: "windows", player: "both", axis: "all", inspect: false };
let tacticsView = "overview";
let pendingRequest = false;
let requestSeq = 0;
let replayIndex = null;
let replayTimer = null;
let boardBaseView = null;
let boardView = null;
let boardViewDirty = false;
let boardDrag = null;
let suppressBoardClick = false;
let adapters = null;
let adapterLoadError = null;
let polling = false;
let pollTimer = null;
let pollAbort = null;
let lastStatusError = "";
let matchConfig = {
  mode: "manual",
  human_player: "player0",
  seed: null,
  bot: { id: "sealbot", variant: "current", time_limit: 0.05 },
};

const svg = document.getElementById("boardSvg");
const boardArea = document.getElementById("boardArea");
const tip = document.getElementById("tip");
const cellHud = document.getElementById("cellHud");

document.getElementById("newBtn").addEventListener("click", () => {
  clearBoardView();
  resetReplay();
  post("/api/new", buildNewMatchPayload(), { resetReplay: true, clearBoard: true });
});
document.getElementById("fitBtn").addEventListener("click", fitBoard);
document.getElementById("zoomInBtn").addEventListener("click", () => zoomBoardAtCenter(0.82));
document.getElementById("zoomOutBtn").addEventListener("click", () => zoomBoardAtCenter(1.22));
document.querySelectorAll("#matchModeSeg button").forEach(button => {
  button.addEventListener("click", () => {
    matchConfig.mode = button.dataset.mode;
    lastStatusError = "";
    render();
  });
});
document.querySelectorAll("#humanSideSeg button").forEach(button => {
  button.addEventListener("click", () => {
    matchConfig.human_player = button.dataset.side;
    render();
  });
});
document.getElementById("botVariantSelect").addEventListener("change", event => {
  matchConfig.bot.variant = event.target.value || "current";
  render();
});
document.getElementById("timeLimitInput").addEventListener("change", event => {
  const value = Number(event.target.value);
  matchConfig.bot.time_limit = Number.isFinite(value) && value > 0 ? value : 0.05;
  event.target.value = String(matchConfig.bot.time_limit);
});
document.getElementById("seedInput").addEventListener("change", event => {
  const value = event.target.value.trim();
  matchConfig.seed = value === "" ? null : Number(value);
});
document.getElementById("tacticsBtn").addEventListener("click", () => {
  tacticsOn = !tacticsOn;
  if (tacticsOn) tacticsView = "overview";
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
  if (tacticFilters.inspect) tacticsView = "cell";
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
  try {
    const res = await fetch("/api/state");
    const data = await safeJson(res);
    if (res.ok) {
      applyState(data, { resetReplay: true, clearBoard: true });
    } else {
      lastStatusError = (data && data.error) || "State unavailable";
      render();
    }
  } finally {
    schedulePoll(250);
  }
}

async function loadAdapters() {
  try {
    const res = await fetch("/api/adapters");
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "Adapter API unavailable");
    adapters = data || {};
    adapterLoadError = null;
    syncDefaultVariant();
  } catch (error) {
    adapters = null;
    adapterLoadError = error && error.message ? error.message : "Adapter API unavailable";
  }
  render();
}

async function post(url, payload, options = {}) {
  if (pendingRequest) return;
  abortPoll();
  const seq = ++requestSeq;
  setPending(true);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await safeJson(res);
    if (seq !== requestSeq) return;
    if (!res.ok) {
      lastStatusError = (data && data.error) || "Request failed";
      if (data && data.state) applyState(data.state, { preserveReplay: true });
      else render();
    } else {
      lastStatusError = "";
      applyState(data, {
        resetReplay: Boolean(options.resetReplay),
        clearBoard: Boolean(options.clearBoard),
        preserveReplay: !options.resetReplay,
      });
    }
  } catch (error) {
    if (seq === requestSeq) {
      lastStatusError = "Request failed";
      render();
    }
  } finally {
    if (seq === requestSeq) {
      setPending(false);
      render();
      schedulePoll(250);
    }
  }
}

function setPending(value) {
  pendingRequest = value;
  if (value) stopReplay();
  document.body.classList.toggle("pending", value);
}

async function safeJson(res) {
  try {
    return await res.json();
  } catch (_) {
    return null;
  }
}

function applyState(next, options = {}) {
  if (!next || typeof next !== "object") return;
  if (!isNewerOrSameState(next)) return;
  const wasLive = !state || isLiveView();
  const currentVersion = Number(state && state.version);
  const nextVersion = Number(next && next.version);
  if (Number.isFinite(currentVersion) && Number.isFinite(nextVersion) && nextVersion > currentVersion && !next.error) {
    lastStatusError = "";
  }
  state = next;
  if (options.clearBoard) clearBoardView();
  if (options.resetReplay) {
    resetReplay();
  } else if (wasLive && !options.preserveReplay) {
    replayIndex = null;
  } else if (replayIndex !== null) {
    replayIndex = Math.min(replayIndex, totalPlacements());
    if (replayIndex === totalPlacements() && wasLive) replayIndex = null;
  }
  render();
}

function isNewerOrSameState(next) {
  const currentVersion = Number(state && state.version);
  const nextVersion = Number(next && next.version);
  if (!Number.isFinite(currentVersion) || !Number.isFinite(nextVersion)) return true;
  return nextVersion >= currentVersion;
}

function schedulePoll(delay = 0) {
  window.clearTimeout(pollTimer);
  pollTimer = window.setTimeout(pollState, delay);
}

function abortPoll() {
  if (pollAbort) {
    pollAbort.abort();
    pollAbort = null;
  }
  polling = false;
}

async function pollState() {
  if (polling || pendingRequest) {
    schedulePoll(600);
    return;
  }
  polling = true;
  const controller = new AbortController();
  pollAbort = controller;
  try {
    const params = new URLSearchParams();
    const version = stateVersion();
    if (version !== null) {
      params.set("since", String(version));
      params.set("timeout_ms", "15000");
    }
    const res = await fetch(`/api/state${params.toString() ? "?" + params.toString() : ""}`, { signal: controller.signal });
    const data = await safeJson(res);
    if (res.ok && data) {
      if (lastStatusError === "Live update paused") lastStatusError = "";
      applyState(data, { preserveReplay: true });
    }
  } catch (error) {
    if (!controller.signal.aborted) {
      lastStatusError = "Live update paused";
      render();
    }
  } finally {
    if (pollAbort === controller) pollAbort = null;
    polling = false;
    schedulePoll(document.hidden ? 2500 : 300);
  }
}

function render() {
  if (!state) {
    renderMatchControls();
    return;
  }
  renderControls();
  const board = buildBoardModel();
  renderBoard(board);
  renderStatus();
  renderMoves();
  renderTacticsPanel(board.tacticMaps);
  renderBotPanel();
  renderTurnOverlay();
  renderReplay();
}

function renderControls() {
  document.body.classList.toggle("tactics-on", tacticsOn);
  document.body.classList.toggle("pending", pendingRequest);
  document.body.classList.toggle("replay-mode", !isLiveView());
  document.body.classList.toggle("bot-thinking", isBotThinking());
  document.body.classList.toggle("state-error", turnStatus() === "error" || Boolean(state.error || lastStatusError));
  renderMatchControls();
  document.getElementById("tacticsBtn").classList.toggle("active", tacticsOn);
  document.querySelectorAll("#modeSeg button").forEach(button => button.classList.toggle("active", button.dataset.mode === tacticFilters.mode));
  document.querySelectorAll("#playerSeg button").forEach(button => button.classList.toggle("active", button.dataset.player === tacticFilters.player));
  document.querySelectorAll("#axisSeg button").forEach(button => button.classList.toggle("active", button.dataset.axis === tacticFilters.axis));
  document.getElementById("inspectBtn").classList.toggle("active", tacticFilters.inspect);
  document.getElementById("fitBtn").disabled = false;
  document.getElementById("tacticsBtn").disabled = false;
  document.querySelectorAll(".overlay-controls button").forEach(button => { button.disabled = pendingRequest; });
  document.querySelectorAll(".replay-buttons button").forEach(button => { button.disabled = totalPlacements() === 0; });
  document.getElementById("replaySlider").disabled = totalPlacements() === 0;
}

function renderMatchControls() {
  const sealbotReady = hasAvailableSealBotVariant();
  const variants = sealbotVariants();
  const mode = matchConfig.mode || "manual";
  const selectedVariant = matchConfig.bot.variant || sealbotDefaultVariant() || "current";
  document.querySelectorAll("#matchModeSeg button").forEach(button => {
    button.classList.toggle("active", button.dataset.mode === mode);
    button.disabled = pendingRequest || (button.dataset.mode === "sealbot" && !sealbotReady);
  });
  document.querySelectorAll("#humanSideSeg button").forEach(button => {
    button.classList.toggle("active", button.dataset.side === matchConfig.human_player);
    button.disabled = pendingRequest || mode !== "sealbot";
  });

  const select = document.getElementById("botVariantSelect");
  const options = variants.length ? variants : [
    { id: "current", label: "current", available: false, error: adapterLoadError || "SealBot unavailable" },
    { id: "best", label: "best", available: false, error: adapterLoadError || "SealBot unavailable" },
  ];
  const optionsKey = JSON.stringify(options.map(variant => [variant.id, variant.label, variant.available]));
  if (select.dataset.optionsKey !== optionsKey) {
    select.innerHTML = options.map(variant => {
      const label = `${variant.label || variant.id}${variant.available ? "" : " unavailable"}`;
      return `<option value="${escapeAttr(variant.id)}" ${variant.available ? "" : "disabled"}>${escapeText(label)}</option>`;
    }).join("");
    select.dataset.optionsKey = optionsKey;
  }
  if (options.some(variant => variant.id === selectedVariant && variant.available !== false)) {
    select.value = selectedVariant;
  } else {
    const first = options.find(variant => variant.available !== false);
    if (first) {
      matchConfig.bot.variant = first.id;
      select.value = first.id;
    }
  }
  select.disabled = pendingRequest || mode !== "sealbot" || !sealbotReady;

  const timeLimit = document.getElementById("timeLimitInput");
  if (document.activeElement !== timeLimit) timeLimit.value = String(matchConfig.bot.time_limit || 0.05);
  timeLimit.disabled = pendingRequest || mode !== "sealbot";
  const seedInput = document.getElementById("seedInput");
  seedInput.disabled = pendingRequest;
  const newBtn = document.getElementById("newBtn");
  newBtn.textContent = state && totalPlacements() ? "Rematch" : "New Match";
  newBtn.disabled = pendingRequest || (mode === "sealbot" && !sealbotReady);
  renderAdapterStatus();
}

function renderAdapterStatus() {
  const el = document.getElementById("adapterStatus");
  const sealbot = sealbotAdapter();
  if (!el) return;
  if (adapterLoadError) {
    el.className = "adapter-status error";
    el.textContent = adapterLoadError;
    return;
  }
  if (!sealbot) {
    el.className = "adapter-status muted";
    el.textContent = "Manual play available. SealBot API not detected.";
    return;
  }
  if (!sealbot.configured && !hasAvailableSealBotVariant()) {
    el.className = "adapter-status error";
    el.textContent = sealbot.error || "SealBot path is not configured.";
    return;
  }
  const available = sealbotVariants().filter(variant => variant.available !== false);
  if (!available.length) {
    const firstError = (sealbotVariants().find(variant => variant.error) || {}).error;
    el.className = "adapter-status error";
    el.textContent = firstError || sealbot.error || "No SealBot variants are available.";
    return;
  }
  el.className = "adapter-status ok";
  el.textContent = `SealBot ready: ${available.map(variant => variant.label || variant.id).join(", ")}`;
}

function buildNewMatchPayload() {
  const seedText = document.getElementById("seedInput").value.trim();
  const seedValue = seedText === "" ? null : Number(seedText);
  const timeValue = Number(document.getElementById("timeLimitInput").value);
  matchConfig.seed = Number.isFinite(seedValue) ? seedValue : null;
  matchConfig.bot.time_limit = Number.isFinite(timeValue) && timeValue > 0 ? timeValue : 0.05;
  const payload = {
    mode: matchConfig.mode,
    human_player: matchConfig.human_player,
    seed: matchConfig.seed,
  };
  if (matchConfig.mode === "sealbot") {
    payload.bot = {
      id: "sealbot",
      variant: matchConfig.bot.variant || sealbotDefaultVariant() || "current",
      time_limit: matchConfig.bot.time_limit,
    };
  }
  return payload;
}

function sealbotAdapter() {
  if (!adapters) return null;
  return adapters.sealbot || adapters.SealBot || null;
}

function sealbotVariants() {
  const sealbot = sealbotAdapter();
  const raw = sealbot && Array.isArray(sealbot.variants) ? sealbot.variants : [];
  return raw.map(variant => ({
    id: String(variant.id || variant.name || variant.label || ""),
    label: String(variant.label || variant.id || variant.name || "SealBot"),
    available: variant.available !== false,
    error: variant.error || "",
  })).filter(variant => variant.id);
}

function hasAvailableSealBotVariant() {
  return sealbotVariants().some(variant => variant.available !== false);
}

function sealbotDefaultVariant() {
  const sealbot = sealbotAdapter();
  return (sealbot && (sealbot.default_variant || sealbot.defaultVariant)) || (sealbotVariants()[0] && sealbotVariants()[0].id) || "current";
}

function syncDefaultVariant() {
  const current = matchConfig.bot.variant;
  const variants = sealbotVariants();
  if (variants.some(variant => variant.id === current && variant.available !== false)) return;
  const preferred = variants.find(variant => variant.id === sealbotDefaultVariant() && variant.available !== false)
    || variants.find(variant => variant.available !== false);
  if (preferred) matchConfig.bot.variant = preferred.id;
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
    tacticsView = "cell";
    render();
  } else if (el.classList.contains("legal")) {
    if (!canSubmitMove()) {
      lastStatusError = isBotThinking() ? "SealBot is thinking" : "Move submission is locked";
      renderStatus();
      renderTurnOverlay();
      return;
    }
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
  const turn = turnStatus();
  const actor = state.winner ? playerLabel(state.winner) + " wins" : playerLabel(state.current_player);
  document.getElementById("matchVal").textContent = matchLabel();
  document.getElementById("playerVal").textContent = state.winner ? playerLabel(state.winner) + " wins" : playerLabel(state.current_player);
  document.getElementById("phaseVal").textContent = state.winner ? "Complete" : phaseLabel(state.phase);
  document.getElementById("stonesVal").textContent = total;
  document.getElementById("legalVal").textContent = state.legal_count ?? (state.legal || []).length;
  document.getElementById("gameVal").textContent = `${state.game_id || "game"} v${state.version ?? "-"}`;
  document.getElementById("viewVal").textContent = isLiveView() ? "Live" : `${viewed} / ${total}`;
  if (lastStatusError) {
    document.getElementById("statusText").textContent = lastStatusError;
  } else if (!isLiveView()) {
    document.getElementById("statusText").textContent = `Reviewing move ${viewed} / ${total}`;
  } else if (turn === "bot_thinking") {
    document.getElementById("statusText").textContent = `${playerLabel(state.thinking_player || state.current_player)} thinking`;
  } else if (turn === "starting") {
    document.getElementById("statusText").textContent = "Starting match";
  } else if (turn === "error" || state.error) {
    document.getElementById("statusText").textContent = state.error || "Match error";
  } else {
    document.getElementById("statusText").textContent = state.winner ? `${actor} by six in line` : `${actor} to place`;
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

  let body = `<div class="fact-sub">Turn on tactics to inspect windows, threats, and blocks.</div>`;
  let tabs = "";
  if (tacticsOn && !isLiveView()) {
    body = `<div class="fact-sub">Replay view</div>`;
  } else if (tacticsOn) {
    tabs = renderTacticsTabs();
    if (tacticsView === "cell") {
      body = selectedCell ? renderCellInspector(selectedCell) : renderCellEmptyState();
    } else if (tacticsView === "windows") {
      body = renderWindowsExplorer(tacticMaps, selectedWindow);
    } else {
      body = renderTacticsOverview(tacticMaps);
    }
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
      ${tabs}
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

function renderBotPanel() {
  const card = document.getElementById("sealbotCard");
  const panel = document.getElementById("botPanel");
  const show = matchConfig.mode === "sealbot" || isSealBotMatch() || Boolean(state.last_bot_decision) || Boolean(state.adapter_errors);
  card.hidden = !show;
  if (!show) {
    panel.innerHTML = "";
    return;
  }

  const decision = normalizeBotDecision(state.last_bot_decision);
  const errors = adapterErrors();
  const thinking = isBotThinking();
  const configuredOnly = matchConfig.mode === "sealbot" && !isSealBotMatch();
  const statusLabel = configuredOnly ? "Ready for next match" : turnStatusLabel();
  const rows = [
    botMetric("Status", thinking ? "Thinking" : statusLabel),
    botMetric("Variant", activeBotVariantLabel()),
    botMetric("Last Move", decision.moveLabel || "-"),
    botMetric("Duration", decision.durationLabel || "-"),
  ];
  if (decision.depth !== null) rows.push(botMetric("Depth", decision.depth));
  if (decision.nodes !== null) rows.push(botMetric("Nodes", decision.nodes));
  if (decision.score !== null) rows.push(botMetric("Score", decision.score));

  panel.innerHTML = `
    <div class="bot-status-line ${thinking ? "thinking" : ""}">
      <span class="bot-status-dot"></span>
      <span>${escapeText(thinking ? `${playerLabel(state.thinking_player || botPlayer())} is searching` : statusLabel)}</span>
    </div>
    <div class="bot-metrics">${rows.join("")}</div>
    ${errors.length ? `<div class="adapter-error-list">${errors.map(error => `<div>${escapeText(error)}</div>`).join("")}</div>` : ""}
    ${decision.raw ? `<details class="raw-details"><summary>Raw Diagnostics</summary><div class="detail">${escapeText(JSON.stringify(decision.raw, null, 2))}</div></details>` : ""}
  `;
}

function renderTurnOverlay() {
  const overlay = document.getElementById("turnOverlay");
  const title = document.getElementById("turnOverlayTitle");
  const sub = document.getElementById("turnOverlaySub");
  const show = isLiveView() && (isBotThinking() || turnStatus() === "starting");
  overlay.hidden = !show;
  if (!show) return;
  title.textContent = isBotThinking() ? "SealBot thinking" : "Starting match";
  sub.textContent = isBotThinking()
    ? `${playerLabel(state.thinking_player || botPlayer())} is choosing the next placement`
    : "Preparing players";
}

function botMetric(label, value) {
  return `<div class="bot-metric"><span>${escapeText(label)}</span><strong>${escapeText(value)}</strong></div>`;
}

function normalizeBotDecision(decision) {
  if (!decision || typeof decision !== "object") {
    return { raw: null, moveLabel: "", durationLabel: "", depth: null, nodes: null, score: null };
  }
  const diagnostics = decision.diagnostics && typeof decision.diagnostics === "object" ? decision.diagnostics : {};
  const move = decision.move || decision.action || decision.placement || decision;
  const q = firstFinite(move.q, decision.q);
  const r = firstFinite(move.r, decision.r);
  const duration = firstFinite(decision.duration_ms, decision.elapsed_ms, diagnostics.duration_ms, diagnostics.elapsed_ms);
  return {
    raw: decision,
    moveLabel: Number.isFinite(q) && Number.isFinite(r) ? `(${q}, ${r})` : "",
    durationLabel: Number.isFinite(duration) ? `${duration.toFixed(duration >= 10 ? 0 : 1)} ms` : "",
    depth: firstPresent(decision.depth, decision.last_depth, diagnostics.depth, diagnostics.last_depth),
    nodes: firstPresent(decision.nodes, decision._nodes, diagnostics.nodes, diagnostics._nodes),
    score: firstPresent(decision.score, decision.last_score, diagnostics.score, diagnostics.last_score),
  };
}

function renderTacticsTabs() {
  const tabs = [
    ["overview", "Overview"],
    ["cell", "Cell"],
    ["windows", "Windows"],
  ];
  return `<div class="tactics-tabs">${tabs.map(([mode, label]) => `
    <button data-tactics-view="${mode}" class="${tacticsView === mode ? "active" : ""}">${label}</button>
  `).join("")}</div>`;
}

function renderTacticsOverview(tacticMaps) {
  const tactics = state.tactics || {};
  return `
    <div class="overview-grid">
      ${windowCountMetric("P0 Windows", tacticMaps.windows.filter(w => (w.active_player || w.player) === "player0").length, "p0")}
      ${windowCountMetric("P1 Windows", tacticMaps.windows.filter(w => (w.active_player || w.player) === "player1").length, "p1")}
      ${windowCountMetric("Q Axis", tacticMaps.windows.filter(w => w.axis === "Q").length)}
      ${windowCountMetric("R Axis", tacticMaps.windows.filter(w => w.axis === "R").length)}
      ${windowCountMetric("QR Axis", tacticMaps.windows.filter(w => w.axis === "QR").length)}
      ${windowCountMetric("Active", tacticMaps.windows.filter(w => w.is_active).length)}
    </div>
    <div class="tactics-section">
      <div class="tactics-title">Forcing</div>
      <div class="metric-grid">
        ${metric("Forcing Wins", (tactics.immediate_wins || []).length)}
        ${metric("Must Blocks", (tactics.must_blocks || []).length)}
      </div>
    </div>
    ${renderFactSection("Immediate Wins", tactics.immediate_wins || [], "win")}
    ${renderFactSection("Must Blocks", tactics.must_blocks || [], "block")}
    <div class="tactics-section">
      <div class="tactics-title">Browse</div>
      <button class="wide-action" data-tactics-view="windows">Open Window Explorer</button>
    </div>
  `;
}

function renderCellEmptyState() {
  return `
    <div class="empty-panel">
      <div class="fact-main">No cell selected</div>
      <div class="fact-sub">Turn on Inspect, then click a board cell to see containing windows and playable tactical facts.</div>
    </div>
  `;
}

function renderCellInspector(info) {
  return `
    <div class="tactics-section">
      <div class="fact-main"><span><span class="pill threat">cell</span> (${info.q}, ${info.r})</span><span>${info.legal ? "legal" : info.owner ? playerShort(info.owner) : "empty"}</span></div>
      <div class="fact-sub">${info.owner ? `Stone ${info.index} by ${playerShort(info.owner)}` : info.legal ? "Legal move" : "Not currently playable"}</div>
      ${info.legal ? `<button id="playSelectedBtn" data-q="${info.q}" data-r="${info.r}" ${canSubmitMove() ? "" : "disabled"}>Play selected</button>` : ""}
    </div>
    ${renderFactSection("Wins From This Cell", info.wins, "win")}
    ${renderFactSection("Blocks From This Cell", info.blocks, "block")}
    ${renderWindowGroups(info.windows, "Containing Windows")}
  `;
}

function renderWindowInspector(w) {
  const relatedWins = factsForWindow((state.tactics || {}).immediate_wins || [], w.id);
  const relatedBlocks = factsForWindow((state.tactics || {}).must_blocks || [], w.id);
  return `
    <div class="selected-window-card">
      <div class="fact-main">
        <span>${playerPill(w.player || w.active_player)} ${escapeText(w.id)}</span>
        <span>${w.own_count || 0}/6</span>
      </div>
      <div class="window-glyph large">${(w.cells || []).map(c => renderWindowSlot(c, w)).join("")}</div>
      <div class="window-tags">
        ${renderWindowTags(w)}
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
    <details class="raw-details">
      <summary>Raw Window</summary>
      <div class="detail">${escapeText(JSON.stringify(w, null, 2))}</div>
    </details>
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

function renderWindowsExplorer(tacticMaps, selectedWindow) {
  return `
    ${selectedWindow ? renderWindowInspector(selectedWindow) : ""}
    ${renderWindowGroups(tacticMaps.windows, "Window Explorer")}
  `;
}

function renderWindowGroups(windows, title = "Windows") {
  const sorted = [...windows].sort(windowPrioritySort);
  const groups = groupedWindows(sorted);
  return `
    <div class="tactics-section">
      <div class="tactics-title">${title}</div>
      ${groups.length ? `<div class="window-groups">${groups.map(renderWindowGroup).join("")}</div>` : `<div class="fact-sub">No matching windows</div>`}
    </div>
  `;
}

function renderWindowGroup(group) {
  return `
    <div class="window-group">
      <div class="window-group-head">
        <span>${playerPill(group.player)} <strong>${escapeText(group.axis)}</strong></span>
        <span>${group.windows.length} windows</span>
      </div>
      <div class="window-card-grid">${group.windows.map(renderWindowCard).join("")}</div>
    </div>
  `;
}

function renderWindowCard(w) {
  const selected = selectedWindowId === w.id ? "selected" : "";
  const emptyCount = (w.empty_cells || []).length;
  const playableCount = (w.blockable_cells || []).length;
  return `<div class="window-card ${selected}" data-window-id="${escapeAttr(w.id)}">
    <div class="window-card-head">
      <span>${playerPill(w.player || w.active_player)} ${escapeText(w.id)}</span>
      <strong>${w.own_count || 0}/6</strong>
    </div>
    <div class="window-glyph">${(w.cells || []).map(c => renderWindowSlot(c, w)).join("")}</div>
    <div class="window-tags">${renderWindowTags(w)}</div>
    <div class="window-meta"><span>${emptyCount} empty</span><span>${playableCount} playable</span></div>
  </div>`;
}

function renderWindowSlot(cell, w) {
  const ownerClass = cell.owner === "player1" ? "p1" : cell.owner === "player0" ? "p0" : "empty";
  const playable = (w.blockable_cells || []).some(c => c.q === cell.q && c.r === cell.r);
  return `<span class="window-slot ${ownerClass} ${playable ? "playable" : ""}" title="(${cell.q}, ${cell.r})" data-cell-key="${cell.q},${cell.r}">
    ${cell.owner ? playerSlotLabel(cell.owner) : ""}
  </span>`;
}

function renderWindowTags(w) {
  return [
    w.is_win ? "win" : "",
    w.is_threat ? "threat" : "",
    w.is_active ? "active" : "",
    w.blockable_now ? "blockable" : "",
    w.is_blocked ? "blocked" : "",
  ].filter(Boolean).map(tag => `<span class="tag ${tag}">${tag}</span>`).join("") || `<span class="tag quiet">${escapeText(w.severity || "window")}</span>`;
}

function groupedWindows(windows) {
  const map = new Map();
  for (const w of windows) {
    const player = w.active_player || w.player || w.threat_player || "blocked";
    const axis = w.axis || "Axis";
    const key = `${player}:${axis}`;
    if (!map.has(key)) map.set(key, { player, axis, windows: [] });
    map.get(key).windows.push(w);
  }
  return [...map.values()].sort((a, b) => playerShort(a.player).localeCompare(playerShort(b.player)) || String(a.axis).localeCompare(String(b.axis)));
}

function windowPrioritySort(a, b) {
  return windowScore(b) - windowScore(a) || String(a.id).localeCompare(String(b.id));
}

function windowScore(w) {
  return (w.is_win ? 1000 : 0)
    + (w.is_threat ? 500 : 0)
    + (w.blockable_now ? 160 : 0)
    + (w.is_active ? 80 : 0)
    + Number(w.own_count || 0) * 20
    - (w.is_blocked ? 50 : 0);
}

function windowCountMetric(label, value, cls = "") {
  return `<div class="mini-metric ${cls}"><strong>${escapeText(value)}</strong><span>${label}</span></div>`;
}

function bindTacticsPanel() {
  document.querySelectorAll("[data-tactics-view]").forEach(el => {
    el.addEventListener("click", event => {
      event.stopPropagation();
      tacticsView = el.dataset.tacticsView;
      render();
    });
  });
  document.querySelectorAll("[data-window-id]").forEach(el => {
    el.addEventListener("click", () => {
      selectedWindowId = el.dataset.windowId;
      selectedCellKey = null;
      tacticsView = "windows";
      render();
    });
  });
  document.querySelectorAll("[data-cell-key]").forEach(el => {
    el.addEventListener("click", event => {
      event.stopPropagation();
      selectedCellKey = el.dataset.cellKey;
      selectedWindowId = null;
      tacticsView = "cell";
      render();
    });
  });
  const play = document.getElementById("playSelectedBtn");
  if (play) play.addEventListener("click", () => {
    if (!canSubmitMove()) return;
    post("/api/move", { q: Number(play.dataset.q), r: Number(play.dataset.r) });
  });
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

function matchLabel() {
  if (isSealBotMatch()) return activeBotVariantLabel();
  return "Manual";
}

function isSealBotMatch() {
  return Boolean(state && state.mode === "sealbot");
}

function canSubmitMove() {
  if (!state || pendingRequest || !isLiveView() || state.winner || turnStatus() === "terminal") return false;
  if (isSealBotMatch()) return state.can_submit === true || turnStatus() === "human_turn";
  return true;
}

function turnStatus() {
  if (!state) return "starting";
  if (state.error) return "error";
  if (state.winner || state.turn_status === "terminal") return "terminal";
  return state.turn_status || (isSealBotMatch() ? "human_turn" : "manual_turn");
}

function turnStatusLabel() {
  const turn = turnStatus();
  if (turn === "bot_thinking") return "Bot thinking";
  if (turn === "human_turn") return "Your turn";
  if (turn === "manual_turn") return "Manual turn";
  if (turn === "terminal") return "Complete";
  if (turn === "error") return "Error";
  if (turn === "starting") return "Starting";
  return turn.replace(/_/g, " ");
}

function isBotThinking() {
  return turnStatus() === "bot_thinking";
}

function botPlayer() {
  if (!isSealBotMatch()) return null;
  if (state && state.players) {
    const bot = state.players.find(player => player.kind === "bot" || player.adapter_id === "sealbot" || player.label === "SealBot");
    if (bot) return bot.role || bot.player || bot.id;
  }
  return (state && state.human_player === "player1") || matchConfig.human_player === "player1" ? "player0" : "player1";
}

function playerMeta(player) {
  return state && Array.isArray(state.players)
    ? state.players.find(item => item.role === player || item.player === player || item.id === player)
    : null;
}

function playerLabel(player) {
  if (!player) return "--";
  if (isSealBotMatch() && player) {
    const human = (state && state.human_player) || matchConfig.human_player;
    if (player === human) return "You";
    if (player === botPlayer()) return "SealBot";
  }
  const meta = playerMeta(player);
  if (meta && meta.label) return meta.label;
  return player === "player0" ? "Player 0" : "Player 1";
}

function playerShort(player) {
  if (isSealBotMatch() && player) {
    const human = (state && state.human_player) || matchConfig.human_player;
    if (player === human) return "You";
    if (player === botPlayer()) return "Bot";
  }
  if (player === "player0") return "P0";
  if (player === "player1") return "P1";
  return "--";
}

function playerSlotLabel(player) {
  const short = playerShort(player);
  if (short === "P0") return "0";
  if (short === "P1") return "1";
  if (short === "You") return "Y";
  if (short === "Bot") return "B";
  return short.slice(0, 1);
}

function activeBotVariantLabel() {
  const configured = state && state.match && state.match.bot && (state.match.bot.variant || state.match.bot.label);
  const variant = configured || matchConfig.bot.variant || sealbotDefaultVariant();
  const known = sealbotVariants().find(item => item.id === variant);
  return (known && known.label) || variant || "current";
}

function adapterErrors() {
  const values = [];
  if (adapterLoadError) values.push(adapterLoadError);
  if (state && state.error) values.push(state.error);
  const raw = state && state.adapter_errors;
  if (Array.isArray(raw)) values.push(...raw.map(String));
  else if (raw && typeof raw === "object") {
    for (const [key, value] of Object.entries(raw)) values.push(`${key}: ${value}`);
  } else if (raw) values.push(String(raw));
  const selected = sealbotVariants().find(variant => variant.id === matchConfig.bot.variant);
  if (selected && selected.available === false && selected.error) values.push(selected.error);
  return [...new Set(values.filter(Boolean))];
}

function firstPresent(...values) {
  for (const value of values) {
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return null;
}

function firstFinite(...values) {
  for (const value of values) {
    const number = Number(value);
    if (Number.isFinite(number)) return number;
  }
  return NaN;
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

function stateVersion() {
  const version = Number(state && state.version);
  return Number.isFinite(version) ? version : null;
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

async function init() {
  await Promise.allSettled([loadAdapters(), loadState()]);
  render();
}

init();
