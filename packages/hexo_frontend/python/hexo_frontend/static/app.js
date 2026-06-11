// ---------------------------------------------------------------------------
// On-screen diagnostics, anchored at the TOP. The Debug tab failed only on the
// owner's real phone (never in headless/desktop), and an earlier BOTTOM-anchored
// banner/version tag was hidden behind the Samsung system nav bar — so errors and
// the version were invisible. This single top bar always shows the running
// version, a live "last tap" echo (so a tap that registers is visible even if its
// effect isn't), and any JS error (uncaught OR surfaced from the Debug code).
const APP_VERSION = "20260611-dbg2";
function __diagBar() {
  let el = document.getElementById("__diag");
  if (!el) {
    el = document.createElement("div");
    el.id = "__diag";
    el.style.cssText =
      "position:fixed;left:0;right:0;top:0;z-index:2147483647;" +
      "font:11px/1.4 ui-monospace,Menlo,Consolas,monospace;padding:4px 8px;" +
      "white-space:pre-wrap;word-break:break-word;max-height:42vh;overflow:auto;";
    el.addEventListener("click", () => { el.dataset.err = ""; __renderDiag(); });
    (document.body || document.documentElement).appendChild(el);
  }
  return el;
}
function __renderDiag() {
  const el = __diagBar();
  const err = el.dataset.err || "";
  const tap = el.dataset.tap || "";
  el.style.background = err ? "#5a1020" : "rgba(8,20,30,0.92)";
  el.style.color = err ? "#fff" : "#7fe0ff";
  el.style.borderBottom = err ? "2px solid #ff5650" : "1px solid #1d3b50";
  el.textContent = "v" + APP_VERSION + (tap ? "  ·  " + tap : "") + (err ? "\nERROR (tap to clear): " + err : "");
}
function reportError(msg) { try { __diagBar().dataset.err = String(msg); __renderDiag(); } catch (_e) {} }
function diagTap(msg) { try { __diagBar().dataset.tap = String(msg); __renderDiag(); } catch (_e) {} }
function showVersionTag() { try { __renderDiag(); } catch (_e) {} }
window.__appVersion = APP_VERSION;
window.addEventListener("error", e => {
  const m = (e && e.error && (e.error.stack || e.error.message)) ||
    `${e && e.message} @ ${e && e.filename}:${e && e.lineno}:${e && e.colno}`;
  reportError(String(m));
});
window.addEventListener("unhandledrejection", e => {
  const r = e && e.reason;
  reportError("promise rejection: " + ((r && (r.stack || r.message)) || String(r)));
});
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", showVersionTag);
} else {
  showVersionTag();
}

const HEX = 19;
const SQRT3 = Math.sqrt(3);
const FIT_MOVE_COUNT = 8;
const FIT_LEGAL_RADIUS = 5;
const HISTORY_ALL_RUNS = "__all__";
const HISTORY_PAGE_SIZE = 50;
const HISTORY_AUTOLOAD_PAGE_SIZE = 500;
const HISTORY_REFRESH_INTERVAL_MS = 15000;
const ARTIFACT_PAGE_SIZE = 50;

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
const activePointers = new Map(); // pointerId -> {x, y} for pan + pinch gestures
let pinchState = null;
let adapters = null;
let adapterLoadError = null;
let trainingRuns = [];
let trainingRun = null;
let trainingRunDetails = {};
let trainingLoadError = "";
const SCREENS = ["match", "history", "debug"];
let activeScreen = screenFromHash();
let historyFilters = { query: "", source: "all", winner: "all" };
let historySort = "newest";
let historySelectedRun = "";
let historySelectionTouched = false;
let historyVisibleLimit = HISTORY_PAGE_SIZE;
let historyDetailsLoading = false;
let historyRefreshInFlight = false;
let historySearchTimer = null;
let historyPage = {
  items: [],
  nextCursor: null,
  complete: true,
  totalMatches: null,
  loading: false,
  loaded: false,
  requestKey: "",
  countLoading: false,
  countRequestKey: "",
};
let selectedHistoryKey = "";
let historyView = false;
let polling = false;
let pollTimer = null;
let pollAbort = null;
let pollFailures = 0;
let lastStatusError = "";
let matchConfig = {
  players: { player0: "manual", player1: "sealbot-current" },
  seed: null,
  time_limit: 0.05,
};

const PLAYER_KIND_LABELS = {
  manual: "Manual",
  "sealbot-current": "SealBot current",
  "sealbot-best": "SealBot best",
  "dense-cnn": "Dense CNN",
  unknown: "Unknown",
};

const svg = document.getElementById("boardSvg");
const boardArea = document.getElementById("boardArea");
const tip = document.getElementById("tip");
const cellHud = document.getElementById("cellHud");
const matchScreen = document.getElementById("matchScreen");
const historyScreen = document.getElementById("historyScreen");
const debugScreen = document.getElementById("debugScreen");
const trainingRunSelect = document.getElementById("trainingRunSelect");
const trainingSummary = document.getElementById("trainingSummary");
const trainingArtifacts = document.getElementById("trainingArtifacts");
const historyRunSelect = document.getElementById("historyRunSelect");
const historyRefreshBtn = document.getElementById("historyRefreshBtn");
const historyOverview = document.getElementById("historyOverview");
const historySearchInput = document.getElementById("historySearchInput");
const historySourceSelect = document.getElementById("historySourceSelect");
const historyWinnerSelect = document.getElementById("historyWinnerSelect");
const historySortSelect = document.getElementById("historySortSelect");
const gameHistoryList = document.getElementById("gameHistoryList");
const gameHistoryDetail = document.getElementById("gameHistoryDetail");
const historyLearningHealth = document.getElementById("historyLearningHealth");
const historyEvalTrend = document.getElementById("historyEvalTrend");
const historyEpochProgress = document.getElementById("historyEpochProgress");

// Null-guarded binding helper: one missing/renamed id should warn and skip,
// not throw and brick the whole script before init() ever runs.
function on(id, evt, fn, opts) {
  const el = document.getElementById(id);
  if (!el) {
    console.warn("missing element #" + id);
    return null;
  }
  el.addEventListener(evt, fn, opts);
  return el;
}

on("newBtn", "click", () => {
  historyView = false;
  clearBoardView();
  resetReplay();
  post("/api/new", buildNewMatchPayload(), { resetReplay: true, clearBoard: true });
});
on("trainingRefreshBtn", "click", () => loadTrainingRuns());
if (historyRefreshBtn) historyRefreshBtn.addEventListener("click", () => loadTrainingRuns({ preserveHistoryPage: true }));
trainingRunSelect.addEventListener("change", () => loadTrainingRun(trainingRunSelect.value));
if (historyRunSelect) historyRunSelect.addEventListener("change", async () => {
  historySelectedRun = historyRunSelect.value || HISTORY_ALL_RUNS;
  historySelectionTouched = true;
  historyVisibleLimit = HISTORY_PAGE_SIZE;
  resetHistoryPage();
  await ensureHistorySelectionLoaded();
  await loadHistoryPage({ reset: true });
  renderGameHistoryPage();
});
document.querySelectorAll("[data-screen]").forEach(button => {
  button.addEventListener("click", () => {
    navigateScreen(button.dataset.screen || "match");
  });
});
trainingArtifacts.addEventListener("click", event => {
  const moreButton = event.target.closest("[data-artifacts-more]");
  if (moreButton) {
    event.preventDefault();
    loadMoreArtifacts();
    return;
  }
  const debugButton = event.target.closest("[data-debug-open]");
  if (debugButton) {
    event.preventDefault();
    debugOpenFromHistory({
      run: debugButton.dataset.debugRun || (trainingRun && trainingRun.name) || "",
      path: debugButton.dataset.debugPath,
      record: Number(debugButton.dataset.debugRecord || 0),
      ply: null,
    });
    return;
  }
  const button = event.target.closest("[data-history-path]");
  if (!button) return;
  event.preventDefault();
  loadTrainingHistory(trainingRun && trainingRun.name, button.dataset.historyPath, Number(button.dataset.recordIndex || 0));
});
if (gameHistoryList) gameHistoryList.addEventListener("click", handleGameHistoryClick);
if (gameHistoryDetail) gameHistoryDetail.addEventListener("click", handleGameHistoryClick);
if (historySearchInput) historySearchInput.addEventListener("input", event => {
  historyFilters.query = event.target.value || "";
  historyVisibleLimit = HISTORY_PAGE_SIZE;
  resetHistoryPage();
  window.clearTimeout(historySearchTimer);
  historySearchTimer = window.setTimeout(() => loadHistoryPage({ reset: true }), 250);
  renderGameHistoryPage();
});
if (historySourceSelect) historySourceSelect.addEventListener("change", event => {
  historyFilters.source = event.target.value || "all";
  historyVisibleLimit = HISTORY_PAGE_SIZE;
  resetHistoryPage();
  loadHistoryPage({ reset: true });
  renderGameHistoryPage();
});
if (historyWinnerSelect) historyWinnerSelect.addEventListener("change", event => {
  historyFilters.winner = event.target.value || "all";
  historyVisibleLimit = HISTORY_PAGE_SIZE;
  resetHistoryPage();
  loadHistoryPage({ reset: true });
  renderGameHistoryPage();
});
if (historySortSelect) historySortSelect.addEventListener("change", event => {
  historySort = event.target.value || "newest";
  historyVisibleLimit = HISTORY_PAGE_SIZE;
  resetHistoryPage();
  loadHistoryPage({ reset: true });
  renderGameHistoryPage();
});
window.addEventListener("hashchange", () => setScreen(screenFromHash(), { preserveHash: true }));
window.setInterval(refreshHistoryIfVisible, HISTORY_REFRESH_INTERVAL_MS);
on("fitBtn", "click", fitBoard);
on("zoomInBtn", "click", () => zoomBoardAtCenter(0.82));
on("zoomOutBtn", "click", () => zoomBoardAtCenter(1.22));
document.querySelectorAll("[data-player-select]").forEach(select => {
  select.addEventListener("change", event => {
    matchConfig.players[event.target.dataset.playerSelect] = event.target.value || "manual";
    lastStatusError = "";
    render();
  });
});
on("timeLimitInput", "change", event => {
  const value = Number(event.target.value);
  matchConfig.time_limit = Number.isFinite(value) && value > 0 ? value : 0.05;
  event.target.value = String(matchConfig.time_limit);
});
on("seedInput", "change", event => {
  const value = event.target.value.trim();
  matchConfig.seed = value === "" ? null : Number(value);
});
on("tacticsBtn", "click", () => {
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
on("inspectBtn", "click", () => {
  tacticFilters.inspect = !tacticFilters.inspect;
  if (!tacticFilters.inspect) clearTacticSelection();
  if (tacticFilters.inspect) tacticsView = "cell";
  render();
});
on("replayStartBtn", "click", () => setReplayIndex(0));
on("replayPrevBtn", "click", () => setReplayIndex(viewedPlacementCount() - 1));
on("replayPlayBtn", "click", toggleReplayPlay);
on("replayNextBtn", "click", () => setReplayIndex(viewedPlacementCount() + 1));
on("replayLiveBtn", "click", () => setReplayIndex(totalPlacements()));
on("replaySlider", "input", event => setReplayIndex(Number(event.target.value)));
window.addEventListener("resize", () => { if (state) render(); });
boardArea.addEventListener("click", handleBoardClick);
bindBoardViewEvents();

async function loadState() {
  historyView = false;
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
    console.warn("loadAdapters: adapter API request failed", error);
    adapters = null;
    adapterLoadError = error && error.message ? error.message : "Adapter API unavailable";
  }
  render();
}

async function loadTrainingRuns(options = {}) {
  const preserveHistoryPage = Boolean(options.preserveHistoryPage);
  const previousHistoryPageKey = activeScreen === "history" ? currentHistoryPageKey() : "";
  const preferred = (trainingRun && trainingRun.name) || trainingRunSelect.value || (historyRunSelect && historyRunSelect.value) || "";
  try {
    const res = await fetch("/api/training/runs");
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "Training runs unavailable");
    trainingRuns = (data && data.runs) || [];
    trainingLoadError = "";
    const selected = trainingRuns.some(run => run.name === preferred) ? preferred : ((trainingRuns[0] && trainingRuns[0].name) || "");
    if (!historySelectionTouched && selected) historySelectedRun = selected;
    if (historySelectedRun !== HISTORY_ALL_RUNS && !trainingRuns.some(run => run.name === historySelectedRun)) {
      historySelectedRun = selected || HISTORY_ALL_RUNS;
    }
    syncTrainingRunSelect(selected);
    syncHistoryRunSelect(historySelectedRun);
    if (trainingRuns.length) {
      await loadTrainingRun(selected, { preserveHistorySelection: true });
      if (activeScreen === "history") {
        const canPreserveHistoryPage = preserveHistoryPage && currentHistoryPageKey() === previousHistoryPageKey;
        if (!canPreserveHistoryPage) resetHistoryPage();
        await ensureHistorySelectionLoaded();
        await loadHistoryPage({ reset: true, preserve: canPreserveHistoryPage });
      }
    }
    else {
      trainingRun = null;
      renderTraining();
    }
  } catch (error) {
    trainingLoadError = error && error.message ? error.message : "Training runs unavailable";
    trainingRuns = [];
    trainingRun = null;
    renderTraining();
  }
}

async function fetchTrainingRunDetail(name) {
  const res = await fetch(`/api/training/run?name=${encodeURIComponent(name)}`);
  const data = await safeJson(res);
  if (!res.ok) throw new Error((data && data.error) || "Training run unavailable");
  trainingRunDetails[name] = data;
  return data;
}

async function loadTrainingRun(name, options = {}) {
  if (!name) {
    trainingRun = null;
    syncTrainingRunSelect("");
    renderTraining();
    return;
  }
  try {
    trainingRun = await fetchTrainingRunDetail(name);
    trainingLoadError = "";
    syncTrainingRunSelect(trainingRun.name || name);
    if (!options.preserveHistorySelection && historySelectedRun !== HISTORY_ALL_RUNS) {
      historySelectedRun = trainingRun.name || name;
      syncHistoryRunSelect(historySelectedRun);
    }
  } catch (error) {
    trainingRun = null;
    trainingLoadError = error && error.message ? error.message : "Training run unavailable";
  }
  renderTraining();
}

async function ensureHistorySelectionLoaded() {
  if (!trainingRuns.length) return;
  const names = historySelectedRun === HISTORY_ALL_RUNS
    ? trainingRuns.map(run => run.name)
    : [historySelectedRun];
  const missing = names.filter(name => name && !trainingRunDetails[name]);
  if (!missing.length) return;
  historyDetailsLoading = true;
  renderGameHistoryPage();
  try {
    await Promise.all(missing.map(name => fetchTrainingRunDetail(name)));
    trainingLoadError = "";
  } catch (error) {
    trainingLoadError = error && error.message ? error.message : "Training run unavailable";
  } finally {
    historyDetailsLoading = false;
    renderGameHistoryPage();
  }
}

function resetHistoryPage() {
  historyPage = {
    items: [],
    nextCursor: null,
    complete: true,
    totalMatches: null,
    loading: false,
    loaded: false,
    requestKey: "",
    countLoading: false,
    countRequestKey: "",
  };
  selectedHistoryKey = "";
}

function currentHistoryPageKey() {
  return JSON.stringify({
    run: historySelectedRun || HISTORY_ALL_RUNS,
    source: historyFilters.source || "all",
    winner: historyFilters.winner || "all",
    sort: historySort || "newest",
    query: historyFilters.query || "",
  });
}

function currentHistoryTargets() {
  const runs = historyRunsForPage();
  const liveStatus = latestRunStatusForHistoryPage();
  const liveEpoch = asFinite(liveStatus && liveStatus.current_epoch);
  const selfplayEpochsSeen = runs
    .flatMap(run => (run.epoch_history || []).map(item => asFinite(item.epoch)))
    .filter(value => value !== null);
  const latestSelfplayEpoch = selfplayEpochsSeen.length ? Math.max(...selfplayEpochsSeen) : null;
  const currentEpoch = liveEpoch !== null ? liveEpoch : latestSelfplayEpoch;
  const previousEpoch = currentEpoch !== null ? currentEpoch - 1 : null;
  const selfplayEpochs = new Set([currentEpoch, previousEpoch].filter(value => value !== null && value >= 0));
  const evalEpochsSeen = runs
    .flatMap(run => (run.evaluation_history || []).map(item => asFinite(item.epoch)))
    .filter(value => value !== null && (currentEpoch === null || value <= currentEpoch));
  const latestEvalEpoch = evalEpochsSeen.length ? Math.max(...evalEpochsSeen) : null;
  const evaluationEpochs = new Set([latestEvalEpoch].filter(value => value !== null));
  const allTargets = [...selfplayEpochs, ...evaluationEpochs];
  return {
    currentEpoch,
    previousEpoch,
    selfplayEpochs,
    evaluationEpochs,
    minEpoch: allTargets.length ? Math.min(...allTargets) : null,
  };
}

function shouldAutoloadHistoryWindow() {
  return historySort === "newest" &&
    (historyFilters.source || "all") === "all" &&
    (historyFilters.winner || "all") === "all" &&
    !(historyFilters.query || "").trim();
}

function historyItemInTargetWindow(item, targets) {
  const epoch = asFinite(item && item.epoch);
  if (epoch === null) return false;
  const source = String(item && item.source || "history");
  if (source === "selfplay") return targets.selfplayEpochs.has(epoch);
  if (source === "evaluation") return targets.evaluationEpochs.has(epoch);
  return false;
}

function historyWindowBoundaryReached(items, targets) {
  if (targets.minEpoch === null) return true;
  return items.some(item => {
    const epoch = asFinite(item && item.epoch);
    return epoch !== null && epoch < targets.minEpoch;
  });
}

async function enterHistoryScreen() {
  if (!trainingRuns.length) return;
  await ensureHistorySelectionLoaded();
  if (!historyPage.loaded && !historyPage.loading) {
    await loadHistoryPage({ reset: true });
  }
}

async function loadHistoryPage(options = {}) {
  if (!trainingRuns.length || historyPage.loading) return;
  const reset = Boolean(options.reset);
  const append = Boolean(options.append);
  const preserve = Boolean(options.preserve);
  const autoloadWindow = reset && !append && shouldAutoloadHistoryWindow();
  const targets = autoloadWindow ? currentHistoryTargets() : null;
  if (reset && !preserve) {
    historyPage.items = [];
    historyPage.nextCursor = null;
    historyPage.complete = true;
    historyPage.totalMatches = null;
    historyPage.loaded = false;
  }
  if (append && !historyPage.nextCursor) return;

  const requestKey = currentHistoryPageKey();
  historyPage.loading = true;
  historyPage.requestKey = requestKey;
  renderGameHistoryPage();
  try {
    const fetchedItems = [];
    let data = null;
    let cursor = append ? historyPage.nextCursor : "";
    let pageCount = 0;
    do {
      const params = new URLSearchParams({
        run: historySelectedRun || HISTORY_ALL_RUNS,
        limit: String(autoloadWindow ? HISTORY_AUTOLOAD_PAGE_SIZE : HISTORY_PAGE_SIZE),
        source: historyFilters.source || "all",
        winner: historyFilters.winner || "all",
        sort: historySort || "newest",
        query: historyFilters.query || "",
        include_total: "0",
      });
      if (cursor) params.set("cursor", cursor);
      const res = await fetch(`/api/training/history-page?${params.toString()}`);
      data = await safeJson(res);
      if (!res.ok) throw new Error((data && data.error) || "Game histories unavailable");
      fetchedItems.push(...((data && data.items) || []));
      cursor = (data && data.next_cursor) || "";
      pageCount += 1;
    } while (
      autoloadWindow &&
      cursor &&
      pageCount < 8 &&
      !historyWindowBoundaryReached(fetchedItems, targets)
    );
    if (historyPage.requestKey !== requestKey) return;
    const items = autoloadWindow && targets && targets.minEpoch !== null
      ? fetchedItems.filter(item => historyItemInTargetWindow(item, targets))
      : fetchedItems;
    historyPage.items = append ? [...historyPage.items, ...items] : items;
    historyPage.nextCursor = (data && data.next_cursor) || null;
    historyPage.complete = Boolean(data && data.complete);
    if (data && data.total_matches !== null && data.total_matches !== undefined) {
      historyPage.totalMatches = data.total_matches;
    } else if (autoloadWindow) {
      historyPage.totalMatches = items.length;
    } else if (!append && !preserve) {
      historyPage.totalMatches = null;
    }
    historyPage.loaded = true;
    trainingLoadError = "";
    if (!append && !autoloadWindow) loadHistoryCount(requestKey);
  } catch (error) {
    trainingLoadError = error && error.message ? error.message : "Game histories unavailable";
  } finally {
    if (historyPage.requestKey === requestKey) {
      historyPage.loading = false;
      renderGameHistoryPage();
    }
  }
}

async function loadHistoryCount(expectedKey = "") {
  if (!trainingRuns.length) return;
  const requestKey = expectedKey || currentHistoryPageKey();
  if (historyPage.countLoading && historyPage.countRequestKey === requestKey) return;
  historyPage.countLoading = true;
  historyPage.countRequestKey = requestKey;
  renderGameHistoryPage();
  try {
    const params = new URLSearchParams({
      run: historySelectedRun || HISTORY_ALL_RUNS,
      source: historyFilters.source || "all",
      winner: historyFilters.winner || "all",
      query: historyFilters.query || "",
    });
    const res = await fetch(`/api/training/history-count?${params.toString()}`);
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "Game count unavailable");
    if (historyPage.countRequestKey !== requestKey || currentHistoryPageKey() !== requestKey) return;
    historyPage.totalMatches = data && data.total_matches !== undefined ? data.total_matches : null;
    trainingLoadError = "";
  } catch (error) {
    console.warn("loadHistoryCount: history count request failed", error);
  } finally {
    if (historyPage.countRequestKey === requestKey) {
      historyPage.countLoading = false;
      renderGameHistoryPage();
    }
  }
}

async function loadMoreArtifacts() {
  if (!trainingRun || !trainingRun.artifacts_page || !trainingRun.artifacts_page.next_cursor) return;
  try {
    const params = new URLSearchParams({
      run: trainingRun.name,
      limit: String(ARTIFACT_PAGE_SIZE),
      cursor: trainingRun.artifacts_page.next_cursor,
    });
    const res = await fetch(`/api/training/artifacts-page?${params.toString()}`);
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "Artifacts unavailable");
    trainingRun.artifacts = [...(trainingRun.artifacts || []), ...((data && data.items) || [])];
    trainingRun.artifacts_page = {
      ...(trainingRun.artifacts_page || {}),
      next_cursor: (data && data.next_cursor) || null,
      complete: Boolean(data && data.complete),
    };
    trainingLoadError = "";
  } catch (error) {
    trainingLoadError = error && error.message ? error.message : "Artifacts unavailable";
  }
  renderTraining();
}

async function refreshHistoryIfVisible() {
  if (activeScreen !== "history" || historyRefreshInFlight || pendingRequest) return;
  historyRefreshInFlight = true;
  try {
    await loadTrainingRuns({ preserveHistoryPage: true });
  } finally {
    historyRefreshInFlight = false;
  }
}

async function loadTrainingHistory(runName, artifactPath, recordIndex = 0) {
  if (!runName || !artifactPath) return;
  abortPoll();
  stopReplay();
  setPending(true);
  try {
    const params = new URLSearchParams({ run: runName, path: artifactPath, record: String(recordIndex || 0) });
    const res = await fetch(`/api/training/history?${params.toString()}`);
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "Game history unavailable");
    historyView = true;
    selectedHistoryKey = historyItemKey({ run: runName, path: artifactPath, record_index: recordIndex || 0 });
    lastStatusError = `Loaded ${artifactPath}`;
    applyState(data, { resetReplay: true, clearBoard: true });
    navigateScreen("match");
  } catch (error) {
    lastStatusError = error && error.message ? error.message : "Game history unavailable";
    render();
  } finally {
    setPending(false);
    renderTraining();
  }
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
    console.error("post: request to " + url + " failed", error);
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
  } catch (error) {
    console.warn("safeJson: failed to parse response body", error);
    return null;
  }
}

function screenFromHash() {
  const hash = String(window.location.hash || "").replace(/^#\/?/, "");
  // `#debug?run=...` deep links carry nav state after `?` — any `#debug...`
  // prefix still routes to the debug screen (the debug module owns the query).
  const name = hash.split("?")[0];
  return SCREENS.includes(name) ? name : "match";
}

function navigateScreen(screen) {
  setScreen(screen);
  const hash = `#${activeScreen}`;
  if (window.location.hash !== hash) window.location.hash = hash;
}

function setScreen(screen, options = {}) {
  const previousScreen = activeScreen;
  activeScreen = SCREENS.includes(screen) ? screen : "match";
  // Match-screen lifecycle: stop the long-poll and any replay timer when we
  // leave it, and resume polling when we (re)enter it. schedulePoll/pollState
  // also gate on activeScreen === "match", so this can never run while History
  // or Debug is up.
  if (previousScreen === "match" && activeScreen !== "match") {
    abortPoll();
    stopReplay();
    window.clearTimeout(pollTimer);
  } else if (activeScreen === "match" && previousScreen !== "match") {
    schedulePoll(0);
  }
  if (matchScreen) matchScreen.hidden = activeScreen !== "match";
  if (historyScreen) historyScreen.hidden = activeScreen !== "history";
  if (debugScreen) debugScreen.hidden = activeScreen !== "debug";
  document.querySelectorAll("[data-screen]").forEach(button => {
    button.classList.toggle("active", button.dataset.screen === activeScreen);
  });
  document.body.classList.toggle("history-screen-active", activeScreen === "history");
  document.body.classList.toggle("debug-screen-active", activeScreen === "debug");
  if (!options.preserveHash) {
    const hash = `#${activeScreen}`;
    if (window.location.hash && window.location.hash !== hash) window.history.replaceState(null, "", hash);
  }
  renderGameHistoryPage();
  if (activeScreen === "history") enterHistoryScreen();
  if (activeScreen === "debug") enterDebugScreen();
  // Re-render the match view once it is actually visible so layout-dependent
  // work (move-history centering, board fit) runs with real element widths
  // instead of the zero width it would see while the screen was hidden.
  if (activeScreen === "match" && state) render();
}

function syncTrainingRunSelect(selected = "") {
  const options = trainingRuns.length
    ? trainingRuns.map(run => `<option value="${escapeAttr(run.name)}">${escapeText(run.name)}</option>`).join("")
    : `<option value="">No runs</option>`;
  if (!trainingRunSelect) return;
  trainingRunSelect.innerHTML = options;
  trainingRunSelect.value = selected;
}

function syncHistoryRunSelect(selected = historySelectedRun) {
  if (!historyRunSelect) return;
  const runOptions = trainingRuns
    .map(run => `<option value="${escapeAttr(run.name)}">${escapeText(run.name)}</option>`)
    .join("");
  historyRunSelect.innerHTML = trainingRuns.length
    ? `<option value="${HISTORY_ALL_RUNS}">All runs</option>${runOptions}`
    : `<option value="">No runs</option>`;
  const value = selected === HISTORY_ALL_RUNS || trainingRuns.some(run => run.name === selected)
    ? selected
    : HISTORY_ALL_RUNS;
  historySelectedRun = value || HISTORY_ALL_RUNS;
  historyRunSelect.value = historySelectedRun;
}

function applyState(next, options = {}) {
  if (!next || typeof next !== "object") return;
  if (isSameGame(next) && !isNewerOrSameState(next)) return;
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

function isSameGame(next) {
  if (!state || !next) return true;
  if (!state.game_id || !next.game_id) return true;
  return state.game_id === next.game_id;
}

function isNewerOrSameState(next) {
  const currentVersion = Number(state && state.version);
  const nextVersion = Number(next && next.version);
  if (!Number.isFinite(currentVersion) || !Number.isFinite(nextVersion)) return true;
  return nextVersion >= currentVersion;
}

function schedulePoll(delay = 0) {
  // The match long-poll only runs while the match screen is active and we are
  // not pinned to a static history view.
  if (historyView || activeScreen !== "match") return;
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
  if (historyView || activeScreen !== "match") return;
  if (polling || pendingRequest) {
    schedulePoll(600);
    return;
  }
  polling = true;
  const controller = new AbortController();
  pollAbort = controller;
  let failed = false;
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
      pollFailures = 0;
      if (lastStatusError === "Live update paused") lastStatusError = "";
      applyState(data, { preserveReplay: true });
    }
  } catch (error) {
    if (!controller.signal.aborted) {
      failed = true;
      console.warn("pollState: live state poll failed", error);
      lastStatusError = "Live update paused";
      render();
    }
  } finally {
    if (pollAbort === controller) pollAbort = null;
    polling = false;
    // On consecutive failures, back off exponentially (capped) instead of
    // hammering the server every 300ms; a successful response resets the streak.
    if (failed) {
      pollFailures += 1;
      schedulePoll(Math.min(300 * Math.pow(2, pollFailures), 5000));
    } else {
      schedulePoll(document.hidden ? 2500 : 300);
    }
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
  renderMoveHistory();
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
  document.querySelectorAll("[data-player-select]").forEach(select => {
    const role = select.dataset.playerSelect;
    const selected = matchConfig.players[role] || select.value || "manual";
    if (select.value !== selected) select.value = selected;
    select.disabled = pendingRequest;
    for (const option of select.options) {
      option.disabled = pendingRequest || !playerKindAvailable(option.value);
    }
  });
  const timeLimit = document.getElementById("timeLimitInput");
  if (document.activeElement !== timeLimit) timeLimit.value = String(matchConfig.time_limit || 0.05);
  timeLimit.disabled = pendingRequest || !setupHasSealBot();
  const seedInput = document.getElementById("seedInput");
  seedInput.disabled = pendingRequest;
  const newBtn = document.getElementById("newBtn");
  newBtn.textContent = state && totalPlacements() ? "Rematch" : "New Match";
  newBtn.disabled = pendingRequest || !selectedSetupAvailable();
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
  matchConfig.time_limit = Number.isFinite(timeValue) && timeValue > 0 ? timeValue : 0.05;
  matchConfig.players = {
    player0: document.getElementById("player0Kind")?.value || "manual",
    player1: document.getElementById("player1Kind")?.value || "sealbot-current",
  };
  return {
    players: { ...matchConfig.players },
    time_limit: matchConfig.time_limit,
    seed: matchConfig.seed,
  };
}

function setupHasSealBot() {
  return Object.values(matchConfig.players || {}).some(kind => String(kind).startsWith("sealbot-"));
}

function selectedSetupAvailable() {
  return Object.values(matchConfig.players || {}).every(playerKindAvailable);
}

function playerKindAvailable(kind) {
  if (!String(kind).startsWith("sealbot-")) return true;
  const variant = String(kind).replace("sealbot-", "");
  return sealbotVariants().some(item => item.id === variant && item.available !== false);
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
  const variants = sealbotVariants();
  const preferred = variants.find(variant => variant.id === sealbotDefaultVariant() && variant.available !== false)
    || variants.find(variant => variant.available !== false);
  if (!preferred) return;
  for (const [role, kind] of Object.entries(matchConfig.players)) {
    if (String(kind).startsWith("sealbot-") && !playerKindAvailable(kind)) {
      matchConfig.players[role] = `sealbot-${preferred.id}`;
    }
  }
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
    const recentRank = recentPlacementRank(h.placement);
    const recentClass = recentRank === 1 ? "last" : recentRank === 2 ? "previous" : "";
    const cls = (h.legal && !isStone ? "cell legal" : "cell")
      + " " + tacticClasses
      + " " + selectedClass
      + (recentRank ? ` recent-stone recent-${recentRank}` : "");
    html += `<path class="${cls}" d="${path(h.x, h.y, HEX - 1)}" fill="${fill}" stroke="${stroke}" stroke-width="1" opacity="${opacity}" data-q="${h.q}" data-r="${h.r}"></path>`;
    if (isStone && recentRank) {
      html += `<path class="last-move-outline ${recentClass}" d="${path(h.x, h.y, HEX - 0.5)}"></path>`;
    }
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
    activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    boardArea.setPointerCapture(event.pointerId);
    boardArea.classList.add("dragging");
    hideTip();
    if (activePointers.size >= 2) {
      beginPinch(); // a second finger upgrades the gesture to pinch-zoom
    } else {
      beginPan(event);
    }
  });

  boardArea.addEventListener("pointermove", event => {
    if (!activePointers.has(event.pointerId)) return;
    event.preventDefault();
    activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pinchState && activePointers.size >= 2) {
      updatePinch();
    } else if (boardDrag && event.pointerId === boardDrag.pointerId) {
      updatePan(event);
    }
  }, { passive: false });

  boardArea.addEventListener("pointerup", endBoardPointer);
  boardArea.addEventListener("pointercancel", endBoardPointer);
}

function boardDragScales() {
  const rect = svg.getBoundingClientRect();
  return {
    scaleX: boardView.width / Math.max(1, rect.width),
    scaleY: boardView.height / Math.max(1, rect.height),
  };
}

function beginPan(event) {
  const { scaleX, scaleY } = boardDragScales();
  boardDrag = {
    pointerId: event.pointerId,
    clientX: event.clientX,
    clientY: event.clientY,
    scaleX,
    scaleY,
    view: { ...boardView },
    moved: false,
  };
}

function updatePan(event) {
  const dx = (event.clientX - boardDrag.clientX) * boardDrag.scaleX;
  const dy = (event.clientY - boardDrag.clientY) * boardDrag.scaleY;
  if (Math.hypot(event.clientX - boardDrag.clientX, event.clientY - boardDrag.clientY) > 4) boardDrag.moved = true;
  boardView = { ...boardDrag.view, x: boardDrag.view.x - dx, y: boardDrag.view.y - dy };
  boardViewDirty = true;
  applyBoardView();
}

function pinchPointers() {
  return [...activePointers.values()].slice(0, 2);
}

function beginPinch() {
  boardDrag = null; // pan and pinch are mutually exclusive
  const [a, b] = pinchPointers();
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  pinchState = {
    rect: svg.getBoundingClientRect(),
    startDist: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)),
    startView: { ...boardView },
    anchorBoard: clientToBoardPoint(mid.x, mid.y),
  };
  suppressBoardClick = true;
}

// Two-finger pinch: zoom by the change in finger distance, keeping the board
// point that was under the initial midpoint pinned beneath the moving midpoint
// (so the gesture also pans).
function updatePinch() {
  const [a, b] = pinchPointers();
  if (!a || !b) return;
  const dist = Math.max(1, Math.hypot(a.x - b.x, a.y - b.y));
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  const base = boardBaseView || pinchState.startView;
  const nextWidth = clamp(pinchState.startView.width * (pinchState.startDist / dist), base.width * 0.14, base.width * 4.2);
  const scale = nextWidth / pinchState.startView.width;
  const nextHeight = pinchState.startView.height * scale;
  const rect = pinchState.rect;
  const viewScaleX = nextWidth / Math.max(1, rect.width);
  const viewScaleY = nextHeight / Math.max(1, rect.height);
  boardView = {
    width: nextWidth,
    height: nextHeight,
    x: pinchState.anchorBoard.x - (mid.x - rect.left) * viewScaleX,
    y: pinchState.anchorBoard.y - (mid.y - rect.top) * viewScaleY,
  };
  boardViewDirty = true;
  applyBoardView();
}

function endBoardPointer(event) {
  if (!activePointers.has(event.pointerId)) return;
  activePointers.delete(event.pointerId);
  if (boardArea.hasPointerCapture(event.pointerId)) boardArea.releasePointerCapture(event.pointerId);

  const moved = (boardDrag && boardDrag.moved) || Boolean(pinchState);
  if (activePointers.size < 2) pinchState = null;

  if (activePointers.size === 1) {
    // Dropped from pinch to a single finger — resume panning from it.
    const [pointerId, point] = [...activePointers.entries()][0];
    beginPan({ pointerId, clientX: point.x, clientY: point.y });
    boardDrag.moved = true;
  } else if (activePointers.size === 0) {
    boardDrag = null;
    boardArea.classList.remove("dragging");
    if (moved) {
      suppressBoardClick = true;
      window.setTimeout(() => { suppressBoardClick = false; }, 80);
    }
  }
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
  const live = isLiveView();
  const active = state.winner ? null : state.current_player;
  const last = lastVisiblePlacement();
  document.body.classList.toggle("player0-turn", active === "player0");
  document.body.classList.toggle("player1-turn", active === "player1");

  document.getElementById("matchVal").textContent = matchLabel();
  document.getElementById("playerVal").textContent = state.winner ? playerLabel(state.winner) + " wins" : playerLabel(state.current_player);
  document.getElementById("phaseVal").textContent = state.winner ? "Complete" : phaseLabel(state.phase);
  document.getElementById("stonesVal").textContent = total;
  document.getElementById("legalVal").textContent = state.legal_count ?? (state.legal || []).length;
  document.getElementById("gameVal").textContent = `${state.game_id || "game"} v${state.version ?? "-"}`;
  document.getElementById("viewVal").textContent = live ? "Live" : `${viewed} / ${total}`;
  setText("turnVal", active ? `${playerShort(active)} - ${playerKindLabel(active)}` : "Complete");
  setText("turnPlacementVal", state.winner ? "Complete" : placementStepLabel());
  setText("lastMoveVal", last ? `#${last.index} ${playerShort(last.player)} (${last.q}, ${last.r})` : "None");
  renderTurnBanner(active);

  if (lastStatusError) {
    document.getElementById("statusText").textContent = lastStatusError;
  } else if (!live) {
    document.getElementById("statusText").textContent = `Reviewing move ${viewed} / ${total}`;
  } else if (state.winner) {
    document.getElementById("statusText").textContent = `${playerLabel(state.winner)} wins by six in line`;
  } else if (state.mode === "history") {
    const history = state.history || {};
    const status = history.status ? ` (${history.status})` : "";
    document.getElementById("statusText").textContent = `Viewing ${state.game_id || "game history"}${status}`;
  } else if (turnStatus() === "bot_thinking") {
    document.getElementById("statusText").textContent = `${playerShort(active)} ${playerKindLabel(active)} thinking`;
  } else if (turnStatus() === "starting") {
    document.getElementById("statusText").textContent = "Starting match";
  } else if (turnStatus() === "error" || state.error) {
    document.getElementById("statusText").textContent = state.error || "Match error";
  } else {
    document.getElementById("statusText").textContent =
      `${playerShort(active)} ${playerKindLabel(active)} to place - ${placementStepLabel()}`;
  }
}

function renderTurnBanner(active) {
  const banner = document.getElementById("turnBanner");
  const title = document.getElementById("turnTitle");
  const sub = document.getElementById("turnSub");
  if (!banner || !title || !sub) return;

  banner.classList.toggle("p0", active === "player0");
  banner.classList.toggle("p1", active === "player1");

  if (state.winner) {
    title.textContent = `${playerLabel(state.winner)} wins`;
    sub.textContent = "Game complete";
    return;
  }

  title.textContent = `${playerShort(active)} to play`;
  sub.textContent = `${playerKindLabel(active)} - ${placementStepLabel()}`;
}

// The move list is a bounded, wrapping, vertically-scrolling box. Chips flow and
// wrap inside it, so the element is always exactly its container's width and can
// never push the page sideways — regardless of move count (a 1000-move game just
// scrolls vertically). Chips are only rebuilt when the move list grows, so live
// polling of a long game doesn't churn ~1000 DOM nodes every tick.
let moveHistoryBound = false;
let moveHistoryStructSig = "";

function renderMoveHistory() {
  const history = document.getElementById("moveHistory");
  if (!history) return;
  ensureMoveHistoryEvents(history);
  const placements = state.placements || [];
  const selected = viewedPlacementCount();

  if (!placements.length) {
    moveHistoryStructSig = "";
    history.classList.remove("has-moves");
    history.innerHTML = `<div class="empty-list">No moves yet</div>`;
    return;
  }

  history.classList.add("has-moves");
  const structSig = `${placements.length}:${placements[placements.length - 1].index}`;
  if (structSig !== moveHistoryStructSig) {
    moveHistoryStructSig = structSig;
    history.innerHTML = placements.map(p => {
      const cls = p.player === "player0" ? "p0" : "p1";
      return `<button class="history-chip ${cls}" data-move-index="${p.index}">
      <span class="chip-index">${p.index}</span>
      <span class="chip-dot"></span>
      <span class="chip-text">${playerShort(p.player)} (${p.q}, ${p.r})</span>
    </button>`;
    }).join("");
  }

  // Update the selection without a full rebuild, then scroll it into view.
  const previous = history.querySelector(".history-chip.selected");
  if (previous) previous.classList.remove("selected");
  const current = history.querySelector(`.history-chip[data-move-index="${selected}"]`);
  if (current) {
    current.classList.add("selected");
    if (history.clientHeight) {
      const top = current.offsetTop - history.clientHeight / 2 + current.clientHeight / 2;
      history.scrollTop = Math.max(0, top);
    }
  }
}

function ensureMoveHistoryEvents(history) {
  if (moveHistoryBound) return;
  moveHistoryBound = true;
  // Delegated click so chip rebuilds never leave stale per-chip listeners.
  history.addEventListener("click", event => {
    const chip = event.target.closest("[data-move-index]");
    if (!chip) return;
    setReplayIndex(Number(chip.dataset.moveIndex));
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
  const show = setupHasSealBot() || isSealBotMatch() || Boolean(state.last_bot_decision) || Boolean(state.adapter_errors);
  card.hidden = !show;
  if (!show) {
    panel.innerHTML = "";
    return;
  }

  const decision = normalizeBotDecision(state.last_bot_decision);
  const errors = adapterErrors();
  const thinking = isBotThinking();
  const configuredOnly = setupHasSealBot() && !isSealBotMatch();
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
  title.textContent = isBotThinking() ? `${playerShort(state.thinking_player || botPlayer())} thinking` : "Starting match";
  sub.textContent = isBotThinking()
    ? `${playerKindLabel(state.thinking_player || botPlayer())} is choosing the next placement`
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

function idList(ids) {
  return (ids || []).map(escapeText).join(" ");
}

function matchLabel() {
  return `P0 ${playerKindLabel("player0")} - P1 ${playerKindLabel("player1")}`;
}

function isSealBotMatch() {
  return Boolean(state && (state.mode === "sealbot" || ["player0", "player1"].some(player => playerKind(player).startsWith("sealbot-"))));
}

function canSubmitMove() {
  if (!state || pendingRequest || !isLiveView() || state.winner || turnStatus() === "terminal") return false;
  if (typeof state.can_submit === "boolean") return state.can_submit;
  return playerKind(state.current_player) === "manual";
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
  if (turn === "human_turn") return "Manual turn";
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
  if (state && state.thinking_player) return state.thinking_player;
  return ["player0", "player1"].find(player => playerKind(player).startsWith("sealbot-")) || null;
}

function playerMeta(player) {
  if (!state || !state.players) return null;
  if (Array.isArray(state.players)) {
    return state.players.find(item => item.role === player || item.player === player || item.id === player) || null;
  }
  const item = state.players[player];
  if (typeof item === "string") return { role: player, kind: item, label: PLAYER_KIND_LABELS[item] || item };
  if (item && typeof item === "object") return item;
  return null;
}

function playerKind(player) {
  const meta = playerMeta(player);
  if (meta) {
    if (typeof meta.kind === "string") return normalizePlayerKind(meta.kind, meta.variant);
    if (typeof meta.variant === "string") return `sealbot-${meta.variant}`;
  }
  const selectId = player === "player0" ? "player0Kind" : "player1Kind";
  return document.getElementById(selectId)?.value || "manual";
}

function normalizePlayerKind(kind, variant = "") {
  if (kind === "human") return "manual";
  if (kind === "bot" || kind === "sealbot") return `sealbot-${variant || "current"}`;
  return kind || "manual";
}

function playerKindLabel(player) {
  const kind = playerKind(player);
  return PLAYER_KIND_LABELS[kind] || kind;
}

function playerLabel(player) {
  if (!player) return "--";
  const slot = player === "player0" ? "P0" : player === "player1" ? "P1" : player;
  return `${slot} ${playerKindLabel(player)}`;
}

function playerShort(player) {
  if (player === "player0") return "P0";
  if (player === "player1") return "P1";
  return "--";
}

function playerSlotLabel(player) {
  const short = playerShort(player);
  if (short === "P0") return "0";
  if (short === "P1") return "1";
  return short.slice(0, 1);
}

function activeBotVariantLabel() {
  const bot = botPlayer();
  const kind = bot ? playerKind(bot) : "sealbot-current";
  const variant = kind.startsWith("sealbot-") ? kind.replace("sealbot-", "") : sealbotDefaultVariant();
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
  for (const kind of Object.values(matchConfig.players || {})) {
    if (!String(kind).startsWith("sealbot-")) continue;
    const selected = sealbotVariants().find(variant => variant.id === String(kind).replace("sealbot-", ""));
    if (selected && selected.available === false && selected.error) values.push(selected.error);
  }
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

function lastVisiblePlacement(offset = 0) {
  const placements = visiblePlacements();
  return placements[placements.length - 1 - offset] || null;
}

function recentPlacementRank(placement) {
  if (!placement) return 0;
  if (samePlacement(placement, lastVisiblePlacement(0))) return 1;
  if (samePlacement(placement, lastVisiblePlacement(1))) return 2;
  return 0;
}

function samePlacement(a, b) {
  return Boolean(a && b && a.index === b.index);
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

// Human-friendly stringification for raw values interpolated into the UI:
// null/undefined/"" become an em dash rather than the literal "null"/"undefined",
// finite numbers pass through, and objects/arrays are JSON-encoded compactly.
function displayValue(value, empty = "—") {
  if (value === null || value === undefined || value === "") return empty;
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : empty;
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch (error) {
      return empty;
    }
  }
  return String(value);
}

function placementStepLabel() {
  if (!state || state.phase === "opening") return "Opening";
  return state.phase === "second_stone" ? "Placement 2 of 2" : "Placement 1 of 2";
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function renderTraining() {
  if (!trainingSummary || !trainingArtifacts) return;
  if (trainingLoadError) {
    trainingSummary.textContent = trainingLoadError;
    trainingArtifacts.innerHTML = "";
    renderGameHistoryPage();
    return;
  }
  if (!trainingRun) {
    trainingSummary.textContent = "No training run selected";
    trainingArtifacts.innerHTML = "";
    renderGameHistoryPage();
    return;
  }
  const artifacts = trainingRun.artifacts || [];
  const histories = trainingRun.histories || [];
  const latest = artifacts.find(item => item.name.includes("performance_calibration")) ||
    artifacts.find(item => item.name.startsWith("epoch_"));
  const epochs = historyEpochs(histories);
  const p0Wins = histories.filter(item => item.winner === "player0").length;
  const p1Wins = histories.filter(item => item.winner === "player1").length;
  const status = trainingRun.status || {};
  const history = status.history || {};
  const watchdog = status.watchdog || {};
  const calibration = status.calibration || {};
  const statusMetrics = [
    summaryMetric("Stage", runStageLabel(status)),
    summaryMetric("Recent Games", firstPresent(history.games, histories.length)),
    summaryMetric("Epochs", epochs.length ? `${epochs[0]}-${epochs[epochs.length - 1]}` : "--"),
    summaryMetric("P0 / P1", `${firstPresent(history.p0_wins, p0Wins)} / ${firstPresent(history.p1_wins, p1Wins)}`),
    summaryMetric("Avg Len", formatDecimal(firstPresent(history.avg_length, averageHistoryLength(histories)), 1)),
    summaryMetric("Selfplay", formatRate(calibration.selfplay_pos_s, "pos/s")),
    summaryMetric("RAM Free", formatGib(watchdog.free_ram_gb)),
    summaryMetric("GPU Free", formatGib(watchdog.gpu_free_gb)),
  ];
  const fallbackMetrics = latest && latest.summary
    ? Object.entries(latest.summary).slice(0, 4).map(([key, value]) => summaryMetric(key, value))
    : [summaryMetric("Artifacts", artifacts.length)];
  trainingSummary.innerHTML = [...statusMetrics, ...fallbackMetrics].join("");
  const shown = artifacts.slice(0, artifacts.length);
  const moreArtifacts = trainingRun.artifacts_page && trainingRun.artifacts_page.next_cursor
    ? `<button class="history-list-more" type="button" data-artifacts-more>Show more artifacts</button>`
    : "";
  trainingArtifacts.innerHTML = `${shown.map(item => trainingArtifactRow(trainingRun.name, item)).join("")}${moreArtifacts}`;
  renderGameHistoryPage();
}

function trainingArtifactRow(runName, item) {
  const href = `/api/training/file?run=${encodeURIComponent(runName)}&path=${encodeURIComponent(item.path)}`;
  const summary = item.summary
    ? Object.entries(item.summary).map(([key, value]) => `${key}: ${displayValue(value)}`).join(" | ")
    : `${item.kind || "file"} | ${formatBytes(item.bytes)}`;
  const preview = item.kind === "png" ? `<img src="${href}" alt="">` : "";
  const loadButton = item.loadable_history
    ? `<button class="artifact-load-btn" type="button" data-history-path="${escapeAttr(item.path)}" data-record-index="0">Load game</button>`
      + `<button class="artifact-debug-btn" type="button" data-debug-open data-debug-run="${escapeAttr(runName || "")}" data-debug-path="${escapeAttr(item.path)}" data-debug-record="0">Debug</button>`
    : "";
  return `<div class="artifact-row">
    <a class="artifact-link" href="${href}" target="_blank" rel="noreferrer" title="${escapeAttr(item.path || item.name || "")}">
      ${preview}
      <span>${escapeText(item.name)}</span>
      <small>${escapeText(summary)}</small>
    </a>
    ${loadButton}
  </div>`;
}

function renderGameHistoryPage() {
  if (!historyOverview || !gameHistoryList || !gameHistoryDetail) return;
  syncHistoryRunSelect(historySelectedRun);
  if (trainingLoadError) {
    historyOverview.innerHTML = "";
    if (historyLearningHealth) historyLearningHealth.innerHTML = "";
    if (historyEvalTrend) historyEvalTrend.innerHTML = "";
    if (historyEpochProgress) historyEpochProgress.innerHTML = "";
    gameHistoryList.innerHTML = `<div class="empty-list">${escapeText(trainingLoadError)}</div>`;
    gameHistoryDetail.innerHTML = `<div class="empty-list">No game selected</div>`;
    return;
  }
  const runs = historyRunsForPage();
  const histories = historyItemsForPage(runs);
  if (!runs.length) {
    historyOverview.innerHTML = "";
    if (historyLearningHealth) historyLearningHealth.innerHTML = "";
    if (historyEvalTrend) historyEvalTrend.innerHTML = "";
    if (historyEpochProgress) historyEpochProgress.innerHTML = "";
    const pendingSelection = historyDetailsLoading || historySelectionPendingDetails();
    gameHistoryList.innerHTML = `<div class="empty-list">${pendingSelection ? "Loading game histories" : "No training run selected"}</div>`;
    gameHistoryDetail.innerHTML = `<div class="empty-list">No game selected</div>`;
    return;
  }

  const usingServerPage = historyPage.loaded || historyPage.loading || historyPage.items.length > 0;
  const filtered = usingServerPage ? histories : sortedHistoryItems(filteredHistoryItems(histories));
  const selected = selectedHistoryItem(histories, filtered);
  const visible = usingServerPage ? filtered : filtered.slice(0, historyVisibleLimit);
  historyOverview.innerHTML = renderHistoryOverview(histories, filtered);
  if (historyLearningHealth) historyLearningHealth.innerHTML = renderLearningHealth(runs);
  if (historyEvalTrend) historyEvalTrend.innerHTML = renderEvaluationTrend(runs);
  if (historyEpochProgress) historyEpochProgress.innerHTML = renderEpochProgress(runs);
  gameHistoryList.innerHTML = filtered.length
    ? [
      ...visible.map(item => gameHistoryListRow(item.run, item)),
      usingServerPage && historyPage.nextCursor
        ? `<button class="history-list-more" type="button" data-history-more>${historyPage.loading ? "Loading games" : `Load more games (${visible.length} loaded)`}</button>`
        : !usingServerPage && filtered.length > visible.length
        ? `<button class="history-list-more" type="button" data-history-more>Show ${Math.min(HISTORY_PAGE_SIZE, filtered.length - visible.length)} more games (${visible.length} of ${filtered.length})</button>`
        : "",
    ].join("")
    : `<div class="empty-list">${historyPage.loading ? "Loading game histories" : "No games match the current filters"}</div>`;
  gameHistoryDetail.innerHTML = selected
    ? gameHistoryDetailHtml(selected.run, selected)
    : `<div class="empty-list">No game selected</div>`;
}

function summaryMetric(key, value) {
  return `<div><span>${escapeText(key)}</span><strong>${escapeText(displayValue(value))}</strong></div>`;
}

function historyRunsForPage() {
  if (historySelectedRun === HISTORY_ALL_RUNS) {
    return trainingRuns
      .map(run => trainingRunDetails[run.name])
      .filter(Boolean);
  }
  const selected = trainingRunDetails[historySelectedRun] ||
    (trainingRun && trainingRun.name === historySelectedRun ? trainingRun : null);
  return selected ? [selected] : [];
}

function historySelectionPendingDetails() {
  if (!trainingRuns.length) return false;
  if (historySelectedRun === HISTORY_ALL_RUNS) {
    return trainingRuns.some(run => run.name && !trainingRunDetails[run.name]);
  }
  return trainingRuns.some(run => run.name === historySelectedRun) && !trainingRunDetails[historySelectedRun];
}

function historyItemsForPage(runs) {
  if (historyPage.loaded || historyPage.loading || historyPage.items.length > 0) {
    return historyPage.items || [];
  }
  return runs.flatMap(run => (run.histories || []).map(item => ({ ...item, run: run.name })));
}

function handleGameHistoryClick(event) {
  const moreButton = event.target.closest("[data-history-more]");
  if (moreButton) {
    event.preventDefault();
    if (historyPage.nextCursor) {
      loadHistoryPage({ append: true });
      return;
    }
    historyVisibleLimit += HISTORY_PAGE_SIZE;
    renderGameHistoryPage();
    return;
  }
  const debugButton = event.target.closest("[data-debug-open]");
  if (debugButton) {
    event.preventDefault();
    debugOpenFromHistory({
      run: debugButton.dataset.debugRun || (trainingRun && trainingRun.name) || "",
      path: debugButton.dataset.debugPath,
      record: Number(debugButton.dataset.debugRecord || 0),
      ply: null,
    });
    return;
  }
  const loadButton = event.target.closest("[data-history-load]");
  if (loadButton) {
    event.preventDefault();
    const runName = loadButton.dataset.historyRun || (trainingRun && trainingRun.name);
    selectedHistoryKey = historyItemKey({
      run: runName,
      path: loadButton.dataset.historyPath,
      record_index: Number(loadButton.dataset.recordIndex || 0),
    });
    renderGameHistoryPage();
    loadTrainingHistory(runName, loadButton.dataset.historyPath, Number(loadButton.dataset.recordIndex || 0));
    return;
  }
  const row = event.target.closest("[data-history-key]");
  if (!row) return;
  event.preventDefault();
  selectedHistoryKey = row.dataset.historyKey || "";
  renderGameHistoryPage();
}

function filteredHistoryItems(histories) {
  const query = historyFilters.query.trim().toLowerCase();
  return histories.filter(item => {
    if (historyFilters.source !== "all" && String(item.source || "history") !== historyFilters.source) return false;
    if (historyFilters.winner === "none" && item.winner) return false;
    if (historyFilters.winner !== "all" && historyFilters.winner !== "none" && item.winner !== historyFilters.winner) return false;
    if (!query) return true;
    const haystack = [
      item.game_id,
      item.run,
      item.path,
      item.status,
      item.source,
      item.epoch,
      item.seed,
      item.winner_label,
      item.length,
      historyPlayerLabel(item.players && item.players.player0),
      historyPlayerLabel(item.players && item.players.player1),
      historyDiagnosticsText(item.diagnostics),
    ].filter(value => value !== undefined && value !== null).join(" ").toLowerCase();
    return haystack.includes(query);
  });
}

function sortedHistoryItems(histories) {
  const items = [...histories];
  const newest = (a, b) => compareHistoryNewest(a, b);
  if (historySort === "longest") {
    return items.sort((a, b) => compareNumber(historyLength(b), historyLength(a)) || newest(a, b));
  }
  if (historySort === "shortest") {
    return items.sort((a, b) => compareNumber(historyLength(a), historyLength(b)) || newest(a, b));
  }
  if (historySort === "oldest") {
    return items.sort((a, b) => -newest(a, b));
  }
  if (historySort === "winner") {
    return items.sort((a, b) => String(a.winner_label || winnerLabel(a.winner)).localeCompare(String(b.winner_label || winnerLabel(b.winner))) || newest(a, b));
  }
  return items.sort(newest);
}

function compareHistoryNewest(a, b) {
  return compareNumber(Number(b.modified || 0), Number(a.modified || 0)) ||
    compareNumber(Number(b.epoch || 0), Number(a.epoch || 0)) ||
    compareNumber(Number(b.record_index || 0), Number(a.record_index || 0));
}

function compareNumber(a, b) {
  const left = Number.isFinite(Number(a)) ? Number(a) : 0;
  const right = Number.isFinite(Number(b)) ? Number(b) : 0;
  return left === right ? 0 : left > right ? 1 : -1;
}

function historyLength(item) {
  return Number(item && (item.length || item.actions || 0)) || 0;
}

function selectedHistoryItem(histories, filtered) {
  const candidates = filtered.length ? filtered : histories;
  let selected = candidates.find(item => historyItemKey(item) === selectedHistoryKey) || null;
  if (!selected && candidates.length) {
    selected = candidates[0];
    selectedHistoryKey = historyItemKey(selected);
  }
  if (!selected) selectedHistoryKey = "";
  return selected;
}

function historyItemKey(item) {
  return `${item && item.run ? item.run : ""}::${item && item.path ? item.path : ""}::${Number(item && item.record_index || 0)}`;
}

function historyEpochs(histories) {
  return [...new Set((histories || [])
    .map(item => Number(item.epoch))
    .filter(epoch => Number.isFinite(epoch)))].sort((a, b) => a - b);
}

function latestRunStatusForHistoryPage() {
  const runs = historyRunsForPage();
  return runs
    .map(run => run && run.status)
    .filter(Boolean)
    .sort((a, b) => Number(b.history && b.history.latest_modified || 0) - Number(a.history && a.history.latest_modified || 0))[0] || null;
}

function humanizeStageId(stage) {
  const raw = String(stage || "").trim();
  if (!raw) return "Unknown";
  const lower = raw.toLowerCase();
  const epochMatch = lower.match(/^epoch[_-]?0*(\d+)/);
  if (epochMatch) return `Epoch ${Number(epochMatch[1])}`;
  if (lower.includes("write_diagnostics") || lower.includes("diagnostic")) return "Writing diagnostics";
  if (lower.includes("calibrat")) return "Calibrating";
  if (lower.includes("initialize")) return "Initializing";
  if (lower.includes("load_checkpoint")) return "Loading checkpoint";
  if (lower.includes("publish")) return "Publishing checkpoint";
  if (lower.includes("selfplay")) return "Self-play";
  if (lower.includes("shuffle")) return "Shuffling data";
  if (lower.includes("evaluat")) return "Evaluating";
  if (lower.includes("train")) return "Training";
  // Unknown id: prettify rather than dumping the raw token.
  return raw.replace(/[_-]+/g, " ").replace(/\b\w/g, ch => ch.toUpperCase());
}

function runStageLabel(status) {
  if (!status || typeof status !== "object") return "--";
  const stage = humanizeStageId(status.stage || "unknown");
  // Epoch ids already carry the epoch number; avoid "Epoch 1 · e1".
  const epochNum = asFinite(status.current_epoch);
  const epoch = epochNum !== null && !/^epoch/i.test(String(status.stage || "")) ? ` · Epoch ${epochNum}` : "";
  // Prefer the derived within-epoch sub-phase (self-play / shuffling / training /
  // evaluating) over the generic "running" stage_status, so the long SealBot eval
  // is distinguishable from the rest of an epoch. Falls back to stage_status when
  // no sub-phase is available.
  const subPhase = status.sub_phase ? String(status.sub_phase) : "";
  if (subPhase) {
    const detail = status.sub_phase_detail ? ` ${String(status.sub_phase_detail)}` : "";
    return `${stage}${epoch} · ${subPhase}${detail}`;
  }
  const stageStatus = status.stage_status && status.stage_status !== "unknown"
    ? ` · ${String(status.stage_status).replace(/[_-]+/g, " ")}`
    : "";
  return `${stage}${epoch}${stageStatus}`;
}

function averageHistoryLength(histories) {
  const lengths = (histories || [])
    .map(item => Number(item.length || item.actions || 0))
    .filter(value => Number.isFinite(value) && value > 0);
  return lengths.length ? lengths.reduce((sum, value) => sum + value, 0) / lengths.length : null;
}

function asFinite(value) {
  // Treat null/undefined/"" as missing, not as the numeric 0 that Number()
  // would coerce them to (Number(null) === 0). A real numeric 0 still passes.
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function formatDecimal(value, digits = 1) {
  const number = asFinite(value);
  if (number === null) return "--";
  return number.toFixed(digits);
}

function formatRate(value, unit) {
  const number = asFinite(value);
  if (number === null) return "--";
  return `${number.toFixed(number >= 100 ? 0 : 1)} ${unit}`;
}

function formatPercent(value, digits = 0) {
  const number = asFinite(value);
  if (number === null) return "--";
  return `${(number * 100).toFixed(digits)}%`;
}

function formatGib(value) {
  const number = asFinite(value);
  if (number === null) return "--";
  // Values are GiB (binary); label honestly to match the /1024 sizing used
  // elsewhere rather than the misleading decimal "GB".
  return `${number.toFixed(1)} GiB`;
}

function historyWinStats(items) {
  const rows = (items || []).filter(item => item && item.status === "completed");
  const p0Wins = rows.filter(item => item.winner === "player0").length;
  const p1Wins = rows.filter(item => item.winner === "player1").length;
  const games = rows.length;
  return { games, p0Wins, p1Wins };
}

function historyWinRateText(stats) {
  if (!stats || !stats.games) return "P0 -- | P1 --";
  return `P0 ${formatPercent(stats.p0Wins / stats.games)} | P1 ${formatPercent(stats.p1Wins / stats.games)}`;
}

function historyWinRateSubtext(stats, label) {
  if (!stats || !stats.games) return `${label} | 0 completed`;
  return `${label} | ${stats.p0Wins}-${stats.p1Wins} (${stats.games}g)`;
}

function renderHistoryOverview(histories, filtered) {
  const epochs = historyEpochs(filtered);
  const runCount = new Set(histories.map(item => item.run).filter(Boolean)).size;
  const filteredRunCount = new Set(filtered.map(item => item.run).filter(Boolean)).size;
  const lengths = filtered.map(item => Number(item.length || item.actions || 0)).filter(value => Number.isFinite(value) && value > 0);
  const avgLength = lengths.length ? (lengths.reduce((sum, value) => sum + value, 0) / lengths.length).toFixed(1) : "--";
  const p0Wins = filtered.filter(item => item.winner === "player0").length;
  const p1Wins = filtered.filter(item => item.winner === "player1").length;
  const completed = filtered.filter(item => item.status === "completed").length;
  const evalSummary = latestDiagnosticSummary(histories, "evaluation");
  const selfplaySummary = latestDiagnosticSummary(histories, "selfplay");
  const liveStatus = latestRunStatusForHistoryPage();
  const targets = currentHistoryTargets();
  const liveWatchdog = liveStatus && liveStatus.watchdog ? liveStatus.watchdog : {};
  const liveCalibration = liveStatus && liveStatus.calibration ? liveStatus.calibration : {};
  const liveSelfplay = liveStatus && liveStatus.selfplay_live ? liveStatus.selfplay_live : {};
  const liveTraining = liveStatus && liveStatus.training_progress ? liveStatus.training_progress : {};
  const currentSelfplayRows = targets.currentEpoch !== null
    ? filtered.filter(item => item.source === "selfplay" && asFinite(item.epoch) === targets.currentEpoch)
    : [];
  const currentSelfplayStats = historyWinStats(currentSelfplayRows);
  const currentLabel = targets.currentEpoch !== null ? `Current e${targets.currentEpoch}` : "Current epoch";
  const evaluationRows = filtered.filter(item => item.source === "evaluation" && targets.evaluationEpochs.has(asFinite(item.epoch)));
  const evaluationStats = historyWinStats(evaluationRows);
  const evaluationEpoch = [...targets.evaluationEpochs][0];
  const evaluationLabel = evaluationEpoch !== undefined ? `Eval e${evaluationEpoch}` : "Eval";
  const totalMatches = historyPage.totalMatches !== null && historyPage.totalMatches !== undefined
    ? historyPage.totalMatches
    : null;
  const gameCountText = totalMatches !== null
    ? `${filtered.length} / ${totalMatches}`
    : historyPage.countLoading
      ? `${filtered.length} / counting`
    : historyPage.loaded || historyPage.items.length
      ? `${filtered.length} loaded`
      : `${filtered.length} / ${histories.length}`;
  const cards = [
    ["Stage", runStageLabel(liveStatus), "Live trainer"],
    ["Runs", filteredRunCount || runCount || "--", historySelectedRun === HISTORY_ALL_RUNS ? "Filtered / loaded runs" : "Selected run"],
    ["Games", gameCountText, historyPage.loaded || historyPage.items.length ? "Paged result set" : "Filtered / recent"],
    ["Epochs", epochs.length ? `${epochs[0]}-${epochs[epochs.length - 1]}` : "--", `${epochs.length} observed`],
    ["Winners", historyWinRateText(currentSelfplayStats), historyWinRateSubtext(currentSelfplayStats, currentLabel)],
    ["Avg Length", avgLength, "Moves per game"],
  ];
  const liveSpeed = liveSelfplay && liveSelfplay.search_pos_s !== undefined && liveSelfplay.search_pos_s !== null;
  const liveSelfplayEpoch = asFinite(liveSelfplay && liveSelfplay.epoch);
  const liveSelfplayEpochLabel = liveSelfplayEpoch !== null ? `e${liveSelfplayEpoch}` : "selfplay";
  if (liveSpeed && liveSelfplay.live) {
    const games = liveSelfplay.requested_games
      ? `${liveSelfplay.games_finished || 0}/${liveSelfplay.requested_games} games`
      : "selfplay";
    cards.push(["Speed", formatRate(liveSelfplay.search_pos_s, "pos/s"), `● LIVE · ${liveSelfplayEpochLabel} · ${games}`]);
  } else if (liveSpeed && liveSelfplay.status === "completed") {
    cards.push(["Speed", formatRate(liveSelfplay.search_pos_s, "pos/s"), `${liveSelfplayEpochLabel} selfplay (done)`]);
  } else if (liveCalibration && liveCalibration.selfplay_pos_s !== undefined && liveCalibration.selfplay_pos_s !== null) {
    cards.push(["Speed", formatRate(liveCalibration.selfplay_pos_s, "pos/s"), liveCalibration.exact_128 ? "Exact 128 sims" : "Calibration"]);
  }
  if (liveWatchdog && (liveWatchdog.free_ram_gb !== undefined || liveWatchdog.gpu_free_gb !== undefined)) {
    cards.push(["Resources", `${formatGib(liveWatchdog.free_ram_gb)} RAM | ${formatGib(liveWatchdog.gpu_free_gb)} GPU`, liveWatchdog.status || "watchdog"]);
  }
  if (liveTraining && liveTraining.epoch !== undefined) {
    cards.push(["Training", formatTrainingProgress(liveTraining), trainingProgressSubtext(liveTraining)]);
  }
  if (evaluationRows.length) {
    cards.push(["Evaluation", historyWinRateText(evaluationStats), historyWinRateSubtext(evaluationStats, evaluationLabel)]);
  } else if (evalSummary) {
    cards.push(["Evaluation", historyDiagnosticsText({ evaluation: { summary: evalSummary } }), "Latest diagnostics"]);
  }
  if (selfplaySummary) cards.push(["Selfplay", historyDiagnosticsText({ selfplay: { summary: selfplaySummary } }), "Latest diagnostics"]);
  return cards.map(([label, value, sub]) => `
    <div class="history-metric-card">
      <span>${escapeText(label)}</span>
      <strong>${escapeText(value)}</strong>
      <small>${escapeText(sub)}</small>
    </div>
  `).join("");
}

function formatTrainingProgress(progress) {
  const epochNum = progress ? asFinite(progress.epoch) : null;
  const epoch = epochNum !== null ? `e${epochNum}` : "train";
  const pct = progress && progress.progress !== undefined && progress.progress !== null ? formatPercent(progress.progress) : "--";
  return `${epoch} ${pct}`;
}

function trainingProgressSubtext(progress) {
  if (!progress || typeof progress !== "object") return "Training progress";
  const steps = asFinite(progress.steps) !== null && asFinite(progress.total_steps) !== null
    ? `${progress.steps}/${progress.total_steps} steps`
    : "steps pending";
  const loss = progress.loss !== undefined && progress.loss !== null ? `loss ${formatDecimal(progress.loss, 3)}` : String(progress.status || "training");
  return `${steps} | ${loss}`;
}

function renderLearningHealth(runs) {
  const health = latestLearningHealth(runs);
  if (!health) return "";
  const messages = Array.isArray(health.messages) ? health.messages : [];
  const status = health.status || "collecting";
  const metrics = [
    ["Latest", health.latest_epoch ? `Epoch ${health.latest_epoch}` : "--"],
    ["Loss", health.latest_loss !== null && health.latest_loss !== undefined ? formatDecimal(health.latest_loss, 3) : "--"],
    ["Eval", health.latest_eval_mean_turns !== null && health.latest_eval_mean_turns !== undefined ? `${formatDecimal(health.latest_eval_mean_turns, 1)} turns` : "--"],
    ["Best", health.best_eval_mean_turns !== null && health.best_eval_mean_turns !== undefined ? `${formatDecimal(health.best_eval_mean_turns, 1)} turns` : "--"],
    ["Speed", formatRate(health.latest_selfplay_pos_s, "pos/s")],
    ["Exact", health.latest_exact_128 ? "128 sims" : "--"],
    ["Classical", formatPercent(health.latest_classical_fraction)],
    ["Policy@1", formatPercent(health.latest_policy_top1)],
    ["Target Mass", formatPercent(health.latest_policy_target_mass, 1)],
  ];
  return `<section class="learning-health-panel ${learningHealthClass(status)}" aria-label="Learning health">
    <div class="learning-health-head">
      <span>Learning Health</span>
      <strong>${escapeText(learningHealthLabel(status))}</strong>
    </div>
    <div class="learning-health-metrics">
      ${metrics.map(([label, value]) => `<div><span>${escapeText(label)}</span><strong>${escapeText(value)}</strong></div>`).join("")}
    </div>
    <div class="learning-health-messages">
      ${messages.slice(0, 4).map(message => `<div>${escapeText(message)}</div>`).join("")}
    </div>
  </section>`;
}

function latestLearningHealth(runs) {
  return runs
    .map(run => run && run.learning_health)
    .filter(Boolean)
    .sort((a, b) => Number(b.latest_epoch || 0) - Number(a.latest_epoch || 0))[0] || null;
}

function learningHealthClass(status) {
  if (status === "intervene") return "intervene";
  if (status === "watch") return "watch";
  if (status === "improving") return "improving";
  return "ok";
}

function learningHealthLabel(status) {
  if (status === "intervene") return "Intervene";
  if (status === "watch") return "Watch";
  if (status === "improving") return "Improving";
  if (status === "collecting") return "Collecting";
  return "OK";
}

function renderEvaluationTrend(runs) {
  const rows = runs
    .flatMap(run => (run.evaluation_history || []).map(item => ({ ...item, run: run.name })))
    .filter(item => item.epoch !== undefined && item.epoch !== null)
    .sort((a, b) => Number(a.epoch || 0) - Number(b.epoch || 0) || String(a.run || "").localeCompare(String(b.run || "")));
  if (!rows.length) {
    return `<div class="eval-trend-empty">No SealBot evaluation diagnostics yet.</div>`;
  }
  const latest = rows[rows.length - 1];
  const best = rows.reduce((current, item) => {
    const currentTurns = Number(current.mean_turns || 0);
    const itemTurns = Number(item.mean_turns || 0);
    if (itemTurns !== currentTurns) return itemTurns > currentTurns ? item : current;
    return Number(item.wins || 0) > Number(current.wins || 0) ? item : current;
  }, rows[0]);
  return `<section class="eval-trend-panel" aria-label="SealBot evaluation trend">
    <div class="eval-trend-head">
      <div>
        <span>SealBot Evaluation Trend</span>
        <strong>${escapeText(evalTrendSummary(latest))}</strong>
      </div>
      <small>Best survival: ${escapeText(evalTrendSummary(best))}</small>
    </div>
    <div class="eval-trend-list">
      ${rows.slice(-12).map(evalTrendRow).join("")}
    </div>
  </section>`;
}

function evalTrendRow(item) {
  const games = Number(item.games || 0);
  const wins = Number(item.wins || 0);
  const losses = Number(item.losses || 0);
  const meanTurns = formatDecimal(item.mean_turns, 1);
  const winRate = games > 0 ? `${((wins / games) * 100).toFixed(0)}%` : "--";
  const cls = wins > 0 ? "has-win" : "no-win";
  return `<div class="eval-trend-row ${cls}">
    <strong>Epoch ${escapeText(item.epoch)}</strong>
    <span>${escapeText(`${wins}-${losses}`)}</span>
    <span>${escapeText(meanTurns)} turns</span>
    <span>${escapeText(winRate)} win</span>
  </div>`;
}

function evalTrendSummary(item) {
  if (!item) return "--";
  return `E${item.epoch}: ${Number(item.wins || 0)}-${Number(item.losses || 0)}, ${formatDecimal(item.mean_turns, 1)} turns`;
}

function renderEpochProgress(runs) {
  const rows = runs
    .flatMap(run => (run.epoch_history || []).map(item => ({ ...item, run: run.name })))
    .sort((a, b) => Number(a.epoch || 0) - Number(b.epoch || 0) || String(a.run || "").localeCompare(String(b.run || "")));
  if (!rows.length) return "";
  return `<section class="epoch-progress-panel" aria-label="Epoch progress">
    <div class="epoch-progress-head">
      <span>Epoch Progress</span>
      <strong>${escapeText(epochProgressSummary(rows[rows.length - 1]))}</strong>
    </div>
    <div class="epoch-progress-table">
      <div class="epoch-progress-row epoch-progress-header">
        <span>Epoch</span>
        <span>Selfplay</span>
        <span>Train</span>
        <span>Eval</span>
        <span>D6</span>
        <span>Checkpoint</span>
      </div>
      ${rows.slice(-10).map(epochProgressRow).join("")}
    </div>
  </section>`;
}

function epochChip(label, value, extraClass = "") {
  return `<span class="epoch-chip ${extraClass}"><i>${escapeText(label)}</i> ${escapeText(value)}</span>`;
}

// Optional full-width detail band beneath an epoch row: a Buffer group, a per-head
// Losses group (total + policy/value/opp + every short-term-value head), and a
// Calibration group (value-head optimism). Rendered only when the producer emits
// the nested `buffer` object (hexgt RL); dense_cnn rows have no buffer, so this
// returns "" and the row is unchanged (additive / dense-safe).
function epochProgressDetail(buf) {
  if (!buf) return "";
  const k = (n) => (asFinite(n) === null ? "--" : `${Math.round(Number(n) / 1000)}k`);
  const windowSpan = buf.window_span ? ` [${buf.window_span}]` : "";
  const bufferChips = [
    epochChip("pool", `${k(buf.samples)}/${k(buf.cap)}`),
    epochChip("window", `${asFinite(buf.window_epochs) ?? "--"}ep${windowSpan}`),
    epochChip("decay", formatDecimal(buf.decay, 2)),
    epochChip("train", `${asFinite(buf.train_steps) ?? "--"}×${asFinite(buf.train_batch) ?? "--"} = ${k(buf.train_samples_per_epoch)}/ep`),
  ].join("");

  // Per-head losses; each head is emitted only when present, so a run missing a
  // head (e.g. pre-deploy epochs without stvalue) just omits that chip.
  const lossChips = [];
  if (asFinite(buf.loss_total) !== null) lossChips.push(epochChip("Σ total", formatDecimal(buf.loss_total, 3), "epoch-chip-total"));
  if (asFinite(buf.loss_policy) !== null) lossChips.push(epochChip("policy", formatDecimal(buf.loss_policy, 3)));
  if (asFinite(buf.loss_value) !== null) lossChips.push(epochChip("value", formatDecimal(buf.loss_value, 3)));
  if (asFinite(buf.loss_opp) !== null) lossChips.push(epochChip("opp", formatDecimal(buf.loss_opp, 3)));
  // Short-term-value heads: every loss_stvalue_<h> the bridge surfaced, by horizon.
  Object.keys(buf)
    .map(key => /^loss_stvalue_(\d+)$/.exec(key))
    .filter(Boolean)
    .sort((a, b) => Number(a[1]) - Number(b[1]))
    .forEach(match => {
      if (asFinite(buf[match[0]]) !== null) lossChips.push(epochChip(`stv${match[1]}`, formatDecimal(buf[match[0]], 3)));
    });

  const lossGroup = lossChips.length
    ? `<div class="epoch-detail-group"><span class="epoch-detail-label">Losses</span>${lossChips.join("")}</div>`
    : "";
  // Value-head calibration: optimism_sum_mean (0 = zero-sum-consistent, >0 =
  // optimistic). Emitted by the bridge once the per-epoch calib line is logged.
  const calibGroup = asFinite(buf.optimism_sum_mean) !== null
    ? `<div class="epoch-detail-group"><span class="epoch-detail-label">Calibration</span>${epochChip("optimism", formatDecimal(buf.optimism_sum_mean, 3))}</div>`
    : "";
  return `<div class="epoch-progress-detail">
    <div class="epoch-detail-group"><span class="epoch-detail-label">Buffer</span>${bufferChips}</div>
    ${lossGroup}
    ${calibGroup}
  </div>`;
}

function epochProgressRow(item) {
  const selfplay = item.selfplay || {};
  const training = item.training || {};
  const evaluation = item.evaluation || {};
  const d6 = item.d6 || {};
  const checkpoint = item.checkpoint || {};
  const samplesAdded = asFinite(selfplay.samples_added);
  const selfplayRate = formatRate(selfplay.search_positions_per_second, "pos/s");
  let selfplayText = samplesAdded !== null
    ? `${samplesAdded} samples | ${selfplayRate}`
    : (selfplay.search_positions_per_second !== undefined && selfplay.search_positions_per_second !== null
      ? selfplayRate
      : "pending");
  // Game-length stats (mean/median/max/stdev). hexgnn/hexgt emit these inline;
  // dense_cnn's are backfilled server-side from the epoch .hxr (web.py). Each
  // segment is appended only when its field is present, so a run with neither is
  // unaffected and a run with partial data shows what it has.
  const lenMean = asFinite(selfplay.game_length_mean);
  const lenMed = asFinite(selfplay.game_length_median);
  if (lenMean !== null || lenMed !== null) {
    selfplayText += ` | len μ${formatDecimal(lenMean, 1)} med ${formatDecimal(lenMed, 0)}`
      + ` max ${asFinite(selfplay.game_length_max) ?? "--"} σ${formatDecimal(selfplay.game_length_stdev, 1)}`;
  }
  // Outcome distribution (P0/P1 win + draw share). Same backfill story as lengths.
  const winP0 = asFinite(selfplay.win_p0_fraction);
  const winP1 = asFinite(selfplay.win_p1_fraction);
  const drawFrac = asFinite(selfplay.draw_fraction);
  if (winP0 !== null || winP1 !== null) {
    selfplayText += ` | W ${formatPercent(winP0)}/${formatPercent(winP1)}`;
    if (drawFrac !== null) selfplayText += ` d ${formatPercent(drawFrac)}`;
  }
  const trainText = (training.loss !== undefined && training.loss !== null)
    ? `loss ${formatDecimal(training.loss, 3)} | C ${formatPercent(classicalReplayFraction(training))} | P@1 ${formatPercent(policyTop1(training))}`
    : training.progress
      ? `${formatTrainingProgress({ ...training.progress, epoch: item.epoch })} | ${trainingProgressSubtext(training.progress)}`
    : "pending";
  const evalText = (evaluation.games !== undefined && evaluation.games !== null)
    ? `${Number(evaluation.wins || 0)}-${Number(evaluation.losses || 0)} | ${formatDecimal(evaluation.mean_turns, 1)} turns`
    : "pending";
  const d6Text = d6.preview_symmetries && d6.preview_symmetries.length
    ? `${d6.preview_symmetries.slice(0, 6).join(",")}${d6.preview_symmetries.length > 6 ? "..." : ""}`
    : (d6.mode ? "random" : "pending");
  const checkpointText = checkpoint.path || checkpoint.name ? "saved" : "pending";
  const status = item.status || "partial";
  const mainRow = `<div class="epoch-progress-row ${status === "completed" ? "completed" : "partial"}">
    <strong>Epoch ${escapeText(item.epoch)}</strong>
    <span>${escapeText(selfplayText)}</span>
    <span>${escapeText(trainText)}</span>
    <span>${escapeText(evalText)}</span>
    <span>${escapeText(d6Text)}</span>
    <span>${escapeText(checkpointText)}</span>
  </div>`;
  // The bridge attaches the replay-buffer + per-head-loss + calibration block to
  // the selfplay payload, and web.py passes it through as selfplay.buffer.
  const buf = selfplay.buffer || null;
  return `<div class="epoch-progress-group">${mainRow}${epochProgressDetail(buf)}</div>`;
}

function classicalReplayFraction(training) {
  const counts = training && training.source_summary && training.source_summary.source_counts;
  if (!counts || typeof counts !== "object") return null;
  let total = 0;
  let classical = 0;
  for (const [key, rawValue] of Object.entries(counts)) {
    const value = Number(rawValue);
    if (!Number.isFinite(value)) continue;
    total += value;
    if (String(key).toLowerCase().includes("classical")) classical += value;
  }
  return total > 0 ? classical / total : null;
}

function policyTop1(training) {
  const overall = training && training.policy_imitation && training.policy_imitation.overall;
  return overall ? overall.top1_accuracy : null;
}

function epochProgressSummary(item) {
  if (!item) return "--";
  const training = item.training || {};
  const evaluation = item.evaluation || {};
  const parts = [`E${item.epoch}`];
  if (training.loss !== undefined && training.loss !== null) parts.push(`loss ${formatDecimal(training.loss, 3)}`);
  if (evaluation.games !== undefined && evaluation.games !== null) parts.push(`eval ${Number(evaluation.wins || 0)}-${Number(evaluation.losses || 0)}`);
  if (item.status && item.status !== "completed") parts.push(item.status);
  return parts.join(" | ");
}

function latestDiagnosticSummary(histories, label) {
  for (const item of histories || []) {
    const diagnostic = item.diagnostics && item.diagnostics[label];
    if (diagnostic && diagnostic.summary) return diagnostic.summary;
  }
  return null;
}

function gameHistoryListRow(runName, item) {
  const winner = item.winner_label || winnerLabel(item.winner);
  const status = item.status || "unknown";
  const epoch = item.epoch ? `Epoch ${item.epoch}` : "No epoch";
  const source = item.source || "history";
  const p0 = historyPlayerLabel(item.players && item.players.player0);
  const p1 = historyPlayerLabel(item.players && item.players.player1);
  const diagnostics = historyDiagnosticsText(item.diagnostics);
  const key = historyItemKey(item);
  const selected = key === selectedHistoryKey;
  return `<div class="game-history-row ${selected ? "selected" : ""}" data-history-key="${escapeAttr(key)}">
    <button class="game-history-select" type="button" data-history-key="${escapeAttr(key)}">
      <span class="history-game-title" title="${escapeAttr(item.game_id || item.path || "")}">${escapeText(item.game_id || item.path || "—")}</span>
      <span class="history-game-meta" title="${escapeAttr(`${runName || "run"} | ${item.path || ""}`)}">${escapeText(runName || "run")} | ${escapeText(item.path || "—")}</span>
    </button>
    <div><strong>${escapeText(epoch)}</strong><span>${escapeText(source)} | ${escapeText(status)}</span></div>
    <div><span class="winner-pill ${winnerClass(item.winner)}">${escapeText(winner)}</span></div>
    <div><strong>${escapeText(item.length || item.actions || 0)}</strong><span>moves</span></div>
    <div><strong>P0 ${escapeText(p0)}</strong><span>P1 ${escapeText(p1)}</span></div>
    <div class="history-row-actions">
      <strong>${escapeText(diagnostics)}</strong>
      <span>${escapeText(formatHistoryDate(item.modified))}</span>
      <button class="history-row-load" type="button" data-history-load data-history-run="${escapeAttr(runName || "")}" data-history-path="${escapeAttr(item.path)}" data-record-index="${escapeAttr(item.record_index || 0)}">Replay</button>
    </div>
  </div>`;
}

function gameHistoryDetailHtml(runName, item) {
  const winner = item.winner_label || winnerLabel(item.winner);
  const diagnostics = item.diagnostics || {};
  const p0 = item.players && item.players.player0;
  const p1 = item.players && item.players.player1;
  return `<div class="history-detail-body">
    <div class="history-detail-hero">
      <div>
        <span>Winner</span>
        <strong class="${winnerClass(item.winner)}">${escapeText(winner)}</strong>
      </div>
      <div>
        <span>Length</span>
        <strong>${escapeText(item.length || item.actions || 0)}</strong>
      </div>
      <div>
        <span>Epoch</span>
        <strong>${escapeText(item.epoch || "--")}</strong>
      </div>
      <div>
        <span>Source</span>
        <strong>${escapeText(item.source || "history")}</strong>
      </div>
    </div>
    <div class="detail-stack">
      ${detailRow("Run", runName || "Unknown")}
      ${detailRow("Game", item.game_id || "Unknown")}
      ${detailRow("Status", item.status || "unknown")}
      ${detailRow("Seed", item.seed === null || item.seed === undefined ? "--" : item.seed)}
      ${detailRow("Record", Number(item.record_index || 0))}
      ${detailRow("Path", item.path || "—", item.path || "")}
      ${detailRow("Modified", formatHistoryDate(item.modified))}
    </div>
    <div class="history-detail-section">
      <div class="detail-section-title">Players</div>
      <div class="player-detail-grid">
        ${playerDetail("P0", p0)}
        ${playerDetail("P1", p1)}
      </div>
    </div>
    <div class="history-detail-section">
      <div class="detail-section-title">Diagnostics</div>
      ${diagnosticDetailsHtml(diagnostics)}
    </div>
    ${item.abort ? `<div class="history-detail-section"><div class="detail-section-title">Abort</div><div class="detail-note">${escapeText(item.abort)}</div></div>` : ""}
    <div class="history-detail-actions">
      <button class="primary-action history-replay-btn" type="button" data-history-load data-history-run="${escapeAttr(runName || "")}" data-history-path="${escapeAttr(item.path)}" data-record-index="${escapeAttr(item.record_index || 0)}">Load Replay</button>
      ${String(item.path || "").endsWith(".hxr") ? `<button class="history-debug-btn" type="button" data-debug-open data-debug-run="${escapeAttr(runName || "")}" data-debug-path="${escapeAttr(item.path)}" data-debug-record="${escapeAttr(item.record_index || 0)}">Open in Debug</button>` : ""}
    </div>
  </div>`;
}

function detailRow(label, value, titleValue) {
  const title = titleValue ? ` title="${escapeAttr(titleValue)}"` : "";
  return `<div class="detail-row"><span>${escapeText(label)}</span><strong${title}>${escapeText(value)}</strong></div>`;
}

function playerDetail(slot, player) {
  const label = historyPlayerLabel(player);
  const kind = player && (player.kind || player.variant || player.id || "");
  return `<div class="player-detail">
    <span>${escapeText(slot)}</span>
    <strong>${escapeText(label)}</strong>
    <small>${escapeText(kind || "unknown")}</small>
  </div>`;
}

function diagnosticDetailsHtml(diagnostics) {
  if (!diagnostics || typeof diagnostics !== "object" || !Object.keys(diagnostics).length) {
    return `<div class="detail-note">No diagnostics attached to this game.</div>`;
  }
  return Object.entries(diagnostics).map(([label, diagnostic]) => {
    const summary = diagnostic && diagnostic.summary ? diagnostic.summary : {};
    const entries = Object.entries(summary);
    return `<div class="diagnostic-block">
      <div class="diagnostic-title">${escapeText(label)}</div>
      <div class="diagnostic-grid">
        ${entries.length ? entries.map(([key, value]) => `<div><span>${escapeText(key)}</span><strong>${escapeText(displayValue(value))}</strong></div>`).join("") : `<div><span>Artifact</span><strong>${escapeText(diagnostic && diagnostic.name || "attached")}</strong></div>`}
      </div>
    </div>`;
  }).join("");
}

function historyPlayerLabel(player) {
  if (!player) return "Unknown";
  return player.label || PLAYER_KIND_LABELS[player.kind] || player.kind || "Unknown";
}

function winnerLabel(winner) {
  if (winner === "player0") return "P0";
  if (winner === "player1") return "P1";
  return "None";
}

function historyDiagnosticsText(diagnostics) {
  if (!diagnostics || typeof diagnostics !== "object") return "None";
  const evalSummary = diagnostics.evaluation && diagnostics.evaluation.summary;
  if (evalSummary) {
    const parts = [];
    if (asFinite(evalSummary.games) !== null) parts.push(`${evalSummary.games}g`);
    if (asFinite(evalSummary.wins) !== null || asFinite(evalSummary.losses) !== null) parts.push(`${evalSummary.wins || 0}-${evalSummary.losses || 0}`);
    if (asFinite(evalSummary.mean_turns) !== null) parts.push(`${asFinite(evalSummary.mean_turns).toFixed(1)}t`);
    return parts.length ? parts.join(" ") : "Eval";
  }
  const selfplaySummary = diagnostics.selfplay && diagnostics.selfplay.summary;
  if (selfplaySummary) {
    const parts = [];
    if (asFinite(selfplaySummary.samples_added) !== null) parts.push(`${selfplaySummary.samples_added} samples`);
    if (asFinite(selfplaySummary.games) !== null) parts.push(`${selfplaySummary.games}g`);
    if (selfplaySummary.lengths && asFinite(selfplaySummary.lengths.mean) !== null) parts.push(`${asFinite(selfplaySummary.lengths.mean).toFixed(1)}t`);
    if (parts.length) return parts.join(" ");
    if (asFinite(selfplaySummary.searched_positions) !== null) return `${selfplaySummary.searched_positions} pos`;
    return "Selfplay";
  }
  return Object.keys(diagnostics).length ? Object.keys(diagnostics).join(", ") : "None";
}

function winnerClass(winner) {
  if (winner === "player0") return "p0";
  if (winner === "player1") return "p1";
  return "none";
}

function formatHistoryDate(value) {
  if (!value) return "--";
  const raw = Number(value);
  const date = Number.isFinite(raw) ? new Date(raw < 1000000000000 ? raw * 1000 : raw) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function formatBytes(value) {
  const bytes = asFinite(value) || 0;
  // Binary (1024-based) divisions, so use binary unit labels (KiB/MiB).
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

// ===========================================================================
// Debug tab — Model Debug v2 (single-position forensics workbench).
//
// Self-contained from the Match/History screens: its own board SVG, state and
// pan/zoom, reusing only the pure shared helpers (center/path/playerColor/HEX,
// viewForBox/clamp, escape*, formatBytes, loadTrainingRuns, navigateScreen,
// diagTap/reportError). All model outputs come from the CPU inference worker
// via /api/debug/* per scripts/_debug_rewrite_spec.md §3; nothing here touches
// the live-match polling.
//
// Canonical nav state (spec M1): one `dbg.nav` tuple serialized to the URL
// hash (`#debug?run=..&src=..&path=..&rec=..&ply=..&ckptA=..&ckptB=..&acts=..
// &tab=..&mode=..`). `dbgNavigate(patch, {replace})` is the ONLY mutation
// entry point — it writes the hash, and `dbgApplyHash()` (driven by
// `hashchange`) is the only code that applies state → fetch/render. Browser
// Back/Forward therefore step the workbench, and any URL restores the exact
// position including injected what-if moves and checkpoint B.
// ===========================================================================

const DBG_TABS = ["heads", "search", "targets", "compare", "inputs", "ckpt"];
const DBG_TAB_IDS = {
  heads: "dbgTabHeads",
  search: "dbgTabSearch",
  targets: "dbgTabTargets",
  compare: "dbgTabCompare",
  inputs: "dbgTabInputs",
  ckpt: "dbgTabCkpt",
};
// Base heat modes — index in DBG_MODE_ORDER is the 1..8 keyboard digit.
const DBG_MODE_ORDER = ["prior", "visits", "delta", "opp", "target", "mismatch", "childq", "plane"];
const DBG_MODES = ["none"].concat(DBG_MODE_ORDER);
const DBG_CACHE_MAX = 300;  // spec M12: client analysis cache LRU cap
// Action ids are packed exactly like hexo_engine.types.pack_coord_id (which
// mirrors the Rust legal-move store, so the packing is pinned): the board's
// click-to-inject needs an id for ANY empty legal cell, and the position
// payload's legal list carries only coordinates.
const DBG_COORD_OFFSET = 32768;

const dbg = {
  inited: false,
  loading: false,
  // Canonical nav tuple (spec M1). `ply: null` means "end of game, not yet
  // resolved" — dbgApplyNav replaces it with the real total once known.
  nav: dbgDefaultNav(),
  navApplied: "",      // last hash fully handed to dbgApplyNav (dedupes the two hashchange listeners)
  pendingDeepLink: null,
  // Loaded source lists + their identity (so run/src switches reload them).
  loadedRun: "",
  loadedGamesKey: "",
  games: [],
  checkpoints: [],
  worker: null,        // worker status from /api/debug/checkpoints
  records: [],         // record_games from the last recorded position payload
  // Current data + the cache key each piece was rendered for (stale dots, M14).
  position: null,
  analysis: null,      // ckptA analyze
  analysisB: null,     // ckptB analyze (COMPARE)
  search: null,        // /api/debug/search result
  searchBusy: false,
  tree: null,          // /api/debug/search_tree result (childq/Q/scatter/PV read it)
  treeBusy: false,
  treeOpen: new Map(), // tree-node path -> explicit expand/collapse (default: PV open)
  treePreview: null,   // {key, ids} — clicked tree node's line, ghosted on the board
  treeGhostsOff: false, // T with a fresh tree toggles its board ghosts (§1.7 "run / toggle")
  ladder: null,        // visit-ladder rows for ladderKey (spec S10)
  ladderBusy: false,
  // Game Error Sweeps (spec M9), keyed run|path|rec|ckptA so flipping checkpoints
  // or games keeps each sweep; sweepRun is the single in-flight chunk loop.
  sweeps: new Map(),
  sweepRun: null,
  trajectory: null,
  trajectoryKey: "",
  cmpHeat: false,      // COMPARE: client-computed prior Δ(A−B) board overlay
  split: false,        // COMPARE: A/B split-board mode (spec S5)
  pins: [],            // pin tray (spec S2), localStorage hexoDbgPins:<run>
  pinsRun: "",
  journal: [],         // session journal (spec S3), localStorage hexoDbgJournal:<run>
  journalRun: "",
  paletteItems: [],    // command palette (spec S1)
  paletteShown: [],
  paletteIdx: 0,
  paletteGames: [],    // all-source game list for the palette (lazy)
  paletteGamesRun: "",
  ckptSweep: null,     // checkpoint sweep dock tab (spec S4)
  prefetchAbort: null, // the ONE in-flight ply±1 prefetch (spec S7)
  keys: { position: "", analysis: "", analysisB: "", search: "", tree: "" },
  // Board UI state that intentionally does NOT live in the hash.
  overlays: { threats: false, numbers: true, last: true, legalDim: false },
  logScale: false,
  opacity: 0.9,
  sortKey: "prior",    // top-moves table sort column
  dockTab: "trajectory",
  hudMetrics: new Map(),  // per-render `q,r` -> metrics, so the hover HUD never does a linear .find (M3)
  cellIndex: new Map(),   // per-render `q,r` -> cell info for click routing
  // Debug-board pan/zoom (own copies — the Match board owns the globals).
  view: null,
  baseView: null,
  viewDirty: false,
  suppressClick: false,
  // Monotonic request tokens so the LATEST navigation/analysis always wins. The
  // dense_cnn_restnet CPU forward is far slower than hexgt's tiny graph forward,
  // so without these guards a slow/stale position fetch or analyze can resolve
  // AFTER a newer ply click and clobber dbg.position/dbg.analysis — which wedged
  // forward/back stepping and slider scrubbing for that lineage. Each load/analyze
  // claims a token and only commits if it is still the newest.
  posSeq: 0,
  anlSeq: 0,
  anlSeqB: 0,
  applySeq: 0,
  // Client caches (spec M12): one LRU keyed run|path|rec|ply|ckpt|acts holding
  // {position, analysis, search, tree, record_row}; revisiting a ply, undoing a
  // chip, or flipping ckptA<->ckptB re-renders with zero network requests.
  cache: new Map(),
  ckptInfo: new Map(),    // `${run}|${ckpt}` -> /api/debug/ckpt_info payload (B3/M7)
  // Full ordered stone list + action ids for the current RECORDED game, captured
  // from any loaded position. Lets prev/next/slider re-render the board for any
  // ply INSTANTLY client-side (stones with index <= ply) without waiting on the
  // server position fetch or the slow ResTNet analyze.
  gamePlacements: [],
  gameKey: "",
  gameActs: [],
  gameActsKey: "",
  placementsBackfill: "",  // gameKey already backfilled to the final ply (one-shot)
};

const dbgEl = id => document.getElementById(id);
const debugBoardSvg = dbgEl("debugBoardSvg");

function debugSetStatus(message, kind = "info") {
  // Mirror real errors to the always-on-top banner so they're unmissable on a
  // phone (the small inline status line is easy to miss on a narrow screen).
  if (message && kind === "error") reportError(message);
  const el = dbgEl("debugStatus");
  if (!el) return;
  if (!message) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.className = `debug-status debug-status-${kind}`;
  el.textContent = message;
}

async function debugFetchJson(url, options) {
  const res = await fetch(url, options);
  const data = await safeJson(res);
  if (!res.ok || (data && data.error)) {
    // A 404 on a /api/debug/* route means the running server predates the Debug
    // backend (its routes were loaded at process start). Static assets are read
    // from disk per request, so the tab can appear while the API is missing —
    // say so plainly instead of a cryptic "HTTP 404".
    if (res.status === 404 && url.indexOf("/api/debug/") !== -1) {
      throw new Error("Debug API not found — restart the dashboard server to load the new Debug backend (it must serve the current hexo_frontend from this worktree).");
    }
    throw new Error((data && data.error) || `HTTP ${res.status}`);
  }
  return data;
}

function dbgStub(message) {
  diagTap(message);
  debugSetStatus(message, "info");
}

// ---- action-id packing (mirrors hexo_engine.types.pack/unpack_coord_id) ----

function dbgPackActionId(q, r) {
  return (q + DBG_COORD_OFFSET) * 65536 + (r + DBG_COORD_OFFSET);
}

function dbgUnpackActionId(actionId) {
  const id = Number(actionId);
  return { q: Math.floor(id / 65536) - DBG_COORD_OFFSET, r: (id % 65536) - DBG_COORD_OFFSET };
}

// ---- nav state + URL hash (spec M1) ----------------------------------------

function dbgDefaultNav() {
  return { run: "", src: "selfplay", path: "", rec: 0, ply: null, ckptA: "", ckptB: "", acts: [], tab: "heads", mode: "prior" };
}

function dbgNavToHash(nav) {
  // Deterministic field order so identical navs serialize identically (the
  // dedupe in dbgApplyHash relies on it). `/` and `,` stay readable.
  const enc = v => encodeURIComponent(String(v)).replace(/%2F/gi, "/");
  const parts = [];
  if (nav.run) parts.push("run=" + enc(nav.run));
  if (nav.src && nav.src !== "selfplay") parts.push("src=" + enc(nav.src));
  if (nav.path) parts.push("path=" + enc(nav.path));
  if (nav.rec) parts.push("rec=" + String(nav.rec));
  if (nav.ply != null) parts.push("ply=" + String(nav.ply));
  if (nav.ckptA) parts.push("ckptA=" + enc(nav.ckptA));
  if (nav.ckptB) parts.push("ckptB=" + enc(nav.ckptB));
  if (nav.acts && nav.acts.length) parts.push("acts=" + nav.acts.join(","));
  if (nav.tab && nav.tab !== "heads") parts.push("tab=" + enc(nav.tab));
  if (nav.mode && nav.mode !== "prior") parts.push("mode=" + enc(nav.mode));
  return "#debug" + (parts.length ? "?" + parts.join("&") : "");
}

function dbgHashToNav(hash) {
  const nav = dbgDefaultNav();
  const qi = hash.indexOf("?");
  if (qi === -1) return nav;
  const params = new URLSearchParams(hash.slice(qi + 1));
  nav.run = params.get("run") || "";
  nav.src = params.get("src") === "evaluation" ? "evaluation" : "selfplay";
  nav.path = params.get("path") || "";
  nav.rec = Math.max(0, parseInt(params.get("rec"), 10) || 0);
  const ply = parseInt(params.get("ply"), 10);
  nav.ply = Number.isFinite(ply) ? Math.max(0, ply) : null;
  nav.ckptA = params.get("ckptA") || "";
  nav.ckptB = params.get("ckptB") || "";
  nav.acts = (params.get("acts") || "").split(",").map(s => parseInt(s, 10)).filter(n => Number.isFinite(n));
  const tab = params.get("tab");
  nav.tab = DBG_TABS.includes(tab) ? tab : "heads";
  const mode = params.get("mode");
  nav.mode = DBG_MODES.includes(mode) ? mode : "prior";
  return nav;
}

function dbgNavigate(patch, { replace = false } = {}) {
  dbgAbortPrefetch();  // S7: any user action kills the speculative ply±1 fetch
  const nav = Object.assign({}, dbg.nav, patch);
  if (patch && patch.acts) nav.acts = patch.acts.slice();
  const hash = dbgNavToHash(nav);
  if (String(window.location.hash || "") === hash) {
    // Explicit re-selection of the identical tuple: clear the dedupe so an apply
    // that failed midway can be retried (re-apply on a cache hit is idempotent).
    dbg.navApplied = "";
    dbgApplyHash();
    return;
  }
  // Optimistic nav update: `location.hash =` fires hashchange ASYNCHRONOUSLY,
  // so without this a rapid key-repeat step would read the stale nav and drop
  // steps. dbgApplyHash re-parses the same tuple from the hash when it lands.
  dbg.nav = nav;
  if (replace) {
    // replaceState fires no hashchange — apply directly (used for default
    // resolution, slider scrubs, and tab/mode flips that shouldn't spam history).
    window.history.replaceState(null, "", hash);
    dbgApplyHash();
  } else {
    window.location.hash = hash;  // hashchange -> dbgApplyHash, the ONLY applier
  }
}

function dbgApplyHash() {
  if (!dbg.inited) return;
  if (dbg.pendingDeepLink) {
    const link = dbg.pendingDeepLink;
    dbg.pendingDeepLink = null;
    dbgNavigate(dbgDeepLinkPatch(link), { replace: true });
    return;
  }
  const hash = String(window.location.hash || "");
  if (!hash.startsWith("#debug")) return;
  if (hash === dbg.navApplied) return;  // both hashchange listeners route here — apply once
  if (hash === "#debug" && dbg.nav.run && dbg.nav.path) {
    // A bare `#debug` (setScreen's replaceState strips the query when the nav
    // buttons are used) restores the in-memory position instead of resetting.
    dbgNavigate(dbg.nav, { replace: true });
    return;
  }
  dbg.navApplied = hash;
  dbg.nav = dbgHashToNav(hash);
  const seq = ++dbg.applySeq;
  dbgApplyNav(dbg.nav, seq).catch(e => {
    if (seq === dbg.applySeq) dbg.navApplied = "";  // failed tuple stays re-appliable
    debugSetStatus(`Debug: ${(e && e.message) || e}`, "error");
    reportError("dbgApplyNav: " + (e && (e.stack || e.message) || e));
  });
}

function dbgDeepLinkPatch(link) {
  // detail from [data-debug-open]: { run, path, record, ply } — ply null = final.
  const patch = {
    run: link.run || dbg.nav.run,
    src: link.path && String(link.path).startsWith("eval") ? "evaluation" : "selfplay",
    path: link.path || "",
    rec: Number(link.record != null ? link.record : link.rec) || 0,
    ply: link.ply != null ? Number(link.ply) : null,
    acts: [],
  };
  if (patch.run !== dbg.nav.run) {
    patch.ckptA = "";  // checkpoints are per-run; let apply re-default
    patch.ckptB = "";
  }
  return patch;
}

function debugOpenFromHistory(detail) {
  // detail: { run, path, record, ply } — kept signature for the untouched
  // [data-debug-open] delegation in History/Match. The deep link is consumed by
  // dbgApplyHash on the hashchange navigateScreen() triggers (or directly from
  // enterDebugScreen when no hashchange is coming).
  dbg.pendingDeepLink = detail;
  navigateScreen("debug");
}

// ---- apply: nav -> sources -> position -> panels ----------------------------

async function dbgApplyNav(nav, seq) {
  dbgSyncControls();
  dbgRenderCrumb();
  if (!trainingRuns.length) {
    debugSetStatus("Loading runs…");
    try {
      await loadTrainingRuns();
    } catch (_e) { /* fall through with whatever we have */ }
    if (seq !== dbg.applySeq) return;
    debugSetStatus("");
  }
  const runNames = trainingRuns.map(r => r.name);
  if (!runNames.length) {
    debugSetStatus("No training runs found.", "error");
    return;
  }
  if (!nav.run || !runNames.includes(nav.run)) {
    // Prefer the run already selected on History, else the first run.
    const preferred = historySelectedRun && runNames.includes(historySelectedRun) ? historySelectedRun : runNames[0];
    dbgNavigate({ run: preferred, path: "", rec: 0, ply: null, acts: [], ckptA: "", ckptB: "" }, { replace: true });
    return;
  }
  if (dbg.loadedRun !== nav.run) {
    await dbgLoadCheckpoints(nav.run);
    if (seq !== dbg.applySeq) return;
    dbgSyncControls();
  }
  dbgLoadPins();
  dbgLoadJournal();
  if (dbg.checkpoints.length) {
    if (!nav.ckptA || !dbg.checkpoints.some(c => c.name === nav.ckptA)) {
      const latest = dbg.checkpoints.find(c => c.latest) || dbg.checkpoints[0];
      dbgNavigate({ ckptA: latest.name }, { replace: true });
      return;
    }
    if (nav.ckptB && !dbg.checkpoints.some(c => c.name === nav.ckptB)) {
      dbgNavigate({ ckptB: "" }, { replace: true });
      return;
    }
  }
  // Checkpoint provenance is position-independent — ensure it BEFORE the games/
  // position early-returns so CKPT loads even when the source has no games (M7).
  if (nav.tab === "ckpt") dbgEnsureCkptInfo(false);
  const gamesKey = `${nav.run}|${nav.src}`;
  if (dbg.loadedGamesKey !== gamesKey) {
    await dbgLoadGames(nav.run, nav.src);
    if (seq !== dbg.applySeq) return;
    dbgSyncControls();
  }
  if (!dbg.games.length) {
    dbgResetPosition();
    dbgRenderAll();
    debugSetStatus(`No ${nav.src} games in ${nav.run}.`, "error");
    return;
  }
  if (!nav.path || !dbg.games.some(g => g.path === nav.path)) {
    await dbgAutoPickGame(seq);
    return;
  }
  const recorded = dbgRecordedActs();
  if (nav.ply == null) {
    if (recorded) {
      dbgNavigate({ ply: recorded.length }, { replace: true });
      return;
    }
    await dbgResolveEndPly(seq);
    return;
  }
  if (recorded && nav.ply > recorded.length) {
    dbgNavigate({ ply: recorded.length }, { replace: true });
    return;
  }
  const key = dbgCurrentKey();
  const entry = dbg.cache.get(key);
  if (entry && entry.position) {
    // Client cache hit (M12): zero network for the board; panels refill from the
    // same entry, and only missing pieces are fetched.
    dbgCommitEntry(key, entry);
    if (nav.ckptB) {
      const keyB = dbgCacheKey(nav, nav.ckptB);
      const entryB = dbg.cache.get(keyB);
      if (entryB && entryB.analysis) {
        dbg.analysisB = entryB.analysis;
        dbg.keys.analysisB = keyB;
      }
    }
    debugSetStatus("");
    dbgRenderAll();
    if (!entry.analysis || (nav.ckptB && dbg.keys.analysisB !== dbgCacheKey(nav, nav.ckptB))) dbgScheduleFetch(0);
    // Fully cached: dbgFetchCurrent (whose tail prefetches) never runs, so extend
    // the S7 prefetch window here — else steady stepping prefetches alternate plies.
    else dbgMaybePrefetch();
  } else {
    if (!nav.acts.length && recorded && dbg.gamePlacements.length) {
      // INSTANT, analyze-independent step off the client-side placement cache;
      // the debounced fetch below fills in legal cells/tactics + analysis.
      dbgOptimisticStep();
    } else {
      debugSetStatus("Loading position…", "busy");
    }
    dbgRenderAll();
    dbgScheduleFetch(120);  // coalesce rapid steps/slider drags into one fetch
  }
  if (nav.tab === "inputs" || nav.mode === "plane") dbgEnsureInputs(false);
  if (dbgWantRecordRow(nav)) dbgEnsureRecordRow(false);
}

// ---- source loading ---------------------------------------------------------

async function dbgLoadCheckpoints(run) {
  try {
    const data = await debugFetchJson(`/api/debug/checkpoints?run=${encodeURIComponent(run)}`);
    dbg.checkpoints = data.checkpoints || [];
    dbg.worker = data.worker || null;
    dbgRenderWorkerDot(dbg.worker && dbg.worker.alive ? "ok" : "");
  } catch (e) {
    dbg.checkpoints = [];
    dbg.worker = null;
    dbgRenderWorkerDot("err");
    debugSetStatus(`Checkpoints: ${e.message}`, "error");
  }
  dbg.loadedRun = run;  // set even on error — the Refresh button clears it to retry
}

async function dbgLoadGames(run, src) {
  try {
    const data = await debugFetchJson(`/api/debug/games?run=${encodeURIComponent(run)}&source=${encodeURIComponent(src)}`);
    dbg.games = data.games || [];
  } catch (e) {
    dbg.games = [];
    debugSetStatus(`Games: ${e.message}`, "error");
  }
  dbg.loadedGamesKey = `${run}|${src}`;
}

async function dbgAutoPickGame(seq) {
  // Auto-pick the newest game that actually loads. The newest selfplay file is
  // usually the IN-PROGRESS epoch (still being written by the live run), which
  // has NO complete games yet — probe newest-first and stop at the first game
  // that loads, so the tab always opens on a usable game.
  debugSetStatus("Loading game…", "busy");
  for (const g of dbg.games) {
    try {
      const params = new URLSearchParams({ run: dbg.nav.run, path: g.path, record: "0", ply: "999999" });
      const data = await debugFetchJson(`/api/debug/position?${params.toString()}`);
      if (seq !== dbg.applySeq) return;
      const total = data.debug.total;
      const entry = dbgCacheEntry([dbg.nav.run, g.path, 0, total, dbg.nav.ckptA, ""].join("|"));
      entry.position = data;
      debugSetStatus("");
      dbgNavigate({ path: g.path, rec: 0, ply: total, acts: [] }, { replace: true });
      return;
    } catch (_e) { /* probe the next (older) file */ }
  }
  if (seq !== dbg.applySeq) return;
  dbgResetPosition();
  dbgRenderAll();
  debugSetStatus(`No loadable ${dbg.nav.src} games in ${dbg.nav.run}.`, "error");
}

async function dbgResolveEndPly(seq) {
  // ply=null means "final position"; ask the server (it clamps) then pin the
  // real total into the hash so the cache key and the URL stay canonical.
  const nav = dbg.nav;
  debugSetStatus("Loading position…", "busy");
  try {
    const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), ply: "999999" });
    const data = await debugFetchJson(`/api/debug/position?${params.toString()}`);
    if (seq !== dbg.applySeq) return;
    const total = data.debug.total;
    const entry = dbgCacheEntry([nav.run, nav.path, nav.rec, total, nav.ckptA, ""].join("|"));
    entry.position = data;
    debugSetStatus("");
    dbgNavigate({ ply: total }, { replace: true });
  } catch (e) {
    if (seq === dbg.applySeq) debugSetStatus(`Position: ${e.message}`, "error");
  }
}

function dbgRefreshSources() {
  dbg.loadedRun = "";
  dbg.loadedGamesKey = "";
  dbg.navApplied = "";
  dbgApplyHash();
}

// ---- cache + keys (spec M12) ------------------------------------------------

function dbgCacheKey(nav, ckpt) {
  return [nav.run, nav.path, nav.rec, nav.ply, ckpt || "", (nav.acts || []).join(",")].join("|");
}

function dbgCurrentKey() {
  return dbgCacheKey(dbg.nav, dbg.nav.ckptA);
}

function dbgCacheEntry(key) {
  let entry = dbg.cache.get(key);
  if (entry) {
    dbg.cache.delete(key);  // LRU bump to most-recently-used
    dbg.cache.set(key, entry);
    return entry;
  }
  entry = {};
  dbg.cache.set(key, entry);
  while (dbg.cache.size > DBG_CACHE_MAX) dbg.cache.delete(dbg.cache.keys().next().value);
  return entry;
}

function dbgRecordRow() {
  // Recorded .npz training row for the current key — fetched in stage F3; the
  // value-dist soft-z marker / target heat read it opportunistically when set.
  const entry = dbg.cache.get(dbgCurrentKey());
  return (entry && entry.record_row) || null;
}

function dbgRecordedActs() {
  const nav = dbg.nav;
  return dbg.gameActsKey === `${nav.run}|${nav.path}|${nav.rec}` && dbg.gameActs.length ? dbg.gameActs : null;
}

function dbgRecordedTotal() {
  // Total plies of the RECORDED game. While branched, dbg.position.debug.total
  // is the branch prefix length (base ply + injected tail), not the game length,
  // so the rail/crumb/clamps must read the recorded action list instead.
  const recorded = dbgRecordedActs();
  if (recorded) return recorded.length;
  if (dbg.position && !dbg.nav.acts.length) return dbg.position.debug.total;
  return null;
}

function dbgFreshData(kind) {
  // Data gated to the CURRENT nav key: the board heat/HUD/Q columns must never
  // paint a previous position's outputs. Panels instead keep their stale data
  // visible with a stale dot until the refill lands (M14).
  return dbg.keys[kind] === dbgCurrentKey() ? dbg[kind] : null;
}

function dbgResetPosition() {
  dbg.position = null;
  dbg.analysis = null;
  dbg.analysisB = null;
  dbg.search = null;
  dbg.tree = null;
  dbg.records = [];
  dbg.gameActs = [];
  dbg.gameActsKey = "";
  dbg.gamePlacements = [];
  dbg.gameKey = "";
  dbg.keys = { position: "", analysis: "", analysisB: "", search: "", tree: "" };
}

function dbgCommitEntry(key, entry) {
  dbgCommitPosition(entry.position, key);
  if (entry.analysis) {
    dbg.analysis = entry.analysis;
    dbg.keys.analysis = key;
  }
  dbg.search = entry.search || null;
  if (entry.search) dbg.keys.search = key;
  dbg.tree = entry.tree || null;
  if (entry.tree) dbg.keys.tree = key;
}

function dbgCommitPosition(data, key) {
  dbg.position = data;
  dbg.keys.position = key;
  const d = data.debug || {};
  if (Array.isArray(data.record_games) && data.record_games.length) dbg.records = data.record_games;
  if (!d.imported && Array.isArray(d.action_ids)) {
    // Harvest the full recorded action list — instant stepping, branch prefixes
    // and the recorded-move highlights all read it client-side.
    dbg.gameActs = d.action_ids;
    dbg.gameActsKey = `${dbg.nav.run}|${dbg.nav.path}|${dbg.nav.rec}`;
  }
  if (!dbg.nav.acts.length) {
    dbgCacheGamePlacements(data);
    // Position payloads carry placements for action_ids[:ply] only, so a deep
    // link to a mid-game ply leaves later stone ownership unknown — and the
    // HEADS recorded-reply highlight needs it. One-shot final-ply backfill.
    if (!d.imported && d.total != null && dbg.gamePlacements.length < d.total) dbgBackfillPlacements(d.total);
  }
}

function dbgBackfillPlacements(total) {
  const nav = dbg.nav;
  const gameKey = `${nav.run}|${nav.path}|${nav.rec}`;
  if (dbg.placementsBackfill === gameKey) return;
  dbg.placementsBackfill = gameKey;
  const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), ply: String(total) });
  debugFetchJson(`/api/debug/position?${params.toString()}`).then(data => {
    if (gameKey !== `${dbg.nav.run}|${dbg.nav.path}|${dbg.nav.rec}`) return;  // game changed mid-flight
    // Warm the M12 cache for the final-ply key while the payload is in hand.
    const entry = dbgCacheEntry(dbgCacheKey(Object.assign({}, dbg.nav, { ply: total, acts: [] }), dbg.nav.ckptA));
    if (!entry.position) entry.position = data;
    dbgCacheGamePlacements(data);
    if (dbg.nav.tab === "heads") dbgRenderHeads();
  }).catch(() => {
    if (dbg.placementsBackfill === gameKey) dbg.placementsBackfill = "";  // transient — retry on next commit
  });
}

function dbgCacheGamePlacements(data) {
  // Merge a loaded position's stones into the per-game cache (keyed by index), so
  // the full game's board is known client-side regardless of which ply loaded.
  // Branch (acts) positions are NOT merged — their indexes shift with the base ply.
  const key = `${dbg.nav.run}|${dbg.nav.path}|${dbg.nav.rec}`;
  if (key !== dbg.gameKey) {
    dbg.gameKey = key;
    dbg.gamePlacements = [];
  }
  const byIndex = new Map(dbg.gamePlacements.map(p => [p.index, p]));
  for (const p of (data.placements || [])) if (p && p.index != null) byIndex.set(p.index, p);
  dbg.gamePlacements = [...byIndex.values()].sort((a, b) => a.index - b.index);
}

function debugCurrentPlacements() {
  // Stones to show for the current ply. Prefer the cached full game filtered to
  // `index <= ply` (instant, no round-trip); branch positions use the payload.
  const pos = dbg.position;
  if (!pos) return [];
  if (dbg.nav.acts.length) return pos.placements || [];
  const ply = pos.debug.ply;
  if (dbg.gamePlacements.length) return dbg.gamePlacements.filter(p => p.index <= ply);
  return pos.placements || [];
}

function dbgOptimisticStep() {
  // Clone (never mutate the cached payload) with the new ply; the real position
  // payload + analysis land via the debounced fetch.
  const nav = dbg.nav;
  const pos = dbg.position;
  const acts = dbgRecordedActs() || (pos.debug && pos.debug.action_ids) || [];
  const ply = Math.max(0, Math.min(nav.ply, acts.length));
  const last = ply > 0 ? dbgUnpackActionId(acts[ply - 1]) : null;
  dbg.position = Object.assign({}, pos, {
    debug: Object.assign({}, pos.debug, {
      ply,
      last_action_id: ply > 0 ? acts[ply - 1] : null,
      last_q: last ? last.q : null,
      last_r: last ? last.r : null,
    }),
  });
  dbg.keys.position = "";  // optimistic — the real payload is still loading
  diagTap("ply " + ply + "/" + acts.length);  // visible step feedback on-device
}

// ---- fetch plumbing ----------------------------------------------------------

let dbgFetchTimer = null;

function dbgScheduleFetch(delay) {
  window.clearTimeout(dbgFetchTimer);
  dbgFetchTimer = window.setTimeout(() => {
    dbgFetchCurrent().catch(e => reportError("dbgFetchCurrent: " + (e && (e.stack || e.message) || e)));
  }, delay);
}

async function dbgFetchCurrent() {
  const nav = dbg.nav;
  if (!nav.run || !nav.path || nav.ply == null) return;
  const key = dbgCurrentKey();
  const entry = dbgCacheEntry(key);
  if (!entry.position && nav.ckptB) {
    // The sibling (ckptB) entry holds the same checkpoint-independent position —
    // after a ckptA↔ckptB flip reuse it instead of refetching (M12).
    const sib = dbg.cache.get(dbgCacheKey(nav, nav.ckptB));
    if (sib && sib.position) {
      entry.position = sib.position;
      if (!entry.record_row && sib.record_row) entry.record_row = sib.record_row;
    }
  }
  if (!entry.position) {
    const seq = ++dbg.posSeq;  // claim latest; a newer ply nav supersedes this fetch
    try {
      let data;
      if (nav.acts.length) {
        // What-if branch (M4): full prefix = recorded actions[0..ply] + injected
        // tail, reconstructed by the SERVER via ?actions= (true engine coords +
        // legal cells + tactics, no client action-id unpacking for the board).
        const recorded = await dbgEnsureRecordedActs();
        if (seq !== dbg.posSeq) return;
        const prefix = recorded.slice(0, nav.ply).concat(nav.acts);
        const params = new URLSearchParams({ run: nav.run, actions: prefix.join(","), ply: String(prefix.length) });
        data = await debugFetchJson(`/api/debug/position?${params.toString()}`);
      } else {
        const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), ply: String(nav.ply) });
        data = await debugFetchJson(`/api/debug/position?${params.toString()}`);
      }
      if (seq !== dbg.posSeq) return;  // superseded by a newer nav — drop stale fetch
      entry.position = data;
    } catch (e) {
      if (seq === dbg.posSeq) debugSetStatus(`Position: ${e.message}`, "error");
      return;
    }
  }
  if (key !== dbgCurrentKey()) return;  // nav moved on while we fetched
  dbgCommitEntry(key, entry);
  debugSetStatus("");
  dbgRenderAll();
  await dbgEnsureAnalysis(key, entry);
  if (dbg.nav.ckptB) await dbgEnsureAnalysisB();
  if (dbgWantRecordRow(dbg.nav)) dbgEnsureRecordRow(false);
  dbgMaybePrefetch();  // S7: speculative ply±1 once the user's own requests settled
}

async function dbgEnsureRecordedActs() {
  const nav = dbg.nav;
  const recorded = dbgRecordedActs();
  if (recorded) return recorded;
  // Deep link straight into a branch: one recorded-position fetch supplies the
  // full action-id list (and lands in the cache for the plain-ply key).
  const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), ply: String(Math.max(0, nav.ply || 0)) });
  const data = await debugFetchJson(`/api/debug/position?${params.toString()}`);
  const d = data.debug || {};
  dbg.gameActs = d.action_ids || [];
  dbg.gameActsKey = `${nav.run}|${nav.path}|${nav.rec}`;
  if (Array.isArray(data.record_games) && data.record_games.length) dbg.records = data.record_games;
  const entry = dbgCacheEntry([nav.run, nav.path, nav.rec, d.ply, nav.ckptA, ""].join("|"));
  if (!entry.position) entry.position = data;
  return dbg.gameActs;
}

async function dbgRequestBody(checkpoint) {
  const nav = dbg.nav;
  const body = { run: nav.run, checkpoint };
  if (nav.acts.length) {
    // Branch prefix = recorded actions[0..ply] + injected tail. The recorded
    // list may be missing (gameActsKey points at another game after the user
    // browsed elsewhere and the branch came back via a cache hit) — NEVER fall
    // back to an empty prefix: the server would silently analyze a near-empty
    // board while the UI shows the full branch position. Re-fetch instead.
    const recorded = await dbgEnsureRecordedActs();
    body.action_ids = recorded.slice(0, nav.ply).concat(nav.acts);
  } else {
    body.path = nav.path;
    body.record = nav.rec;
    body.ply = nav.ply;
  }
  return body;
}

async function dbgEnsureAnalysis(key, entry) {
  const nav = dbg.nav;
  if (!nav.ckptA) return;
  if (entry.analysis) {
    if (key === dbgCurrentKey()) {
      dbg.analysis = entry.analysis;
      dbg.keys.analysis = key;
      dbgRenderAll();
    }
    return;
  }
  if (entry.analysisPending) return;  // one in-flight analyze per cache entry
  entry.analysisPending = true;
  const seq = ++dbg.anlSeq;  // claim latest; a slow dense analyze must not clobber a newer ply's
  dbg.loading = true;
  debugSetStatus("Evaluating position on CPU…", "busy");
  try {
    const analysis = await debugFetchJson("/api/debug/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(await dbgRequestBody(nav.ckptA)),
    });
    // Always cache the response — a superseded seq only skips the status/loading
    // bookkeeping, never the data, or a revisited ply could sit permanently
    // un-analyzed (analysisPending is false again and nothing re-triggers).
    entry.analysis = analysis;
    dbgJournalLog(analysis, nav.ckptA, nav);  // S3: auto-log every completed analyze
    if (!dbg.worker || !dbg.worker.alive) {
      // A successful analyze proves the lazily-spawned worker is up — re-read the
      // live status so the dot turns green and the S7 prefetch gate opens.
      debugFetchJson(`/api/debug/checkpoints?run=${encodeURIComponent(nav.run)}`).then(d => {
        dbg.worker = d.worker || null;
        dbgRenderWorkerDot(dbg.worker && dbg.worker.alive ? "ok" : "");
      }).catch(() => {});
    }
    if (key === dbgCurrentKey()) {
      dbg.analysis = analysis;
      dbg.keys.analysis = key;
      if (seq === dbg.anlSeq) debugSetStatus("");
      dbgRenderAll();  // the finally's render is seq-gated — cover the dropped-seq case
    }
  } catch (e) {
    if (seq === dbg.anlSeq) debugSetStatus(`Analyze: ${e.message}`, "error");
  } finally {
    entry.analysisPending = false;
    if (seq === dbg.anlSeq) {
      dbg.loading = false;
      dbgRenderAll();
    }
  }
}

async function dbgEnsureAnalysisB() {
  const nav = dbg.nav;
  if (!nav.ckptB || nav.ply == null) return;
  const keyB = dbgCacheKey(nav, nav.ckptB);
  const entryB = dbgCacheEntry(keyB);
  // position/record_row are checkpoint-independent — share them with the A entry
  // so a later ckptA↔ckptB flip re-renders with zero network requests (M12).
  const entryA = dbg.cache.get(dbgCacheKey(nav, nav.ckptA));
  if (entryA) {
    if (!entryB.position && entryA.position) entryB.position = entryA.position;
    if (!entryB.record_row && entryA.record_row) entryB.record_row = entryA.record_row;
  }
  if (entryB.analysis) {
    dbg.analysisB = entryB.analysis;
    dbg.keys.analysisB = keyB;
    dbgRenderComparePanel();
    dbgUpdateStaleDots();
    return;
  }
  if (entryB.analysisPending) return;
  entryB.analysisPending = true;
  delete entryB.analysisError;  // a retry starts clean
  const seq = ++dbg.anlSeqB;
  debugSetStatus("Evaluating checkpoint B…", "busy");
  try {
    const analysis = await debugFetchJson("/api/debug/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(await dbgRequestBody(nav.ckptB)),
    });
    // Cache even when superseded (see dbgEnsureAnalysis) — only the commit is gated.
    entryB.analysis = analysis;
    dbgJournalLog(analysis, nav.ckptB, nav);
    if (keyB === dbgCacheKey(dbg.nav, dbg.nav.ckptB)) {
      dbg.analysisB = analysis;
      dbg.keys.analysisB = keyB;
      if (seq === dbg.anlSeqB) debugSetStatus("");
    }
  } catch (e) {
    // Record the failure on the entry so the COMPARE panel can show an error
    // state instead of a perpetual "Evaluating checkpoint B…" (nothing retries
    // automatically — only ↻ or the next nav apply).
    entryB.analysisError = e.message;
    if (seq === dbg.anlSeqB) debugSetStatus(`Compare: ${e.message}`, "error");
  } finally {
    entryB.analysisPending = false;
    dbgRenderComparePanel();
    dbgUpdateStaleDots();
  }
}

function dbgAnalyzeNow(force) {
  const nav = dbg.nav;
  if (!nav.path || nav.ply == null) {
    debugSetStatus("No position loaded — pick a game with completed records.", "error");
    return;
  }
  if (force) delete dbgCacheEntry(dbgCurrentKey()).analysis;
  dbgScheduleFetch(0);
}

async function dbgRunSearch() {
  dbgAbortPrefetch();  // S7: an explicit request preempts the speculative ply±1 fetch
  const nav = dbg.nav;
  if (!nav.ckptA || !dbg.position) {
    debugSetStatus("Pick a checkpoint and position before searching.", "error");
    return;
  }
  if (dbg.searchBusy) {
    dbgStub("A search is already running.");
    return;
  }
  const visitsEl = dbgEl("dbgSearchVisits");
  const cpuctEl = dbgEl("dbgSearchCpuct");
  const seedEl = dbgEl("dbgSearchSeed");
  const visits = Math.max(1, Math.min(20000, parseInt(visitsEl && visitsEl.value, 10) || 512));
  const key = dbgCurrentKey();
  const entry = dbgCacheEntry(key);
  dbg.searchBusy = true;
  // #dbgSearchRun is static HTML (not re-rendered from state) — toggle in place.
  const runBtn = dbgEl("dbgSearchRun");
  if (runBtn) runBtn.disabled = true;
  debugSetStatus(`Running ${visits}-visit CPU search…`, "busy");
  try {
    const body = await dbgRequestBody(nav.ckptA);
    body.visits = visits;
    const cPuct = Number(cpuctEl && cpuctEl.value);
    body.c_puct = Number.isFinite(cPuct) && cPuct > 0 ? cPuct : 1.5;
    const seed = parseInt(seedEl && seedEl.value, 10);
    body.seed = Number.isFinite(seed) ? seed : 0;  // B2/M6: the seed is forwarded server-side now
    const result = await debugFetchJson("/api/debug/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    entry.search = result;
    if (key === dbgCurrentKey()) {
      dbg.search = result;
      dbg.keys.search = key;
      debugSetStatus("");
    }
  } catch (e) {
    debugSetStatus(`Search: ${e.message}`, "error");
  } finally {
    dbg.searchBusy = false;
    if (runBtn) runBtn.disabled = false;
    dbgRenderAll();
  }
}

async function dbgEnsureCkptInfo(force) {
  // B3/M7: checkpoint provenance via the worker `info` op — one round-trip,
  // readable WITHOUT paying an analyze.
  const nav = dbg.nav;
  for (const name of [nav.ckptA, nav.ckptB].filter(Boolean)) {
    const key = `${nav.run}|${name}`;
    if (force) dbg.ckptInfo.delete(key);
    if (dbg.ckptInfo.has(key)) continue;
    dbgRenderCkptPanel();  // show "loading…" while the worker round-trips
    try {
      const params = new URLSearchParams({ run: nav.run, checkpoint: name });
      dbg.ckptInfo.set(key, await debugFetchJson(`/api/debug/ckpt_info?${params.toString()}`));
    } catch (e) {
      dbg.ckptInfo.set(key, { error: e.message });
    }
  }
  dbgRenderCkptPanel();
}

async function dbgPlotTrajectory() {
  // Legacy whole-game re-eval plot (kept as the no-sweep fallback per §3.4).
  const nav = dbg.nav;
  if (!nav.run || !nav.path || !nav.ckptA || nav.acts.length) {
    debugSetStatus("Trajectory needs a recorded game + checkpoint (return to the game first).", "error");
    return;
  }
  debugSetStatus("Re-evaluating the whole game on CPU…", "busy");
  try {
    const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), checkpoint: nav.ckptA });
    dbg.trajectory = await debugFetchJson(`/api/debug/trajectory?${params.toString()}`);
    dbg.trajectoryKey = `${nav.run}|${nav.path}|${nav.rec}|${nav.ckptA}`;
    debugSetStatus("");
  } catch (e) {
    dbg.trajectory = null;
    debugSetStatus(`Trajectory: ${e.message}`, "error");
  }
  dbgRenderDockChart();
}

// ---- TARGETS: recorded .npz training row (spec M8) ---------------------------------

function dbgWantRecordRow(nav) {
  // The row is only fetched when something actually displays it: the TARGETS
  // tab, the target/mismatch board modes (the HEADS soft-z marker reads it
  // opportunistically once cached, per spec).
  return Boolean(nav.run && nav.path && nav.ply != null
    && (nav.tab === "targets" || nav.mode === "target" || nav.mode === "mismatch"));
}

async function dbgEnsureRecordRow(force) {
  const nav = dbg.nav;
  if (!nav.run || !nav.path || nav.ply == null) return;
  const key = dbgCurrentKey();
  const entry = dbgCacheEntry(key);
  if (force) delete entry.record_row;
  if (!entry.record_row && nav.ckptB) {
    // record_row is checkpoint-independent — reuse the sibling (ckptB) entry's
    // copy after a ckptA↔ckptB flip instead of refetching (M12).
    const sib = dbg.cache.get(dbgCacheKey(nav, nav.ckptB));
    if (sib && sib.record_row) entry.record_row = sib.record_row;
  }
  if (nav.acts.length || nav.src !== "selfplay") {
    // Branch positions / eval games never have training rows — synthesize the
    // graceful miss client-side instead of paying a server round-trip.
    if (!entry.record_row) entry.record_row = { found: false, reason: nav.acts.length ? "branched" : "not_selfplay" };
    dbgRenderTargets();
    return;
  }
  if (entry.record_row || entry.recordRowPending) {
    dbgRenderTargets();
    return;
  }
  entry.recordRowPending = true;
  dbgRenderTargets();
  try {
    const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), ply: String(nav.ply) });
    entry.record_row = await debugFetchJson(`/api/debug/record_row?${params.toString()}`);
  } catch (e) {
    entry.record_row = { found: false, reason: e.message };
  } finally {
    entry.recordRowPending = false;
  }
  if (key === dbgCurrentKey()) {
    dbgRenderTargets();
    dbgRenderHeads();   // soft-z marker on the value distribution
    if (dbg.nav.mode === "target" || dbg.nav.mode === "mismatch") dbgRenderBoard();
  }
}

const DBG_ROW_MISS_REASONS = {
  not_selfplay: "eval games have no training rows",
  branched: "what-if branches have no training rows",
  no_shard: "no .npz training shard found for this game",
  no_row: "no training row at this ply (fast-PCR / not recorded)",
  row_mismatch: "row mismatch — not shown (turn/player disagree with the replay)",
};

function dbgRenderTargets() {
  const note = dbgEl("dbgTargetsNote");
  const body = document.querySelector("#dbgTabTargets .dbg-targets-body");
  if (!note || !body) return;
  const entry = dbg.cache.get(dbgCurrentKey());
  const rr = entry && entry.record_row;
  if (!rr) {
    note.textContent = entry && entry.recordRowPending ? "Loading training row…" : "No training row loaded";
    body.innerHTML = "";
    return;
  }
  if (!rr.found || !rr.row) {
    note.textContent = DBG_ROW_MISS_REASONS[rr.reason] || `no training row (${rr.reason || "unknown"})`;
    body.innerHTML = "";
    return;
  }
  const row = rr.row;
  const a = dbgFreshData("analysis");
  note.textContent = `${rr.npz || ""}${rr.npz ? " · " : ""}turn ${rr.turn_index} · P${row.current_player}${row.phase ? ` · ${row.phase}` : ""}`;
  const num = v => (typeof v === "number" ? v.toFixed(3) : "—");
  // The npz does not persist every field — the backend returns null for those,
  // and null must read as "not recorded", never as a plausible 0.000/yes/"".
  const NOT_REC = `<span class="dbg-muted">not recorded</span>`;
  const d = (recV, liveV) => (typeof recV === "number" && typeof liveV === "number")
    ? `${liveV - recV >= 0 ? "+" : ""}${(liveV - recV).toFixed(3)}` : "—";
  const cols = (label, rec, live, dlt) => `<div class="dbg-target-row"><span class="label">${label}</span><span>${rec}</span><span class="dbg-muted">${live}</span><span>${dlt}</span></div>`;
  const out = [`<div class="dbg-target-row dbg-move-head"><span>field</span><span>recorded</span><span>live ${escapeText(dbgCkptShort(dbg.nav.ckptA))}</span><span>Δ</span></div>`];
  out.push(cols("value target", num(row.value_target), num(a && a.value), d(row.value_target, a && a.value)));
  out.push(cols("reason", row.value_target_reason == null ? NOT_REC : escapeText(row.value_target_reason || "—"), "", ""));
  for (const h of Object.keys(row.stvalue || {}).sort((x, y) => Number(x) - Number(y))) {
    const sv = row.stvalue[h];
    const live = a && a.stvalue && a.stvalue[h] ? a.stvalue[h].scalar : null;
    out.push(cols(`STV+${h}${sv.mask ? "" : " (masked)"}`, sv.target == null ? NOT_REC : num(sv.target), num(live), sv.mask ? d(sv.target, live) : "—"));
  }
  if (row.moves_left) {
    const liveMl = a && a.moves_left && typeof a.moves_left.scalar === "number"
      ? (a.moves_left.scalar + 1) / 2 * ((a.meta && a.meta.moves_left_cap) || 512)
      : null;
    const recMl = row.moves_left.target;
    out.push(cols(`moves left${row.moves_left.mask ? "" : " (masked)"}`,
      typeof recMl === "number" && recMl >= 0 ? recMl.toFixed(0) : (recMl == null ? NOT_REC : "—"),
      liveMl != null ? liveMl.toFixed(0) : "—",
      row.moves_left.mask && typeof recMl === "number" && recMl >= 0 && liveMl != null
        ? `${liveMl - recMl >= 0 ? "+" : ""}${(liveMl - recMl).toFixed(0)}` : "—"));
  }
  out.push(cols("policy surprise", row.policy_surprise == null ? NOT_REC : num(row.policy_surprise), "", ""));
  // 0 is a sentinel: the shard stored a normalized visit policy, so the raw count is unknown.
  out.push(cols("search visits", row.search_visits == null ? NOT_REC
      : (row.search_visits > 0 ? String(row.search_visits) : "unknown (normalized weights)"), "", ""));
  out.push(cols("pcr_full", row.pcr_full == null ? NOT_REC : (row.pcr_full ? "yes" : "no (fast)"), "", ""));
  out.push(cols("frequency wt", row.frequency_weight == null ? NOT_REC : num(row.frequency_weight), "", ""));
  if (row.truncated) out.push(cols("truncated", "yes", "", ""));
  if (row.opp_policy_source) out.push(cols("opp source", escapeText(row.opp_policy_source), "", ""));
  const priorById = new Map(((a && a.policy) || []).map(p => [p.action_id, p.p]));
  const tm = (row.policy || []).slice(0, 8).map((p, i) => {
    const live = priorById.get(p.action_id);
    const dp = live != null ? `${live - p.p >= 0 ? "+" : ""}${((live - p.p) * 100).toFixed(1)}%` : "—";
    return `<div class="dbg-target-row"><span>#${i + 1} ${p.q},${p.r}</span><span>${(p.p * 100).toFixed(1)}%</span><span class="dbg-muted">${live != null ? (live * 100).toFixed(1) + "%" : "—"}</span><span>${dp}</span></div>`;
  }).join("");
  body.innerHTML = out.join("")
    + `<div class="dbg-subhead" style="margin-top:8px">Top target moves <span class="dbg-muted">(recorded · live prior · Δ)</span></div>` + tm;
}

// ---- Game Error Sweep (spec M9) -----------------------------------------------------

function dbgSweepKey(nav) {
  nav = nav || dbg.nav;
  return `${nav.run}|${nav.path}|${nav.rec}|${nav.ckptA}`;
}

function dbgSweepData() {
  const sweep = dbg.sweeps.get(dbgSweepKey());
  return sweep && sweep.plies.length ? sweep : null;
}

function dbgBlunders(sweep) {
  // Blunder = value_p0 sign flip or |Δ value_p0| > 0.5 between CONSECUTIVE swept
  // plies; returns the ply AFTER the drop (where to look), ascending.
  const out = [];
  for (let i = 1; i < sweep.plies.length; i++) {
    const a = sweep.plies[i - 1];
    const b = sweep.plies[i];
    if (b.ply !== a.ply + 1) continue;
    if ((a.value_p0 >= 0) !== (b.value_p0 >= 0) || Math.abs(b.value_p0 - a.value_p0) > 0.5) out.push(b.ply);
  }
  return out;
}

async function dbgRunSweep() {
  dbgAbortPrefetch();  // S7: an explicit request preempts the speculative ply±1 fetch
  const nav = dbg.nav;
  if (!nav.run || !nav.path || !nav.ckptA) {
    debugSetStatus("Sweep needs a recorded game + checkpoint.", "error");
    return;
  }
  const key = dbgSweepKey();
  if (dbg.sweepRun && dbg.sweepRun.key === key) {
    dbg.sweepRun.abort = true;  // the button toggles: click while running = stop
    return;
  }
  if (dbg.sweepRun) dbg.sweepRun.abort = true;  // a sweep of another tuple yields
  const runState = { key, abort: false };
  dbg.sweepRun = runState;
  let sweep = dbg.sweeps.get(key);
  if (!sweep) {
    sweep = { total: null, winner: null, plies: [], byPly: new Map(), version: 0 };
    dbg.sweeps.set(key, sweep);
  }
  const progress = dbgEl("dbgSweepProgress");
  const btns = [dbgEl("dbgSweepBtn"), dbgEl("dbgPlySweepBtn")].filter(Boolean);
  btns.forEach(b => b.classList.add("active"));
  try {
    // Resume cheaply (service LRU caches chunks; the client keeps finished plies).
    let start = 0;
    while (sweep.byPly.has(start)) start++;
    while (!runState.abort && (sweep.total == null || start < sweep.total)) {
      const params = new URLSearchParams({
        run: nav.run, path: nav.path, record: String(nav.rec), checkpoint: nav.ckptA,
        start: String(start), count: "16",
      });
      const data = await debugFetchJson(`/api/debug/game_eval?${params.toString()}`);
      sweep.total = data.total;
      sweep.winner = data.winner;
      for (const p of data.plies || []) if (!sweep.byPly.has(p.ply)) sweep.byPly.set(p.ply, p);
      sweep.plies = [...sweep.byPly.values()].sort((x, y) => x.ply - y.ply);
      sweep.version++;
      start = Math.max(start + 1, data.start + (data.plies || []).length);
      while (sweep.byPly.has(start)) start++;
      if (progress) progress.textContent = `${sweep.plies.length}/${sweep.total} plies`;
      if (key === dbgSweepKey()) {
        dbgRenderPlyRail();
        dbgRenderDockChart();
      }
    }
    if (progress) {
      progress.textContent = runState.abort
        ? `stopped at ${sweep.plies.length}/${sweep.total != null ? sweep.total : "?"}`
        : `${sweep.plies.length}/${sweep.total} plies`;
    }
  } catch (e) {
    debugSetStatus(`Sweep: ${e.message}`, "error");
  } finally {
    if (dbg.sweepRun === runState) dbg.sweepRun = null;
    btns.forEach(b => b.classList.remove("active"));
    if (key === dbgSweepKey()) {
      dbgRenderPlyRail();
      dbgRenderDockChart();
    }
  }
}

function dbgStepBlunder(dir) {
  const sweep = dbgSweepData();
  if (!sweep) {
    dbgStub("Run a game error sweep first (Sweep button).");
    return;
  }
  const blunders = dbgBlunders(sweep);
  if (!blunders.length) {
    dbgStub("No blunder plies found by the sweep.");
    return;
  }
  const cur = dbg.nav.ply != null ? dbg.nav.ply : 0;
  let next = null;
  if (dir > 0) next = blunders.find(p => p > cur);
  else for (const p of blunders) if (p < cur) next = p;
  if (next == null) {
    dbgStub(dir > 0 ? "No later blunder." : "No earlier blunder.");
    return;
  }
  dbgGotoPly(next);
}

// ---- MCTS Tree Explorer (spec M10/M11) ------------------------------------------

function dbgTreeParams() {
  const visitsEl = dbgEl("dbgSearchVisits");
  const cpuctEl = dbgEl("dbgSearchCpuct");
  const seedEl = dbgEl("dbgSearchSeed");
  const visits = Math.max(1, Math.min(20000, parseInt(visitsEl && visitsEl.value, 10) || 512));
  const cPuct = Number(cpuctEl && cpuctEl.value);
  const seed = parseInt(seedEl && seedEl.value, 10);
  return {
    visits,
    c_puct: Number.isFinite(cPuct) && cPuct > 0 ? cPuct : 1.5,
    seed: Number.isFinite(seed) ? seed : 0,
  };
}

async function dbgRunTree(opts) {
  // opts: {checkpoint} = run for that checkpoint's cache slot (compare B trees);
  // {rootActions, graftPath} = expand-on-demand — re-root the search at a deep
  // node via root_actions and graft the children back onto the rendered tree.
  dbgAbortPrefetch();  // S7: an explicit request preempts the speculative ply±1 fetch
  const nav = dbg.nav;
  const checkpoint = (opts && opts.checkpoint) || nav.ckptA;
  if (!checkpoint || !dbg.position) {
    debugSetStatus("Pick a checkpoint and position before searching.", "error");
    return;
  }
  if (dbg.treeBusy) {
    dbgStub("A debug tree is already running.");
    return;
  }
  const key = dbgCacheKey(nav, checkpoint);
  dbg.treeBusy = true;
  dbgRenderSearchPanel();
  try {
    const body = Object.assign(await dbgRequestBody(checkpoint), dbgTreeParams());
    if (opts && opts.rootActions && opts.rootActions.length) body.root_actions = opts.rootActions;
    debugSetStatus(`Debug tree: ${body.visits} visits on CPU…`, "busy");
    const result = await debugFetchJson("/api/debug/search_tree", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    debugSetStatus("");
    if (opts && opts.graftPath) {
      const host = dbgFreshData("tree");
      const node = host && dbgTreeNodeByPath(host.tree, opts.graftPath);
      if (node) {
        node.children = (result.tree && result.tree.children) || [];
        node.pruned_children = result.tree ? result.tree.pruned_children : 0;
        node.grafted = true;  // re-rooted deeper search — N not comparable to siblings
        dbg.treeOpen.set(opts.graftPath, true);
      }
    } else {
      const entry = dbgCacheEntry(key);
      entry.tree = result;
      if (key === dbgCurrentKey()) {
        dbg.tree = result;
        dbg.keys.tree = key;
        dbg.treeOpen = new Map();
        dbg.treePreview = null;
        dbg.treeGhostsOff = false;  // a fresh run always shows its ghosts
      }
    }
  } catch (e) {
    debugSetStatus(`Debug tree: ${e.message}`, "error");
  } finally {
    dbg.treeBusy = false;
  }
  dbgRenderAll();
}

function dbgTreeNodeByPath(root, pathStr) {
  let node = root;
  for (const part of String(pathStr).split("/")) {
    if (!part) continue;
    node = (node.children || []).find(ch => String(ch.action_id) === part) || null;
    if (!node) return null;
  }
  return node;
}

function dbgTreePathIds(pathStr) {
  return String(pathStr).split("/").map(s => parseInt(s, 10)).filter(n => Number.isFinite(n));
}

function dbgRenderTree() {
  const el = dbgEl("dbgTree");
  if (!el) return;
  const tree = dbgFreshData("tree");
  if (!tree || !tree.tree) {
    el.innerHTML = `<div class="dbg-empty-note">${dbg.treeBusy ? "Debug tree running…" : "Debug tree (Python; may not match engine tie-breaking)"}</div>`;
    return;
  }
  const pvSet = new Set();
  let acc = "";
  for (const id of tree.pv || []) {
    acc = acc ? `${acc}/${id}` : String(id);
    pvSet.add(acc);
  }
  const render = (node, pathStr, maxNSib) => {
    const kids = node.children || [];
    const isPv = pvSet.has(pathStr);
    const open = dbg.treeOpen.has(pathStr) ? dbg.treeOpen.get(pathStr) : isPv;
    const caret = kids.length ? (open ? "▾" : "▸") : "·";
    const expand = node.pruned_children > 0
      ? `<button type="button" class="dbg-tree-expand" data-tree-path="${pathStr}" title="Load pruned children (re-searches this subtree as a new root)">+${node.pruned_children}</button>`
      : "";
    let html = `<div class="dbg-tree-row${isPv ? " pv" : ""}" data-tree-path="${pathStr}" title="N ${node.n} · Q stm ${node.qm.toFixed(3)} · Q p0 ${node.qm_p0.toFixed(3)} · P ${(node.p * 100).toFixed(2)}%${node.u != null ? ` · U ${node.u.toFixed(3)}` : ""}${node.v != null ? ` · v@expand ${node.v.toFixed(3)}` : ""}${node.grafted ? " · grafted (re-rooted search)" : ""}">`
      + `<span class="dbg-tree-caret">${caret}</span>`
      + `<span class="dbg-tree-cell">${node.q},${node.r}</span>`
      + `<span class="dbg-tree-bar"><span style="width:${Math.max(3, 100 * node.n / Math.max(1, maxNSib)).toFixed(0)}%"></span></span>`
      + `<span class="dbg-tree-n">${node.n}</span>`
      + `<span class="dbg-tree-q">${node.qm.toFixed(2)}/${node.qm_p0.toFixed(2)}</span>`
      + `<span class="dbg-tree-p">${(node.p * 100).toFixed(1)}%</span>`
      + `<span class="dbg-tree-u">${node.u != null ? node.u.toFixed(2) : "—"}</span>`
      + `<span class="dbg-tree-v">${node.v != null ? node.v.toFixed(2) : "—"}</span>`
      + `<button type="button" class="dbg-tree-step" data-tree-path="${pathStr}" title="Step into this line (re-bases via what-if injection)">⤓</button>`
      + expand
      + `</div>`;
    if (open && kids.length) {
      const m = Math.max(1, ...kids.map(k => k.n));
      html += `<div class="dbg-tree-node">${kids.map(k => render(k, `${pathStr}/${k.action_id}`, m)).join("")}</div>`;
    }
    return html;
  };
  const roots = tree.tree.children || [];
  const maxN = Math.max(1, ...roots.map(ch => ch.n));
  const pvCells = (tree.pv || []).map(id => {
    const c = dbgUnpackActionId(id);
    return `${c.q},${c.r}`;
  });
  const head = `<div class="dbg-muted" title="debug search (Python; may not match engine tie-breaking)">`
    + `${escapeText(tree.engine || "py_debug")} · ${tree.visits} visits · root ${tree.root_value.toFixed(3)} · ${tree.node_count} nodes${tree.truncated ? " · truncated" : ""}</div>`
    + `<div class="dbg-muted">PV: ${pvCells.join(" → ") || "—"}</div>`
    + `<div class="dbg-tree-cols dbg-move-head"><span></span><span>cell</span><span>N</span><span></span><span>Q stm/p0</span><span>P</span><span>U</span><span>v</span></div>`;
  el.innerHTML = head + roots.map(ch => render(ch, String(ch.action_id), maxN)).join("");
}

function dbgTreeRowClick(pathStr) {
  // Node click = expand/collapse + preview its line on the board (M10).
  const tree = dbgFreshData("tree");
  const node = tree && dbgTreeNodeByPath(tree.tree, pathStr);
  if (!node) return;
  if ((node.children || []).length) {
    const pvSet = new Set();
    let acc = "";
    for (const id of tree.pv || []) {
      acc = acc ? `${acc}/${id}` : String(id);
      pvSet.add(acc);
    }
    const open = dbg.treeOpen.has(pathStr) ? dbg.treeOpen.get(pathStr) : pvSet.has(pathStr);
    dbg.treeOpen.set(pathStr, !open);
  }
  dbg.treePreview = { key: dbgCurrentKey(), ids: dbgTreePathIds(pathStr) };
  dbgRenderTree();
  dbgRenderBoard();
}

function dbgRenderScatter() {
  const el = dbgEl("dbgScatter");
  if (!el) return;
  const tree = dbgFreshData("tree");
  const kids = tree && tree.tree ? (tree.tree.children || []) : [];
  if (!kids.length) {
    el.innerHTML = `<div class="dbg-empty-note">Q vs prior scatter after a tree run</div>`;
    return;
  }
  const W = 340, H = 190, padL = 34, padR = 10, padT = 12, padB = 20;
  const ps = kids.map(k => Math.max(k.p, 1e-4));
  const lo = Math.log(Math.min(...ps));
  const hi = Math.log(Math.max(Math.max(...ps), Math.min(...ps) * 1.0001));
  const xOf = p => padL + (W - padL - padR) * ((Math.log(Math.max(p, 1e-4)) - lo) / Math.max(1e-9, hi - lo));
  const yOf = q => padT + (H - padT - padB) * (1 - (q + 1) / 2);
  const maxN = Math.max(1, ...kids.map(k => k.n));
  const rootQ = tree.tree.qm;
  const recorded = (!dbg.nav.acts.length && dbg.gameActs.length > dbg.nav.ply) ? dbg.gameActs[dbg.nav.ply] : null;
  const midX = padL + (W - padL - padR) / 2;
  let html = `<line x1="${padL}" y1="${yOf(rootQ).toFixed(1)}" x2="${W - padR}" y2="${yOf(rootQ).toFixed(1)}" stroke="#2c3d50" stroke-dasharray="3 3"></line>`
    + `<line x1="${midX.toFixed(1)}" y1="${padT}" x2="${midX.toFixed(1)}" y2="${H - padB}" stroke="#2c3d50" stroke-dasharray="3 3"></line>`
    + `<text x="${padL + 2}" y="${padT + 9}" fill="#5a6b7a" font-size="9">under-read</text>`
    + `<text x="${W - padR - 2}" y="${padT + 9}" fill="#5a6b7a" font-size="9" text-anchor="end">trusted</text>`
    + `<text x="${padL + 2}" y="${H - padB - 3}" fill="#5a6b7a" font-size="9">ignored</text>`
    + `<text x="${W - padR - 2}" y="${H - padB - 3}" fill="#5a6b7a" font-size="9" text-anchor="end">over-read</text>`
    + `<text x="4" y="${yOf(1) + 8}" fill="#5a6b7a" font-size="10">Q+1</text>`
    + `<text x="4" y="${yOf(-1)}" fill="#5a6b7a" font-size="10">−1</text>`
    + `<text x="${W / 2}" y="${H - 4}" fill="#5a6b7a" font-size="10" text-anchor="middle">prior (log)</text>`;
  for (const k of kids) {
    const r2 = 2 + 6 * Math.sqrt(k.n / maxN);
    const isBest = tree.best_action_id === k.action_id;
    const isRec = recorded != null && recorded === k.action_id;
    html += `<circle cx="${xOf(k.p).toFixed(1)}" cy="${yOf(k.qm).toFixed(1)}" r="${r2.toFixed(1)}" fill="${isBest ? "var(--green)" : "var(--accent)"}" opacity="0.75"${isRec ? ` stroke="var(--yellow)" stroke-width="2"` : ""}><title>${k.q},${k.r} · N ${k.n} · Q ${k.qm.toFixed(3)} · P ${(k.p * 100).toFixed(2)}%${isBest ? " · best" : ""}${isRec ? " · recorded move" : ""}</title></circle>`;
    if (isBest || isRec) {
      html += `<text x="${(xOf(k.p) + r2 + 2).toFixed(1)}" y="${(yOf(k.qm) + 3).toFixed(1)}" fill="${isBest ? "var(--green)" : "var(--yellow)"}" font-size="9">${k.q},${k.r}</text>`;
    }
  }
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}">${html}</svg>`;
}

// ---- visit ladder (spec S10) ---------------------------------------------------

async function dbgRunLadder() {
  dbgAbortPrefetch();  // S7: an explicit request preempts the speculative ply±1 fetch
  const nav = dbg.nav;
  if (!nav.ckptA || !dbg.position) {
    debugSetStatus("Pick a checkpoint and position first.", "error");
    return;
  }
  if (dbg.ladderBusy) return;
  const key = dbgCurrentKey();
  dbg.ladderBusy = true;
  dbg.ladder = { key, rungs: [] };
  dbgRenderLadder();
  const params = dbgTreeParams();
  try {
    for (const visits of [64, 128, 256, 512, 1024]) {
      if (dbgCurrentKey() !== key) break;  // user moved on — stop climbing
      debugSetStatus(`Visit ladder: ${visits} visits…`, "busy");
      const body = Object.assign(await dbgRequestBody(nav.ckptA), { visits, c_puct: params.c_puct, seed: params.seed });
      const s = await debugFetchJson("/api/debug/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      dbg.ladder.rungs.push({ visits, best: s.best, best_action_id: s.best_action_id, root_value: s.root_value });
      dbgRenderLadder();
    }
    // Only clear the shared status line while we still own the context — a nav
    // mid-ladder hands it to the newer action's busy message.
    if (dbgCurrentKey() === key) debugSetStatus("");
  } catch (e) {
    debugSetStatus(`Ladder: ${e.message}`, "error");
  } finally {
    dbg.ladderBusy = false;
    dbgRenderLadder();
  }
}

function dbgRenderLadder() {
  const el = dbgEl("dbgLadder");
  if (!el) return;
  const l = dbg.ladder && dbg.ladder.key === dbgCurrentKey() ? dbg.ladder : null;
  const btn = `<button type="button" id="dbgLadderRun" class="dbg-mini-btn"${dbg.ladderBusy ? " disabled" : ""}>${dbg.ladderBusy ? "Running…" : (l && l.rungs.length ? "Re-run" : "Run")}</button>`;
  if (!l || !l.rungs.length) {
    el.innerHTML = `<div class="dbg-flex-head"><span class="dbg-muted">Visit ladder (64 → 1024): best move per budget</span>${btn}</div>`;
    return;
  }
  // Flag EVERY rung where the best move flipped, not just the last one — the
  // early prior-vs-search flips are usually the diagnostic ones.
  const flips = new Set();
  for (let i = 1; i < l.rungs.length; i++) {
    if (l.rungs[i].best_action_id !== l.rungs[i - 1].best_action_id) flips.add(l.rungs[i].visits);
  }
  const rows = l.rungs.map(r => `<div class="dbg-move-row${flips.has(r.visits) ? " dbg-move-best" : ""}"><span></span><span>${r.visits}v</span><span>${r.best ? `${r.best.q},${r.best.r}` : "—"}</span><span>${r.root_value.toFixed(3)}</span><span>${flips.has(r.visits) ? "flip" : ""}</span><span></span></div>`).join("");
  el.innerHTML = `<div class="dbg-flex-head"><span class="dbg-muted">Visit ladder${flips.size ? ` · best flips at ${[...flips].join("/")}v` : " · stable best"}</span>${btn}</div>`
    + `<div class="dbg-move-row dbg-move-head"><span></span><span>visits</span><span>best</span><span>root v</span><span></span><span></span></div>` + rows;
}

// ---- pin tray (spec S2) ----------------------------------------------------------

function dbgPinsStorageKey() {
  return `hexoDbgPins:${dbg.nav.run}`;
}

function dbgLoadPins() {
  if (!dbg.nav.run || dbg.pinsRun === dbg.nav.run) return;
  dbg.pinsRun = dbg.nav.run;
  try {
    const raw = JSON.parse(window.localStorage.getItem(dbgPinsStorageKey()) || "[]");
    dbg.pins = Array.isArray(raw) ? raw : [];
  } catch (_e) {
    dbg.pins = [];
  }
  dbgRenderPinTray();
}

function dbgSavePins() {
  try {
    window.localStorage.setItem(dbgPinsStorageKey(), JSON.stringify(dbg.pins));
  } catch (e) {
    debugSetStatus(`Pins: ${e.message}`, "error");
  }
  dbgRenderPinTray();
}

function dbgCapPins() {
  // Soft cap 24 — drop oldest WITH a warning, never silently (spec §1.6).
  let dropped = 0;
  while (dbg.pins.length > 24) {
    dbg.pins.shift();
    dropped++;
  }
  if (dropped) debugSetStatus(`Pin tray over 24 — ${dropped} oldest pin${dropped > 1 ? "s" : ""} dropped.`, "info");
}

function dbgAddPin() {
  const nav = dbg.nav;
  if (!nav.run || !nav.path || nav.ply == null) {
    debugSetStatus("No position to pin.", "error");
    return;
  }
  dbgLoadPins();
  const a = dbgFreshData("analysis");
  const s = dbgFreshData("search");
  const best = s && s.best
    ? `${s.best.q},${s.best.r}`
    : (a && a.policy && a.policy[0] ? `${a.policy[0].q},${a.policy[0].r}` : null);
  dbg.pins.push({
    ts: Date.now(),
    note: "",
    nav: Object.assign({}, nav, { acts: nav.acts.slice() }),
    snap: {
      value: a ? a.value : null,
      best,
      ckpt: dbgCkptShort(nav.ckptA),
      thumb: debugCurrentPlacements().map(p => ({ q: p.q, r: p.r, player: p.player })),
    },
  });
  dbgCapPins();
  dbgSavePins();
  diagTap("pinned");
}

function dbgPinThumbSvg(thumb) {
  if (!thumb || !thumb.length) return `<svg viewBox="0 0 80 80"></svg>`;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const pts = thumb.map(p => ({ player: p.player, c: center(p.q, p.r) }));
  for (const p of pts) {
    minX = Math.min(minX, p.c.x - HEX);
    maxX = Math.max(maxX, p.c.x + HEX);
    minY = Math.min(minY, p.c.y - HEX);
    maxY = Math.max(maxY, p.c.y + HEX);
  }
  const stones = pts.map(p => `<circle cx="${p.c.x.toFixed(1)}" cy="${p.c.y.toFixed(1)}" r="${(HEX * 0.62).toFixed(1)}" fill="${playerColor(p.player)}"></circle>`).join("");
  return `<svg viewBox="${minX.toFixed(1)} ${minY.toFixed(1)} ${(maxX - minX).toFixed(1)} ${(maxY - minY).toFixed(1)}" preserveAspectRatio="xMidYMid meet">${stones}</svg>`;
}

function dbgRenderPinTray() {
  const chips = document.querySelector("#dbgPinTray .dbg-pin-chips");
  if (!chips) return;
  if (!dbg.pins.length) {
    chips.innerHTML = `<span class="dbg-empty-note">No pinned positions</span>`;
    return;
  }
  chips.innerHTML = dbg.pins.map((pin, i) => {
    const v = pin.snap && pin.snap.value != null ? pin.snap.value.toFixed(2) : "—";
    const branch = pin.nav && pin.nav.acts && pin.nav.acts.length ? `+${pin.nav.acts.length}` : "";
    const label = `ply ${pin.nav ? pin.nav.ply : "?"}${branch} · ${v} · ${(pin.snap && pin.snap.ckpt) || ""}`;
    const tip = pin.note || (pin.snap && pin.snap.best ? `best ${pin.snap.best}` : "pinned position");
    return `<div class="dbg-pin-chip" data-pin="${i}" title="${escapeAttr(tip)}">`
      + dbgPinThumbSvg(pin.snap && pin.snap.thumb)
      + `<span class="dbg-muted">${escapeText(label)}</span>`
      + `<span class="dbg-pin-chip-actions"><button type="button" class="dbg-pin-note dbg-mini-btn" data-pin="${i}" title="Edit note">✎</button><button type="button" class="dbg-pin-x dbg-mini-btn" data-pin="${i}" title="Remove pin">✕</button></span>`
      + `</div>`;
  }).join("");
}

function dbgRestorePin(i) {
  const pin = dbg.pins[i];
  if (!pin || !pin.nav) return;
  diagTap("pin restore");
  dbgNavigate(Object.assign({}, pin.nav, { acts: (pin.nav.acts || []).slice() }));
}

function dbgNotePin(i) {
  dbgLoadPins();
  const pin = dbg.pins[i];
  if (!pin) return;
  const note = window.prompt("Pin note:", pin.note || "");
  if (note == null) return;
  pin.note = note;
  dbgSavePins();
}

function dbgRemovePin(i) {
  dbgLoadPins();
  dbg.pins.splice(i, 1);
  dbgSavePins();
}

function dbgExportPins() {
  dbgLoadPins();
  if (!dbg.pins.length) {
    dbgStub("No pins to export.");
    return;
  }
  const blob = new Blob([JSON.stringify(dbg.pins, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `hexo-debug-pins-${dbg.nav.run || "run"}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function dbgImportPins(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(String(reader.result || "[]"));
      if (!Array.isArray(data)) throw new Error("not a pin array");
      dbgLoadPins();
      dbg.pins = dbg.pins.concat(data.filter(p => p && typeof p === "object"));
      dbgCapPins();
      dbgSavePins();
      debugSetStatus(`Imported ${data.length} pin${data.length === 1 ? "" : "s"}.`, "info");
    } catch (e) {
      debugSetStatus(`Pin import: ${e.message}`, "error");
    }
  };
  reader.readAsText(file);
}

// ---- session journal (spec S3) ------------------------------------------------------

function dbgJournalStorageKey() {
  return `hexoDbgJournal:${dbg.nav.run}`;
}

function dbgLoadJournal() {
  if (!dbg.nav.run || dbg.journalRun === dbg.nav.run) return;
  dbg.journalRun = dbg.nav.run;
  try {
    const raw = JSON.parse(window.localStorage.getItem(dbgJournalStorageKey()) || "[]");
    dbg.journal = Array.isArray(raw) ? raw : [];
  } catch (_e) {
    dbg.journal = [];
  }
  dbgRenderJournal();
}

function dbgJournalLog(analysis, ckpt, nav) {
  // Auto-log every completed analyze: {ts, ckpt, ply/branch, value, top move}.
  if (!nav.run) return;
  dbgLoadJournal();
  const top = analysis.policy && analysis.policy[0];
  dbg.journal.push({
    ts: Date.now(),
    ckpt: dbgCkptShort(ckpt),
    ply: nav.ply,
    branch: nav.acts.length,
    value: analysis.value,
    top: top ? `${top.q},${top.r}` : null,
    nav: Object.assign({}, nav, { acts: nav.acts.slice() }),
  });
  while (dbg.journal.length > 200) dbg.journal.shift();  // FIFO cap (spec §1.6)
  try {
    window.localStorage.setItem(dbgJournalStorageKey(), JSON.stringify(dbg.journal));
  } catch (_e) { /* storage full — journal degrades to in-memory */ }
  dbgRenderJournal();
}

function dbgRenderJournal() {
  const list = document.querySelector("#dbgJournal .dbg-journal-list");
  if (!list) return;
  if (!dbg.journal.length) {
    list.className = "dbg-journal-list dbg-empty-note";
    list.textContent = "No analyses logged yet";
    return;
  }
  list.className = "dbg-journal-list";
  list.innerHTML = dbg.journal.slice(-40).reverse().map((e, idx) => {
    const i = dbg.journal.length - 1 - idx;
    const t = new Date(e.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return `<div class="dbg-palette-item" data-journal="${i}" title="Click to restore this position">${t} · ${escapeText(e.ckpt || "")} · ply ${e.ply}${e.branch ? `+${e.branch}` : ""} · v ${e.value != null ? e.value.toFixed(2) : "—"}${e.top ? ` · ${e.top}` : ""}</div>`;
  }).join("");
}

function dbgClearJournal() {
  dbg.journal = [];
  try {
    window.localStorage.removeItem(dbgJournalStorageKey());
  } catch (_e) { /* ignore */ }
  dbgRenderJournal();
}

// ---- command palette (spec S1) -------------------------------------------------------

function dbgOpenPalette() {
  const pal = dbgEl("dbgPalette");
  const input = dbgEl("dbgPaletteInput");
  if (!pal || !input) return;
  dbg.paletteItems = dbgPaletteAllItems();
  dbg.paletteIdx = 0;
  pal.hidden = false;
  input.value = "";
  dbgRenderPaletteList("");
  dbgEnsurePaletteGames();
  window.setTimeout(() => input.focus(), 0);
}

async function dbgEnsurePaletteGames() {
  // The palette searches BOTH sources (typing "eval" must surface evaluation
  // files even while src=selfplay) — one lazy all-source listing per run.
  if (!dbg.nav.run || dbg.paletteGamesRun === dbg.nav.run) return;
  try {
    const data = await debugFetchJson(`/api/debug/games?run=${encodeURIComponent(dbg.nav.run)}&source=all`);
    dbg.paletteGames = data.games || [];
    dbg.paletteGamesRun = dbg.nav.run;
    dbg.paletteItems = dbgPaletteAllItems();
    const input = dbgEl("dbgPaletteInput");
    const pal = dbgEl("dbgPalette");
    if (pal && !pal.hidden) dbgRenderPaletteList(input ? input.value : "");
  } catch (_e) { /* palette degrades to the current-source game list */ }
}

function dbgPaletteAllItems() {
  const items = [
    { label: "Action: Analyze now (A)", run: () => dbgAnalyzeNow(true) },
    { label: "Action: Run root search (S)", run: () => dbgRunSearch() },
    { label: "Action: Run debug tree (T)", run: () => dbgRunTree() },
    { label: "Action: Sweep game", run: () => dbgRunSweep() },
    { label: "Action: Return to game (G)", run: () => dbgNavigate({ acts: [] }) },
    { label: "Action: Pin current position (P)", run: () => dbgAddPin() },
    { label: "Action: Export pins", run: () => dbgExportPins() },
    { label: "Action: Run visit ladder", run: () => dbgRunLadder() },
    { label: "Action: Checkpoint sweep", run: () => dbgRunCkptSweep() },
  ];
  const games = dbg.paletteGamesRun === dbg.nav.run && dbg.paletteGames.length ? dbg.paletteGames : dbg.games;
  for (const g of games) {
    items.push({
      label: `Game file: ${g.path}`,
      run: () => dbgNavigate({
        src: String(g.path).startsWith("eval") ? "evaluation" : "selfplay",
        path: g.path, rec: 0, ply: null, acts: [],
      }),
    });
  }
  for (const r of dbg.records) {
    items.push({ label: `Record: ${dbgRecordLabel(r)}`, run: () => dbgNavigate({ rec: r.index, ply: null, acts: [] }) });
  }
  for (const c of dbg.checkpoints) {
    items.push({ label: `Checkpoint: ${dbgCkptLabel(c)} (${c.name})`, run: () => dbgNavigate({ ckptA: c.name }) });
  }
  dbg.pins.forEach((pin, i) => {
    items.push({
      label: `Pin: ply ${pin.nav ? pin.nav.ply : "?"} · ${pin.note || (pin.snap && pin.snap.ckpt) || ""}`,
      run: () => dbgRestorePin(i),
    });
  });
  for (const e of dbg.journal.slice(-20)) {
    items.push({
      label: `Journal: ply ${e.ply}${e.branch ? `+${e.branch}` : ""} · ${e.ckpt} · v ${e.value != null ? e.value.toFixed(2) : "—"}`,
      run: () => dbgNavigate(Object.assign({}, e.nav, { acts: ((e.nav && e.nav.acts) || []).slice() })),
    });
  }
  return items;
}

function dbgPaletteFilter(qstr) {
  const q = qstr.trim().toLowerCase();
  const items = [];
  const plyNum = /^(?:ply\s*)?(\d+)$/.exec(q);
  if (plyNum) items.push({ label: `Go to ply ${plyNum[1]}`, run: () => dbgGotoPly(Number(plyNum[1])) });
  if (!q) return items.concat(dbg.paletteItems.slice(0, 12));
  const scored = [];
  for (const it of dbg.paletteItems) {
    const l = it.label.toLowerCase();
    let score = null;
    const sub = l.indexOf(q);
    if (sub !== -1) score = 1000 - sub;       // substring beats subsequence
    else {
      let i = 0;
      for (const ch of l) if (ch === q[i]) i++;
      if (i >= q.length) score = 100 - l.length * 0.1;  // fuzzy subsequence
    }
    if (score != null) scored.push([score, it]);
  }
  scored.sort((x, y) => y[0] - x[0]);
  return items.concat(scored.slice(0, 14).map(s => s[1]));
}

function dbgRenderPaletteList(qstr) {
  const list = document.querySelector("#dbgPalette .dbg-palette-list");
  if (!list) return;
  dbg.paletteShown = dbgPaletteFilter(qstr);
  dbg.paletteIdx = Math.max(0, Math.min(dbg.paletteIdx, dbg.paletteShown.length - 1));
  list.innerHTML = dbg.paletteShown.length
    ? dbg.paletteShown.map((it, i) => `<div class="dbg-palette-item${i === dbg.paletteIdx ? " active" : ""}" data-pal="${i}">${escapeText(it.label)}</div>`).join("")
    : `<div class="dbg-empty-note">No matches</div>`;
}

function dbgPaletteExec(i) {
  const it = dbg.paletteShown && dbg.paletteShown[i];
  const pal = dbgEl("dbgPalette");
  if (pal) pal.hidden = true;
  if (!it) return;
  try {
    it.run();
  } catch (e) {
    reportError("palette: " + (e && (e.stack || e.message) || e));
  }
}

// ---- checkpoint sweep dock tab (spec S4) ----------------------------------------------

async function dbgRunCkptSweep() {
  dbgAbortPrefetch();  // S7: an explicit request preempts the speculative ply±1 fetch
  const nav0 = Object.assign({}, dbg.nav, { acts: dbg.nav.acts.slice() });
  if (!nav0.run || !nav0.path || nav0.ply == null) {
    debugSetStatus("No position for the checkpoint sweep.", "error");
    return;
  }
  if (!dbg.checkpoints.length) {
    debugSetStatus("No checkpoints in this run.", "error");
    return;
  }
  if (dbg.ckptSweep && dbg.ckptSweep.running) {
    dbgStub("Checkpoint sweep already running — Stop aborts it.");
    return;
  }
  // navKey ties the finished curve to the position it swept (ckpt slot empty —
  // the sweep spans checkpoints); the renderer flags it stale after a nav away.
  const sweep = { navKey: dbgCacheKey(nav0, ""), rows: [], running: true, abort: false, refTop: null, refTopCell: "" };
  dbg.ckptSweep = sweep;
  const aCur = dbgFreshData("analysis");
  if (aCur && aCur.policy && aCur.policy[0]) {
    sweep.refTop = aCur.policy[0].action_id;
    sweep.refTopCell = `${aCur.policy[0].q},${aCur.policy[0].r}`;
  }
  const list = dbg.checkpoints.slice()
    .sort((x, y) => (x.epoch != null ? x.epoch : 1e12) - (y.epoch != null ? y.epoch : 1e12));
  const sameNav = () => dbg.nav.run === nav0.run && dbg.nav.path === nav0.path
    && dbg.nav.rec === nav0.rec && dbg.nav.ply === nav0.ply
    && dbg.nav.acts.join(",") === nav0.acts.join(",");
  dbgRenderCkptSweep();
  try {
    // Branch prefix needs the recorded action list — await it (never fall back
    // to an empty prefix, which would sweep the wrong, near-empty position).
    const bodyBase = nav0.acts.length
      ? { run: nav0.run, action_ids: (await dbgEnsureRecordedActs()).slice(0, nav0.ply).concat(nav0.acts) }
      : { run: nav0.run, path: nav0.path, record: nav0.rec, ply: nav0.ply };
    for (const ck of list) {
      if (sweep.abort || !sameNav()) break;
      const key = dbgCacheKey(nav0, ck.name);
      const entry = dbgCacheEntry(key);
      let analysis = entry.analysis;
      if (!analysis) {
        debugSetStatus(`Checkpoint sweep: ${dbgCkptLabel(ck)}…`, "busy");
        analysis = await debugFetchJson("/api/debug/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(Object.assign({ checkpoint: ck.name }, bodyBase)),
        });
        entry.analysis = analysis;  // lands in the M12 cache — flipping ckptA later is free
      }
      const top = analysis.policy && analysis.policy[0];
      sweep.rows.push({
        name: ck.name,
        label: ck.epoch != null ? `e${ck.epoch}` : ck.name.replace(/\.pt$/, ""),
        value: analysis.value,
        top: top ? top.action_id : null,
        topCell: top ? `${top.q},${top.r}` : "",
      });
      dbgRenderCkptSweep();
    }
    if (!sweep.refTop) {
      const own = sweep.rows.find(r => r.name === nav0.ckptA);
      if (own) {
        sweep.refTop = own.top;
        sweep.refTopCell = own.topCell;
      }
    }
    // Clear the shared status line only while we still own the context — a nav
    // mid-sweep hands it to the newer action's busy message.
    if (sweep.abort) debugSetStatus("Checkpoint sweep stopped.");
    else if (sameNav()) debugSetStatus("");
  } catch (e) {
    debugSetStatus(`Checkpoint sweep: ${e.message}`, "error");
  } finally {
    sweep.running = false;
    dbgRenderCkptSweep();
  }
}

function dbgRenderCkptSweep() {
  const el = document.querySelector("#dbgCkptSweep .dbg-ckpt-sweep-chart");
  if (!el) return;
  const cs = dbg.ckptSweep;
  // The curve belongs to the position it swept — once the user navigates away
  // it must say so instead of posing as the current position's history.
  const stale = Boolean(cs && cs.rows.length && cs.navKey !== dbgCacheKey(dbg.nav, ""));
  // Called from dbgRenderAll on every render (the Run button only exists in this
  // markup) — skip the innerHTML swap when nothing observable changed.
  const sig = cs ? `${cs.rows.length}|${cs.running ? 1 : 0}|${cs.refTop}|${cs.refTopCell}|${stale ? 1 : 0}` : "none";
  if (el.__dbgSig === sig) return;
  el.__dbgSig = sig;
  const btn = `<button type="button" id="dbgCkptSweepRun" class="dbg-mini-btn"${cs && cs.running ? " disabled" : ""}>${cs && cs.running ? "Running…" : (cs && cs.rows.length ? "Re-run on current position" : "Run on current position")}</button>`;
  if (!cs || !cs.rows.length) {
    el.innerHTML = `<div class="dbg-flex-head"><span class="dbg-empty-note">${cs && cs.running ? "Evaluating checkpoints…" : "Not run"}</span>${btn}</div>`;
    return;
  }
  const rows = cs.rows;
  const W = 1000, H = 200, padL = 36, padR = 12, padT = 12, padB = 26;
  const xs = i => padL + (W - padL - padR) * (rows.length > 1 ? i / (rows.length - 1) : 0.5);
  const y = v => padT + (H - padT - padB) * (1 - (v + 1) / 2);
  let html = `<line x1="${padL}" y1="${y(0)}" x2="${W - padR}" y2="${y(0)}" stroke="#2c3d50"></line>`
    + `<text x="4" y="${y(1) + 4}" fill="#5a6b7a" font-size="11">+1</text>`
    + `<text x="4" y="${y(-1) + 2}" fill="#5a6b7a" font-size="11">−1</text>`
    + `<path d="${rows.map((r, i) => `${i ? "L" : "M"}${xs(i).toFixed(1)},${y(r.value).toFixed(1)}`).join("")}" fill="none" stroke="${stale ? "#5a6b7a" : "var(--accent)"}" stroke-width="2"></path>`;
  const labelEvery = Math.max(1, Math.ceil(rows.length / 12));
  rows.forEach((r, i) => {
    const agree = cs.refTop != null && r.top === cs.refTop;
    html += `<circle cx="${xs(i).toFixed(1)}" cy="${y(r.value).toFixed(1)}" r="3.5" fill="${agree ? "var(--green)" : (stale ? "#5a6b7a" : "var(--accent)")}"><title>${r.label} · v ${r.value.toFixed(3)} · top ${r.topCell || "—"}${agree ? " (= current top)" : ""}</title></circle>`;
    if (agree) html += `<line x1="${xs(i).toFixed(1)}" y1="${H - padB + 3}" x2="${xs(i).toFixed(1)}" y2="${H - padB + 10}" stroke="var(--green)" stroke-width="2"></line>`;
    if (i % labelEvery === 0) html += `<text x="${xs(i).toFixed(1)}" y="${H - 4}" fill="#5a6b7a" font-size="10" text-anchor="middle">${r.label}</text>`;
  });
  const headTxt = stale
    ? `<span class="dbg-muted">stale — swept a different position; Re-run for current</span>`
    : `<span class="dbg-muted">value (stm) per checkpoint · green tick = top move matches current${cs.refTopCell ? ` (${cs.refTopCell})` : ""}</span>`;
  el.innerHTML = `<div class="dbg-flex-head">${headTxt}${btn}</div>`
    + `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${html}</svg>`;
}

// ---- ply±1 prefetch (spec S7) ---------------------------------------------------------

function dbgAbortPrefetch() {
  if (dbg.prefetchAbort) {
    dbg.prefetchAbort.abort();
    dbg.prefetchAbort = null;
  }
}

function dbgMaybePrefetch() {
  // Client-side only, at most ONE low-priority in-flight request for ply±1,
  // never for checkpoint B, only when the worker dot is green and the user has
  // nothing pending. Aborted the instant the user acts (dbgNavigate).
  const nav = dbg.nav;
  if (dbg.prefetchAbort || dbg.loading || nav.acts.length || nav.ply == null) return;
  if (!dbg.worker || !dbg.worker.alive || !nav.ckptA || !nav.path) return;
  const total = dbgRecordedTotal();
  let target = null;
  for (const d of [1, -1]) {
    const p = nav.ply + d;
    if (p < 0 || (total != null && p > total)) continue;
    const key = dbgCacheKey(Object.assign({}, nav, { ply: p }), nav.ckptA);
    const entry = dbg.cache.get(key);
    if (!entry || !entry.position || !entry.analysis) {
      target = { ply: p, key };
      break;
    }
  }
  if (!target) return;
  const ctl = new AbortController();
  dbg.prefetchAbort = ctl;
  (async () => {
    try {
      const entry = dbgCacheEntry(target.key);
      if (!entry.position) {
        const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), ply: String(target.ply) });
        const res = await fetch(`/api/debug/position?${params.toString()}`, { signal: ctl.signal });
        const data = await safeJson(res);
        if (!res.ok || !data || data.error) return;
        entry.position = data;
      }
      if (!entry.analysis && !entry.analysisPending) {
        entry.analysisPending = true;
        try {
          const body = { run: nav.run, checkpoint: nav.ckptA, path: nav.path, record: nav.rec, ply: target.ply };
          const res = await fetch("/api/debug/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal: ctl.signal,
          });
          const data = await safeJson(res);
          if (res.ok && data && !data.error) entry.analysis = data;
        } finally {
          entry.analysisPending = false;
        }
      }
    } catch (_e) { /* aborted or failed — prefetch is best-effort */ }
    finally {
      if (dbg.prefetchAbort === ctl) dbg.prefetchAbort = null;
    }
  })();
}

// ---- INPUTS tab: featurizer planes (spec S9, dense lineage only) -----------------------

async function dbgEnsureInputs(force) {
  const nav = dbg.nav;
  if (!nav.run || !nav.path || nav.ply == null || !nav.ckptA) return;
  const key = dbgCurrentKey();
  const entry = dbgCacheEntry(key);
  if (force) delete entry.planes;
  if (entry.planes !== undefined || entry.planesPending) {
    dbgRenderInputs();
    return;
  }
  entry.planesPending = true;
  dbgRenderInputs();
  try {
    const body = Object.assign(await dbgRequestBody(nav.ckptA), { planes: true });
    const data = await debugFetchJson("/api/debug/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    entry.planes = data.input_planes || null;  // null = hexgt lineage (graph-featurized)
    if (!entry.analysis) entry.analysis = data;  // a planes analyze is a full analyze
  } catch (e) {
    debugSetStatus(`Inputs: ${e.message}`, "error");
  } finally {
    entry.planesPending = false;
  }
  if (key === dbgCurrentKey()) {
    dbgRenderInputs();
    dbgSyncPlaneSelect();
    if (dbg.nav.mode === "plane") dbgRenderBoard();
  }
}

function dbgRenderInputs() {
  const el = dbgEl("dbgPlanes");
  if (!el) return;
  const entry = dbg.cache.get(dbgCurrentKey());
  if (entry && entry.planesPending) {
    el.__dbgPlanesSig = "";
    el.innerHTML = `<div class="dbg-empty-note">Loading input planes…</div>`;
    return;
  }
  const planes = entry ? entry.planes : undefined;
  if (planes === undefined) {
    el.__dbgPlanesSig = "";
    el.innerHTML = `<div class="dbg-empty-note">Featurizer input planes (dense lineage only) load on first open</div>`;
    return;
  }
  if (planes === null) {
    el.__dbgPlanesSig = "";
    el.innerHTML = `<div class="dbg-empty-note">n/a for hexgt lineage (graph featurizer — no input planes)</div>`;
    return;
  }
  const dimH = (planes.shape && planes.shape[1]) || 41;
  const dimW = (planes.shape && planes.shape[2]) || 41;
  const sig = `${dbgCurrentKey()}|${(planes.names || []).length}`;
  if (el.__dbgPlanesSig === sig) return;  // canvases already drawn for this key
  el.__dbgPlanesSig = sig;
  el.innerHTML = (planes.names || []).map((name, i) =>
    `<div class="dbg-plane-cell"><canvas data-plane="${i}" width="${dimW}" height="${dimH}"></canvas>${escapeText(name)}</div>`).join("");
  el.querySelectorAll("canvas[data-plane]").forEach(cv => {
    const i = Number(cv.dataset.plane);
    const data = (planes.data && planes.data[i]) || [];
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(dimW, dimH);
    let maxV = 0;
    for (const v of data) maxV = Math.max(maxV, Math.abs(v));
    maxV = maxV || 1;
    for (let px = 0; px < data.length && px < dimW * dimH; px++) {
      const t = Math.max(-1, Math.min(1, data[px] / maxV));
      const o = px * 4;
      img.data[o] = t < 0 ? 255 : 39;       // negative -> p1 red, positive -> accent
      img.data[o + 1] = t < 0 ? 86 : 215;
      img.data[o + 2] = t < 0 ? 80 : 230;
      img.data[o + 3] = Math.round(255 * Math.abs(t));
    }
    ctx.putImageData(img, 0, 0);
  });
}

function dbgSyncPlaneSelect() {
  const sel = dbgEl("dbgPlaneSelect");
  if (!sel) return;
  const entry = dbg.cache.get(dbgCurrentKey());
  const names = (entry && entry.planes && entry.planes.names) || [];
  const html = names.map((nm, i) => `<option value="${i}">${escapeText(nm)}</option>`).join("");
  if (sel.__dbgSig !== html) {
    const prev = sel.value;
    sel.__dbgSig = html;
    sel.innerHTML = html || `<option value="0">plane 0</option>`;
    sel.value = prev;
    if (!sel.value) sel.value = "0";
  }
}

// ---- COMPARE helpers (spec S5 + client Δ overlay) ---------------------------------------

function dbgAnalysisBFresh() {
  const nav = dbg.nav;
  if (!nav.ckptB) return null;
  return dbg.keys.analysisB === dbgCacheKey(nav, nav.ckptB) ? dbg.analysisB : null;
}

function dbgClearCmpHeat() {
  if (!dbg.cmpHeat) return;
  dbg.cmpHeat = false;
  const btn = dbgEl("dbgCmpHeat");
  if (btn) btn.classList.remove("active");
}

function dbgToggleCmpHeat() {
  if (!dbg.nav.ckptB) {
    dbgStub("Select checkpoint B in the context strip first.");
    return;
  }
  dbg.cmpHeat = !dbg.cmpHeat;
  const btn = dbgEl("dbgCmpHeat");
  if (btn) btn.classList.toggle("active", dbg.cmpHeat);
  dbgRenderBoard();
}

function dbgToggleSplit(on) {
  if (on && !dbg.nav.ckptB) {
    dbgStub("Select checkpoint B in the context strip first.");
    const cb = dbgEl("dbgCmpSplit");
    if (cb) cb.checked = false;
    return;
  }
  dbg.split = on;
  const wrap = document.querySelector(".dbg-board-wrap");
  if (wrap) wrap.classList.toggle("dbg-split", on);
  const svgB = dbgEl("dbgBoardSvgB");
  if (svgB) svgB.hidden = !on;
  const cb = dbgEl("dbgCmpSplit");
  if (cb) cb.checked = on;
  dbgRenderBoard();
}

function dbgRenderBoardB() {
  // Half-size B board for the A/B split (S5): same cells (dbg.cellIndex carries
  // the A render's geometry), heat = checkpoint B's prior, shared view/zoom.
  const svgB = dbgEl("dbgBoardSvgB");
  if (!svgB || !dbg.split) return;
  const pos = dbg.position;
  if (!pos) {
    svgB.innerHTML = "";
    return;
  }
  if (dbg.view) svgB.setAttribute("viewBox", `${dbg.view.x} ${dbg.view.y} ${dbg.view.width} ${dbg.view.height}`);
  svgB.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const b = dbgAnalysisBFresh();
  const heatB = new Map();
  let maxP = 0;
  if (b) {
    for (const row of b.policy || []) {
      heatB.set(`${row.q},${row.r}`, row.p);
      maxP = Math.max(maxP, row.p);
    }
  }
  let html = "";
  for (const cell of dbg.cellIndex.values()) {
    if (cell.x == null) continue;
    const isStone = Boolean(cell.placement);
    const fill = isStone ? playerColor(cell.placement.player) : "#101924";
    html += `<path class="dbg-cell" d="${path(cell.x, cell.y, HEX - 1)}" fill="${fill}" stroke="${isStone ? "#708296" : "#2c3d50"}" stroke-width="1" opacity="${isStone ? "1" : "0.7"}" data-q="${cell.q}" data-r="${cell.r}"></path>`;
    const p = heatB.get(cell.key);
    if (!isStone && p != null && maxP > 0) {
      let t = p / maxP;
      if (dbg.logScale) t = Math.log1p(99 * t) / Math.log1p(99);
      if (t > 0.015) html += `<path d="${path(cell.x, cell.y, HEX - 2)}" fill="var(--accent)" opacity="${(dbg.opacity * (0.10 + 0.90 * t)).toFixed(3)}" pointer-events="none"></path>`;
    }
  }
  if (!b && dbg.view) {
    html += `<text x="${(dbg.view.x + 8).toFixed(1)}" y="${(dbg.view.y + 18).toFixed(1)}" fill="#5a6b7a" font-size="12">B analysis pending…</text>`;
  }
  svgB.innerHTML = html;
}

// ---- rendering ---------------------------------------------------------------

function dbgRenderAll() {
  try {
    dbgSyncControls();
    dbgRenderCrumb();
    dbgRenderPlyRail();
    dbgRenderBoard();
    dbgRenderBranchBar();
    dbgRenderHeads();
    dbgRenderSearchPanel();
    dbgRenderTargets();
    dbgRenderComparePanel();
    dbgRenderCkptPanel();
    if (dbg.nav.tab === "inputs") {
      dbgRenderInputs();
      dbgSyncPlaneSelect();
    }
    dbgRenderDockChart();
    dbgRenderCkptSweep();  // keeps the Run button present before any sweep ran
    dbgUpdateStaleDots();
  } catch (e) {
    reportError("dbgRenderAll: " + (e && (e.stack || e.message) || e));
  }
}

function dbgSyncSelect(id, html, value) {
  const sel = dbgEl(id);
  if (!sel) return;
  if (sel.__dbgSig !== html) {
    sel.__dbgSig = html;
    sel.innerHTML = html;
  }
  sel.value = value;
}

function dbgCkptLabel(c) {
  const label = c.epoch != null ? `epoch ${c.epoch}` : c.name.replace(/\.pt$/, "");
  return c.graft ? `${label} · ${c.graft}` : label;
}

function dbgCkptShort(name) {
  const ck = dbg.checkpoints.find(c => c.name === name);
  return ck && ck.epoch != null ? `e${ck.epoch}` : String(name).replace(/\.pt$/, "");
}

function dbgRecordLabel(r) {
  // Per-record metadata chips: "g3 · 142mv · P1 · running".
  const win = r.winner ? ` · ${String(r.winner).replace("player", "P")}` : "";
  const status = r.status && r.status !== "final" ? ` · ${r.status}` : "";
  return `g${r.index} · ${r.actions}mv${win}${status}`;
}

function dbgSyncControls() {
  const nav = dbg.nav;
  const runHtml = trainingRuns.length
    ? trainingRuns.map(r => `<option value="${escapeAttr(r.name)}">${escapeText(r.name)}</option>`).join("")
    : `<option value="">No runs</option>`;
  dbgSyncSelect("dbgCtxRun", runHtml, nav.run);
  const srcSel = dbgEl("dbgCtxSource");
  if (srcSel) srcSel.value = nav.src;
  const fileHtml = dbg.games.length
    ? dbg.games.map(g => `<option value="${escapeAttr(g.path)}">${escapeText(g.path)}</option>`).join("")
    : `<option value="">No ${escapeText(nav.src)} games</option>`;
  dbgSyncSelect("dbgCtxFile", fileHtml, nav.path);
  const recHtml = dbg.records.length
    ? dbg.records.map(r => `<option value="${r.index}">${escapeText(dbgRecordLabel(r))}</option>`).join("")
    : `<option value="0">g0</option>`;
  dbgSyncSelect("dbgCtxRecord", recHtml, String(nav.rec));
  const ckptOptions = dbg.checkpoints.map(c => `<option value="${escapeAttr(c.name)}">${escapeText(dbgCkptLabel(c))}</option>`).join("");
  dbgSyncSelect("dbgCtxCkptA", ckptOptions || `<option value="">—</option>`, nav.ckptA);
  dbgSyncSelect("dbgCtxCkptB", `<option value="">—</option>` + ckptOptions, nav.ckptB);
  document.querySelectorAll("#dbgTabs [data-tab]").forEach(b => b.classList.toggle("active", b.dataset.tab === nav.tab));
  for (const tab of DBG_TABS) {
    const panel = dbgEl(DBG_TAB_IDS[tab]);
    if (panel) panel.classList.toggle("active", tab === nav.tab);
  }
  document.querySelectorAll("#dbgModeBar [data-mode]").forEach(b => b.classList.toggle("active", b.dataset.mode === nav.mode));
  const plane = dbgEl("dbgPlaneSelect");
  if (plane) plane.hidden = nav.mode !== "plane";
}

function dbgRenderCrumb() {
  const el = dbgEl("dbgCtxCrumb");
  if (!el) return;
  const nav = dbg.nav;
  if (!nav.run) {
    el.textContent = "no position loaded";
    return;
  }
  const parts = [nav.run];
  if (nav.path) {
    parts.push(String(nav.path).split("/").pop());
    parts.push(`g${nav.rec}`);
  }
  const total = dbgRecordedTotal();
  if (nav.ply != null) parts.push(`ply ${nav.ply}${total != null ? "/" + total : ""}`);
  if (nav.acts.length) parts.push(`branch +${nav.acts.length}`);
  if (nav.ckptA) parts.push(dbgCkptShort(nav.ckptA) + (nav.ckptB ? ` vs ${dbgCkptShort(nav.ckptB)}` : ""));
  el.textContent = parts.join(" · ");
}

function dbgRenderWorkerDot(state2) {
  const el = dbgEl("dbgCtxWorkerDot");
  if (!el) return;
  el.classList.toggle("ok", state2 === "ok");
  el.classList.toggle("err", state2 === "err");
  const w = dbg.worker;
  el.title = w
    ? `Debug worker: ${w.alive ? "alive" : "idle"} · ${w.cached_results} cached · ${w.mode}`
    : (state2 === "err" ? "Debug worker: error" : "Debug worker: unknown");
}

function dbgRenderPlyRail() {
  const pos = dbg.position;
  const totalRec = dbgRecordedTotal();
  const total = totalRec != null ? totalRec : (pos ? pos.debug.total : 0);
  const ply = dbg.nav.ply != null ? Math.min(dbg.nav.ply, total || dbg.nav.ply) : (pos ? pos.debug.ply : 0);
  const readout = dbgEl("dbgPlyReadout");
  if (readout) readout.textContent = `${ply}/${total}`;
  const slider = dbgEl("dbgPlySlider");
  if (slider) {
    slider.max = String(total);
    slider.value = String(ply);
    slider.disabled = !pos;
  }
  const sweep = dbgSweepData();
  // <=900px the rail is a horizontal 22px strip — rows must be laid out with
  // left/width (a ~150-ply game packed vertically into 22px is ~0.15px/row:
  // invisible and untappable).
  const horiz = Boolean(window.matchMedia && window.matchMedia("(max-width: 900px)").matches);
  // The classic slider is the no-sweep fallback (spec §1.2); once the wrongness
  // strip is colored it takes over as the scrubber — except on mobile, where the
  // thin strip rows are too small to be the only scrubber, so keep the slider.
  const sliderWrap = document.querySelector(".dbg-ply-slider-wrap");
  if (sliderWrap) sliderWrap.hidden = Boolean(sweep) && !horiz;
  const strip = dbgEl("dbgPlyStrip");
  if (!strip) return;
  // Flat neutral rows until a game error sweep colors them (M9): background =
  // policy KL, right 30% tinted by |value error|, blunder/top-1-miss markers.
  const sig = `${total}|${horiz ? "h" : "v"}|${sweep ? `${dbgSweepKey()}#${sweep.version}` : 0}`;
  if (strip.__dbgSig !== sig) {
    strip.__dbgSig = sig;
    let maxKl = 0;
    const blunders = sweep ? new Set(dbgBlunders(sweep)) : null;
    if (sweep) for (const p of sweep.plies) if (p.kl != null) maxKl = Math.max(maxKl, p.kl);
    let html = "";
    // One row per navigable position 0..total (matches nav.ply); sweep rows
    // exist for plies 0..total-1 only (the post-final position is never
    // evaluated), so the bottom row stays neutral.
    for (let i = 0; total > 0 && i <= total; i++) {
      const off = (i / (total + 1)) * 100;
      const row = sweep ? sweep.byPly.get(i) : null;  // sweep row = position AT ply i
      let cls = "dbg-ply-row";
      let style = "";
      let inner = "";
      let title = `ply ${i}`;
      if (row) {
        if (row.kl != null && maxKl > 0) {
          const t = Math.min(1, row.kl / maxKl);
          style = `background:rgba(255,99,90,${(0.05 + 0.65 * t).toFixed(3)});`;
          title += ` · KL ${row.kl.toFixed(3)}`;
        }
        const verr = row.value_err_z != null ? Math.abs(row.value_err_z)
          : (row.value_err_soft != null ? Math.abs(row.value_err_soft) : null);
        if (verr != null) {
          inner = `<span class="dbg-ply-verr" style="background:rgba(245,197,66,${Math.min(0.85, verr / 2).toFixed(3)})"></span>`;
          title += ` · |v err| ${verr.toFixed(2)}`;
        }
        if (blunders && blunders.has(i)) {
          cls += " dbg-ply-blunder";
          title += " · blunder";
        }
        if (row.top1_match === false) {
          cls += " dbg-ply-miss";
          title += " · top-1 miss";
        }
      }
      const geom = horiz
        ? `left:${off.toFixed(3)}%;width:${(100 / (total + 1)).toFixed(3)}%;`
        : `top:${off.toFixed(3)}%;height:${(100 / (total + 1)).toFixed(3)}%;`;
      html += `<div class="${cls}" data-ply="${i}" style="${geom}${style}" title="${title}">${inner}</div>`;
    }
    strip.innerHTML = html;
  }
  strip.querySelectorAll(".dbg-ply-cur").forEach(rowEl => rowEl.classList.remove("dbg-ply-cur"));
  const cur = strip.querySelector(`[data-ply="${ply}"]`);
  if (cur) cur.classList.add("dbg-ply-cur");
}

// ---- board heat (spec M3/M5) ---------------------------------------------------

function dbgHeatForMode(mode) {
  // One heat source per render: {values: Map("q,r" -> number), scale, min, max,
  // note}. Modes without their data yet return an empty map + a legend note.
  const out = { mode, values: new Map(), scale: "seq", min: 0, max: 0, note: "" };
  const a = dbgFreshData("analysis");
  const s = dbgFreshData("search");
  const tree = dbgFreshData("tree");
  if (dbg.cmpHeat) {
    // COMPARE Δ overlay: client-computed per-cell prior Δ(A−B), not a server
    // mode — it overrides the base mode while toggled (spec §1.4).
    out.mode = "cmp";
    out.scale = "div";
    const b = dbgAnalysisBFresh();
    if (!a || !b) out.note = "A/B analyses pending";
    else {
      const bPrior = new Map((b.policy || []).map(row => [`${row.q},${row.r}`, row.p]));
      for (const row of a.policy || []) {
        const key = `${row.q},${row.r}`;
        out.values.set(key, row.p - (bPrior.get(key) || 0));
      }
      for (const [key, p] of bPrior) if (!out.values.has(key)) out.values.set(key, -p);
      out.note = "prior Δ(A−B)";
    }
    return dbgHeatFinish(out);
  }
  if (mode === "none") return dbgHeatFinish(out);
  if (mode === "prior") {
    if (!a) out.note = "analyze pending";
    else for (const row of a.policy || []) out.values.set(`${row.q},${row.r}`, row.p);
  } else if (mode === "visits") {
    if (!s) out.note = "no search yet";
    else for (const row of s.visit_policy || []) out.values.set(`${row.q},${row.r}`, row.p);
  } else if (mode === "delta") {
    if (!s) out.note = "no search yet";
    else {
      out.scale = "div";  // promoted (visits > prior) vs demoted, diverging
      const prior = new Map((s.root_prior || []).map(row => [`${row.q},${row.r}`, row.p]));
      for (const row of s.visit_policy || []) {
        const key = `${row.q},${row.r}`;
        out.values.set(key, row.p - (prior.get(key) || 0));
      }
      for (const [key, p] of prior) if (!out.values.has(key)) out.values.set(key, -p);
    }
  } else if (mode === "opp") {
    if (!a) out.note = "analyze pending";
    else if (!a.opp_policy || !a.opp_policy.length) out.note = "no opp head";
    else for (const row of a.opp_policy) out.values.set(`${row.q},${row.r}`, row.p);
  } else if (mode === "target") {
    const rr = dbgRecordRow();
    if (!rr || !rr.found || !rr.row) out.note = "no training row";
    else for (const row of rr.row.policy || []) out.values.set(`${row.q},${row.r}`, row.p);
  } else if (mode === "mismatch") {
    const rr = dbgRecordRow();
    if (!a) out.note = "analyze pending";
    else if (!rr || !rr.found || !rr.row) out.note = "no training row";
    else {
      out.scale = "div";
      const target = new Map((rr.row.policy || []).map(row => [`${row.q},${row.r}`, row.p]));
      for (const row of a.policy || []) {
        const key = `${row.q},${row.r}`;
        out.values.set(key, row.p - (target.get(key) || 0));
      }
      for (const [key, p] of target) if (!out.values.has(key)) out.values.set(key, -p);
    }
  } else if (mode === "childq") {
    if (!tree || !tree.tree) out.note = "run a debug tree";
    else {
      out.scale = "div";
      for (const ch of tree.tree.children || []) out.values.set(`${ch.q},${ch.r}`, ch.qm);
    }
  } else if (mode === "plane") {
    const entry = dbg.cache.get(dbgCurrentKey());
    const planes = entry ? entry.planes : undefined;
    if (entry && entry.planesPending) out.note = "loading planes…";
    else if (planes === undefined) out.note = "open INPUTS once to load planes";
    else if (planes === null) out.note = "n/a for hexgt lineage";
    else {
      out.scale = "div";
      const sel = dbgEl("dbgPlaneSelect");
      const idx = sel ? Number(sel.value) || 0 : 0;
      const data = (planes.data && planes.data[idx]) || [];
      const dim = (planes.shape && planes.shape[1]) || 41;
      const half = Math.floor(dim / 2);
      // The dense 41x41 crop is anchored on the ROUNDED MEAN of the placed
      // stones (geometry.crop_center / encoding.rs model1_crop_center), NOT on
      // axial (0,0) — projecting without the center shifts every cell by the
      // centroid offset on any non-origin-centered position.
      const center = dbgPlaneCropCenter(planes);
      const pos = dbg.position;
      const coords = pos ? (pos.legal || []).concat(debugCurrentPlacements()) : [];
      for (const c of coords) {
        const qi = c.q - center.q + half;
        const ri = c.r - center.r + half;
        if (qi < 0 || ri < 0 || qi >= dim || ri >= dim) continue;
        const v = data[ri * dim + qi];
        if (v) out.values.set(`${c.q},${c.r}`, v);
      }
      out.note = (planes.names && planes.names[idx]) || "";
    }
  }
  return dbgHeatFinish(out);
}

function dbgRoundHalfEven(x) {
  // Banker's rounding — matches Python round() (geometry.crop_center) and the
  // Rust encoder's ties-to-even (encoding.rs model1_crop_center).
  const f = Math.floor(x);
  if (x - f !== 0.5) return Math.round(x);
  return f % 2 === 0 ? f : f + 1;
}

function dbgPlaneCropCenter(planes) {
  // Crop anchor for projecting board axial coords into the 41x41 input planes.
  // Prefer the server-reported center (input_planes.center = [q, r], additive
  // §3.6 field); fall back to recomputing the stone-centroid client-side so the
  // overlay stays correct against older analyze payloads.
  if (planes && Array.isArray(planes.center) && planes.center.length === 2) {
    return { q: Math.trunc(Number(planes.center[0]) || 0), r: Math.trunc(Number(planes.center[1]) || 0) };
  }
  const stones = debugCurrentPlacements();
  if (!stones.length) return { q: 0, r: 0 };
  let sq = 0;
  let sr = 0;
  for (const s of stones) {
    sq += s.q;
    sr += s.r;
  }
  return { q: dbgRoundHalfEven(sq / stones.length), r: dbgRoundHalfEven(sr / stones.length) };
}

function dbgHeatFinish(out) {
  let min = Infinity;
  let max = -Infinity;
  for (const v of out.values.values()) {
    min = Math.min(min, v);
    max = Math.max(max, v);
  }
  if (!out.values.size) {
    min = 0;
    max = 0;
  }
  out.min = min;
  out.max = max;
  return out;
}

function dbgHeatNorm(heat, v) {
  // Log toggle (key L) compresses ~two decades so a 0.9-prior board and a
  // near-flat board stay visually distinguishable.
  if (heat.scale === "div") {
    const m = Math.max(Math.abs(heat.min), Math.abs(heat.max)) || 1;
    let t = Math.max(-1, Math.min(1, v / m));
    if (dbg.logScale) t = Math.sign(t) * (Math.log1p(99 * Math.abs(t)) / Math.log1p(99));
    return t;
  }
  const m = heat.max > 0 ? heat.max : 1;
  let t = Math.max(0, Math.min(1, v / m));
  if (dbg.logScale) t = Math.log1p(99 * t) / Math.log1p(99);
  return t;
}

function dbgFormatHeatValue(mode, v) {
  if (mode === "childq") return `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
  if (mode === "plane") return `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
  const pct = v * 100;
  const sign = (mode === "delta" || mode === "mismatch" || mode === "cmp") && v >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}%`;
}

function dbgRenderLegend(heat) {
  const minEl = dbgEl("dbgLegendMin");
  const maxEl = dbgEl("dbgLegendMax");
  const scaleEl = document.querySelector("#dbgLegend .dbg-legend-scale");
  if (!minEl || !maxEl) return;
  if (!heat || !heat.values.size) {
    minEl.textContent = "—";
    maxEl.textContent = heat && heat.note ? heat.note : "—";
    if (scaleEl) {
      scaleEl.style.background = "";
      scaleEl.title = heat && heat.note ? heat.note : "";
    }
    return;
  }
  minEl.textContent = dbgFormatHeatValue(heat.mode, heat.min);
  maxEl.textContent = dbgFormatHeatValue(heat.mode, heat.max);
  if (scaleEl) {
    scaleEl.title = heat.note || "";  // cmp/plane keep their label even with data
    scaleEl.style.background = heat.scale === "div"
      ? "linear-gradient(90deg, var(--p1), rgba(16,25,36,0.9), var(--green))"
      : "";  // default CSS gradient covers the sequential scale
  }
}

function dbgBuildHudMetrics() {
  // One Map per render feeds the hover HUD — no linear .find in mousemove (M3).
  const metrics = new Map();
  const get = (q, r) => {
    const key = `${q},${r}`;
    let m = metrics.get(key);
    if (!m) {
      m = {};
      metrics.set(key, m);
    }
    return m;
  };
  const a = dbgFreshData("analysis");
  const s = dbgFreshData("search");
  const tree = dbgFreshData("tree");
  if (a) {
    for (const row of a.policy || []) get(row.q, row.r).prior = row.p;
    for (const row of a.opp_policy || []) get(row.q, row.r).opp = row.p;
  }
  if (s) for (const row of s.visit_policy || []) get(row.q, row.r).visits = row.p;
  if (tree && tree.tree) {
    for (const ch of tree.tree.children || []) {
      const m = get(ch.q, ch.r);
      m.childQ = ch.qm;
      m.childN = ch.n;
    }
  }
  const rr = dbgRecordRow();
  if (rr && rr.found && rr.row) for (const row of rr.row.policy || []) get(row.q, row.r).target = row.p;
  const b = dbgAnalysisBFresh();  // split board / cmp overlay HUD column (S5)
  if (b) for (const row of b.policy || []) get(row.q, row.r).priorB = row.p;
  return metrics;
}

function dbgRenderBoard() {
  if (!debugBoardSvg) return;
  if (!dbg.nav.ckptB && (dbg.cmpHeat || dbg.split)) {
    // Compare modes die with checkpoint B (no re-render here — we ARE rendering).
    dbgClearCmpHeat();
    dbg.split = false;
    const wrap = document.querySelector(".dbg-board-wrap");
    if (wrap) wrap.classList.remove("dbg-split");
    const svgB = dbgEl("dbgBoardSvgB");
    if (svgB) svgB.hidden = true;
    const cb = dbgEl("dbgCmpSplit");
    if (cb) cb.checked = false;
  }
  const heat = dbgHeatForMode(dbg.nav.mode);
  dbgRenderLegend(heat);
  dbg.hudMetrics = dbgBuildHudMetrics();
  const pos = dbg.position;
  if (!pos) {
    debugBoardSvg.innerHTML = "";
    dbg.cellIndex = new Map();
    return;
  }

  const cells = new Map();
  const addCell = (q, r, extra) => {
    const key = `${q},${r}`;
    const existing = cells.get(key) || { q, r, key };
    cells.set(key, Object.assign(existing, extra));
  };
  for (const c of (pos.legal || [])) addCell(c.q, c.r, { legal: true });
  const placements = debugCurrentPlacements();
  for (const p of placements) addCell(p.q, p.r, { placement: p });
  // Candidate cells from the CURRENT key's analysis (so heat shows even off the
  // legal set); a stale analysis must not add phantom cells to a new position.
  const freshA = dbgFreshData("analysis");
  if (freshA) for (const row of freshA.policy || []) addCell(row.q, row.r, { candidate: true });
  for (const key of heat.values.keys()) {
    const [q, r] = key.split(",").map(Number);
    addCell(q, r, {});
  }
  dbg.cellIndex = cells;

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const data = [];
  for (const cell of cells.values()) {
    const c = center(cell.q, cell.r);
    minX = Math.min(minX, c.x - HEX * 1.6);
    maxX = Math.max(maxX, c.x + HEX * 1.6);
    minY = Math.min(minY, c.y - HEX * 1.6);
    maxY = Math.max(maxY, c.y + HEX * 1.6);
    data.push(Object.assign(cell, { x: c.x, y: c.y }));
  }
  if (!Number.isFinite(minX)) {
    debugBoardSvg.innerHTML = "";
    return;
  }
  // Pan/zoom-aware viewBox: the bounds only reset the camera when the user
  // hasn't zoomed (viewDirty), exactly like the Match board.
  dbgSyncView({ x: minX, y: minY, width: maxX - minX, height: maxY - minY });
  debugBoardSvg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  const ov = dbg.overlays;
  const nav = dbg.nav;
  // Last move = the highest-index stone shown at this ply (derived client-side
  // so the highlight tracks instantly on step); server coords as fallback.
  const lastStone = placements.length ? placements[placements.length - 1] : null;
  const lastCoord = lastStone
    ? { q: lastStone.q, r: lastStone.r }
    : ((pos.debug.last_q != null && pos.debug.last_r != null) ? { q: pos.debug.last_q, r: pos.debug.last_r } : null);
  data.sort((a, b) => (a.placement ? 1 : 0) - (b.placement ? 1 : 0));
  let html = "";

  // Threat windows (count >= 4) drawn under the cells as colored connectors.
  if (ov.threats && pos.tactics && Array.isArray(pos.tactics.threats)) {
    for (const w of pos.tactics.threats) {
      const pts = (w.cells || []).map(c => center(c.q, c.r));
      if (pts.length < 2) continue;
      const owner = (w.threat_player || w.active_player || "");
      const color = owner.endsWith("1") ? "var(--p1)" : "var(--p0)";
      const d = pts.map((c, i) => `${i ? "L" : "M"}${c.x.toFixed(1)},${c.y.toFixed(1)}`).join("");
      html += `<path d="${d}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" opacity="0.5" pointer-events="none"></path>`;
    }
  }
  for (const h of data) {
    const isStone = Boolean(h.placement);
    const fill = isStone ? playerColor(h.placement.player) : "#101924";
    const stroke = isStone ? "#708296" : "#2c3d50";
    const opacity = isStone ? "1" : (h.legal || h.candidate ? "0.7" : (ov.legalDim ? "0.18" : "0.45"));
    html += `<path class="dbg-cell" d="${path(h.x, h.y, HEX - 1)}" fill="${fill}" stroke="${stroke}" stroke-width="1" opacity="${opacity}" data-q="${h.q}" data-r="${h.r}"></path>`;
    if (!isStone && heat.values.has(h.key)) {
      const t = dbgHeatNorm(heat, heat.values.get(h.key));
      if (heat.scale === "div") {
        if (Math.abs(t) > 0.015) {
          html += `<path d="${path(h.x, h.y, HEX - 2)}" fill="${t >= 0 ? "var(--green)" : "var(--p1)"}" opacity="${(dbg.opacity * (0.10 + 0.90 * Math.abs(t))).toFixed(3)}" pointer-events="none"></path>`;
        }
      } else if (t > 0.015) {
        html += `<path d="${path(h.x, h.y, HEX - 2)}" fill="var(--accent)" opacity="${(dbg.opacity * (0.10 + 0.90 * t)).toFixed(3)}" pointer-events="none"></path>`;
      }
    }
    if (ov.last && lastCoord && h.q === lastCoord.q && h.r === lastCoord.r) {
      html += `<path class="dbg-last" d="${path(h.x, h.y, HEX - 0.5)}" fill="none" stroke="var(--accent)" stroke-width="2.2" pointer-events="none"></path>`;
    }
    if (isStone && nav.acts.length && h.placement.index > nav.ply) {
      // Injected what-if stone — dashed accent ring matches its branch chip.
      html += `<path d="${path(h.x, h.y, HEX - 3)}" fill="none" stroke="var(--accent)" stroke-width="1.6" stroke-dasharray="4 3" pointer-events="none"></path>`;
    }
    if (isStone && ov.numbers) {
      html += `<text class="dbg-stone-label" x="${h.x}" y="${h.y}" text-anchor="middle" dominant-baseline="central">${h.placement.index}</text>`;
    }
  }
  // Tree PV (or a clicked node's preview line) as NUMBERED GHOST stones (M10).
  // Neutral accent — future-move ownership would need engine turn logic, so no
  // player-color guessing. style beats the .dbg-ghost-stone pointer-events:none
  // so a ghost click is a no-op (it never reaches the .dbg-cell underneath).
  const treeFresh = dbgFreshData("tree");
  const preview = dbg.treePreview && dbg.treePreview.key === dbgCurrentKey() ? dbg.treePreview.ids : null;
  const ghostLine = dbg.treeGhostsOff ? [] : (preview || (treeFresh && treeFresh.pv) || []);
  ghostLine.forEach((id, i) => {
    const g = dbgUnpackActionId(id);
    const c = center(g.q, g.r);
    html += `<g class="dbg-ghost-stone" style="pointer-events:all">`
      + `<path d="${path(c.x, c.y, HEX - 2.5)}" fill="#101924" stroke="var(--accent)" stroke-width="1.6" opacity="0.85"></path>`
      + `<text class="dbg-stone-label" x="${c.x}" y="${c.y}" text-anchor="middle" dominant-baseline="central" style="fill:var(--accent)">${i + 1}</text>`
      + `</g>`;
  });
  debugBoardSvg.innerHTML = html;
  if (dbg.split) dbgRenderBoardB();  // A/B split keeps the B half in lockstep (S5)
  // Hover + click are DELEGATED (bound once in debugBindEvents) — no per-cell
  // listeners on a 1000+ cell board.
}

function dbgHoverCell(q, r) {
  const hud = dbgEl("debugBoardHud");
  if (!hud) return;
  if (dbg.split) {
    // Shared hover cursor (S5): highlight the cell on BOTH half-boards.
    for (const svg of [debugBoardSvg, dbgEl("dbgBoardSvgB")]) {
      if (!svg) continue;
      svg.querySelectorAll(".dbg-hl").forEach(elx => elx.classList.remove("dbg-hl"));
      const cell = svg.querySelector(`.dbg-cell[data-q="${q}"][data-r="${r}"]`);
      if (cell) cell.classList.add("dbg-hl");
    }
  }
  const m = dbg.hudMetrics.get(`${q},${r}`) || {};
  const pct = v => (v != null ? `${(v * 100).toFixed(1)}%` : "—");
  const childQ = m.childQ != null ? `${m.childQ >= 0 ? "+" : ""}${m.childQ.toFixed(2)} (N=${m.childN})` : "—";
  let line = `${q},${r} · prior ${pct(m.prior)} · visits ${pct(m.visits)} · childQ ${childQ} · target ${pct(m.target)}`;
  if ((dbg.split || dbg.cmpHeat) && dbg.nav.ckptB) line += ` · B prior ${pct(m.priorB)}`;
  hud.innerHTML = `<div>${escapeText(line)}</div>`;
}

function dbgRenderBranchBar() {
  const bar = dbgEl("dbgBranchBar");
  if (!bar) return;
  const acts = dbg.nav.acts;
  bar.hidden = !acts.length;
  const chips = bar.querySelector(".dbg-branch-chips");
  if (!chips) return;
  if (!acts.length) {
    chips.innerHTML = "";
    return;
  }
  let html = `<span class="dbg-chip dbg-chip-game" title="Recorded-game prefix">game @ ply ${dbg.nav.ply}</span>`;
  acts.forEach((id, i) => {
    const c = dbgUnpackActionId(id);
    html += `<span class="dbg-chip dbg-chip-inj">+${c.q},${c.r}<button class="dbg-chip-x" type="button" data-act-cut="${i}" title="Undo to before this move">✕</button></span>`;
  });
  chips.innerHTML = html;
}

// ---- HEADS panel ---------------------------------------------------------------

function dbgRenderHeads() {
  const panel = dbgEl("dbgTabHeads");
  if (!panel) return;
  const a = dbg.analysis;
  const scalarEl = panel.querySelector(".dbg-value-scalar strong");
  const chip = dbgEl("dbgValueChip");
  const distEl = dbgEl("dbgValueDist");
  const swapBlock = panel.querySelector(".dbg-ownerswap");
  const swapEl = panel.querySelector(".dbg-ownerswap-body");
  const stvEl = dbgEl("dbgStvRows");
  const mlEl = dbgEl("dbgMovesLeft");
  const movesEl = dbgEl("dbgTopMoves");
  const oppEl = dbgEl("dbgOppList");
  if (!a) {
    const note = dbg.loading ? "Evaluating…" : "No analysis yet";
    if (scalarEl) {
      scalarEl.textContent = "—";
      scalarEl.className = "";
    }
    if (chip) {
      chip.textContent = "—";
      chip.className = "dbg-chip";
      chip.title = "";
    }
    if (distEl) distEl.innerHTML = `<div class="dbg-empty-note">${note}</div>`;
    // Default-hidden pre-analysis: the probe is hexgt-only (spec §1.4) and the
    // analyze commit unhides it only when value_swapped is present.
    if (swapBlock) swapBlock.hidden = true;
    if (swapEl) swapEl.innerHTML = `<span class="dbg-muted">—</span>`;
    if (stvEl) stvEl.innerHTML = `<div class="dbg-empty-note">Short-term value horizons</div>`;
    if (mlEl) {
      mlEl.className = "dbg-moves-left dbg-empty-note";
      mlEl.textContent = "Moves-left: —";
    }
    if (movesEl) movesEl.innerHTML = `<div class="dbg-empty-note">${note}</div>`;
    if (oppEl) oppEl.innerHTML = `<div class="dbg-empty-note">Opponent policy top-k</div>`;
    return;
  }
  if (scalarEl) {
    scalarEl.textContent = a.value.toFixed(4);
    scalarEl.className = a.value >= 0 ? "pos" : "neg";
  }
  if (chip) {
    chip.textContent = `${a.value >= 0 ? "+" : ""}${a.value.toFixed(3)} stm`;
    chip.className = `dbg-chip ${a.value >= 0 ? "pos" : "neg"}`;
    chip.title = `to play P${a.current_player} (${a.current_role || ""}) · candidates ${a.candidate_count} · legal ${a.legal_count}`;
  }
  if (distEl) distEl.innerHTML = dbgValueDistHtml(a);
  // B4: the owner-swap P0/P1 probe is off-distribution-confounded (FirstStone
  // parity) and must NOT be presented as optimism — no ok/warn coloring.
  if (swapBlock) swapBlock.hidden = typeof a.value_swapped !== "number";
  if (swapEl) {
    if (typeof a.value_swapped === "number") {
      const sum = a.optimism != null ? a.optimism : (a.value + a.value_swapped);
      swapEl.innerHTML = `
        <div class="info-row"><span class="label">Side to move</span><span class="value">${a.value.toFixed(3)}</span></div>
        <div class="info-row"><span class="label">Owner-swapped</span><span class="value">${a.value_swapped.toFixed(3)}</span></div>
        <div class="info-row" title="v_self + v_swapped; off-distribution (FirstStone parity) — not an optimism metric"><span class="label">Σ</span><span class="value">${sum >= 0 ? "+" : ""}${sum.toFixed(3)}</span></div>`;
    } else {
      swapEl.innerHTML = `<span class="dbg-muted">n/a for this lineage</span>`;
    }
  }
  if (stvEl) {
    const stvRows = Object.keys(a.stvalue || {}).sort((x, y) => Number(x) - Number(y)).map(h => {
      const s = a.stvalue[h].scalar;
      return `<div class="dbg-stv-row"><span class="label">STV+${h}</span><span class="dbg-bar-track"><span class="dbg-bar-fill ${s >= 0 ? "pos" : "neg"}" style="width:${(Math.abs(s) * 50).toFixed(1)}%;${s >= 0 ? "left:50%" : "right:50%"}"></span></span><span class="value">${s.toFixed(3)}</span></div>`;
    }).join("");
    stvEl.innerHTML = stvRows || `<div class="dbg-empty-note">No STV heads</div>`;
  }
  if (mlEl) {
    if (a.moves_left && typeof a.moves_left.scalar === "number") {
      // B1 fix: decode with the REAL cap from analyze meta (512 for dense),
      // not the old hardcoded *80.
      const cap = a.meta && a.meta.moves_left_cap != null ? a.meta.moves_left_cap : 512;
      const remaining = Math.max(0, (a.moves_left.scalar + 1) / 2 * cap);
      const pos = dbg.position;
      const actual = (pos && !dbg.nav.acts.length && !pos.debug.imported && pos.debug.winner)
        ? ` · actual ${pos.debug.total - Math.min(dbg.nav.ply != null ? dbg.nav.ply : pos.debug.ply, pos.debug.total)}`
        : "";
      mlEl.className = "dbg-moves-left";
      mlEl.textContent = `Moves-left: ~${remaining.toFixed(0)} (cap ${cap})${actual}`;
    } else {
      mlEl.className = "dbg-moves-left dbg-empty-note";
      mlEl.textContent = "Moves-left: — (no head in this lineage)";
    }
  }
  if (movesEl) movesEl.innerHTML = dbgTopMovesHtml(a);
  if (oppEl) oppEl.innerHTML = dbgOppListHtml(a);
}

function dbgValueDistHtml(a) {
  const dist = a.value_dist || [];
  const bins = a.value_bins || [];
  if (!dist.length) return `<div class="dbg-empty-note">No value distribution</div>`;
  const maxP = dist.reduce((m, x) => Math.max(m, x), 0) || 1;
  let entropy = 0;
  const bars = dist.map((p, i) => {
    if (p > 0) entropy -= p * Math.log(p);
    const h = Math.max(1, (p / maxP) * 100);
    const c = bins[i] != null ? bins[i] : 0;
    return `<span class="dbg-vbar ${c >= 0 ? "pos" : "neg"}" style="height:${h.toFixed(1)}%" title="v=${c.toFixed(3)} p=${(p * 100).toFixed(1)}%"></span>`;
  }).join("");
  let markers = "";
  const pos = dbg.position;
  if (pos && !dbg.nav.acts.length && pos.debug.winner && pos.current_player) {
    const z = pos.debug.winner === pos.current_player ? 1 : -1;
    markers += `<span class="dbg-vdist-marker z" style="left:${(((z + 1) / 2) * 100).toFixed(1)}%" title="final z (stm) = ${z}"></span>`;
  }
  const rr = dbgRecordRow();  // soft-z target marker once TARGETS rows load (F3)
  if (rr && rr.found && rr.row && typeof rr.row.value_target === "number") {
    markers += `<span class="dbg-vdist-marker soft" style="left:${(((rr.row.value_target + 1) / 2) * 100).toFixed(1)}%" title="recorded soft-z target ${rr.row.value_target.toFixed(3)}"></span>`;
  }
  return `
    <div class="dbg-vdist" aria-label="65-bin value distribution">${bars}${markers}</div>
    <div class="dbg-vdist-axis"><span>loss −1</span><span>0</span><span>+1 win</span></div>
    <div class="dbg-muted">dist entropy ${entropy.toFixed(2)} nats</div>`;
}

function dbgTopMovesHtml(a) {
  // Search/tree columns are key-gated: a stale ply's visits/Q must never line
  // up against this analysis' prior rows (the action ids would misalign).
  const s = dbgFreshData("search");
  const tree = dbgFreshData("tree");
  const visitsById = new Map();
  if (s) for (const row of s.visit_policy || []) visitsById.set(row.action_id, row.p);
  const treeById = new Map();
  let rootQ = null;
  if (tree && tree.tree) {
    rootQ = tree.tree.qm;
    for (const ch of tree.tree.children || []) treeById.set(ch.action_id, ch);
  }
  const recorded = (!dbg.nav.acts.length && dbg.gameActs.length > dbg.nav.ply) ? dbg.gameActs[dbg.nav.ply] : null;
  // Row candidates = top-12 priors UNION search-promoted moves (fresh visit
  // share > 3%) UNION fresh tree root children: a low-prior move the search
  // promoted must get a row (the `under` badge and visits/Q/Δ sorts depend on
  // it). a.policy is the complete legal set, so prior/coords always resolve.
  const priorById = new Map((a.policy || []).map(p => [p.action_id, p]));
  const ids = [];
  const seen = new Set();
  const add = id => {
    if (priorById.has(id) && !seen.has(id)) {
      seen.add(id);
      ids.push(id);
    }
  };
  for (const p of (a.policy || []).slice(0, 12)) add(p.action_id);
  if (s) for (const row of s.visit_policy || []) if (row.p > 0.03) add(row.action_id);
  for (const [id, node] of treeById) if (node.n > 0) add(id);
  const rows = ids.map(id => {
    const p = priorById.get(id);
    const visits = visitsById.has(id) ? visitsById.get(id) : null;
    const node = treeById.get(id);
    return {
      action_id: id,
      q: p.q,
      r: p.r,
      prior: p.p,
      visits,
      qm: node ? node.qm : null,  // Q column fills from the search_tree root layer (F3)
      delta: visits != null ? visits - p.p : null,
    };
  });
  const sortKey = dbg.sortKey;
  const sortVal = row => {
    const v = sortKey === "delta" ? (row.delta != null ? Math.abs(row.delta) : null) : row[sortKey];
    return v == null ? -Infinity : v;
  };
  rows.sort((x, y) => sortVal(y) - sortVal(x));
  rows.length = Math.min(rows.length, 16);
  const arrow = k => (sortKey === k ? " ▾" : "");
  const head = `<div class="dbg-move-row dbg-move-head"><span>#</span><span>cell</span><span class="sortable" data-dbg-sort="prior">prior${arrow("prior")}</span><span class="sortable" data-dbg-sort="visits">visits${arrow("visits")}</span><span class="sortable" data-dbg-sort="qm">Q${arrow("qm")}</span><span class="sortable" data-dbg-sort="delta">Δ${arrow("delta")}</span></div>`;
  const body = rows.map((row, i) => {
    const isBest = s && s.best_action_id === row.action_id;
    const isRec = recorded != null && recorded === row.action_id;
    let badge = "";
    if (row.qm != null && rootQ != null && row.prior > 0.15 && row.qm < rootQ - 0.15) {
      badge = ` <span class="dbg-badge dbg-badge-over" title="high prior met low Q">over</span>`;
    } else if (row.visits != null && row.prior < 0.03 && row.visits > 0.10) {
      badge = ` <span class="dbg-badge dbg-badge-under" title="low prior earned high visits">under</span>`;
    }
    return `<div class="dbg-move-row${isBest ? " dbg-move-best" : ""}"${isRec ? ` title="recorded move"` : ""}><span>${i + 1}</span><span>${row.q},${row.r}${isRec ? " ●" : ""}${badge}</span><span>${(row.prior * 100).toFixed(1)}%</span><span>${row.visits != null ? (row.visits * 100).toFixed(1) + "%" : "—"}</span><span>${row.qm != null ? row.qm.toFixed(2) : "—"}</span><span>${row.delta != null ? (row.delta >= 0 ? "+" : "") + (row.delta * 100).toFixed(1) + "%" : "—"}</span></div>`;
  }).join("");
  return head + body;
}

function dbgRecordedReplyActionId() {
  // The actual recorded reply by the OPPONENT (Hexo turns place two stones, so
  // scan forward for the first recorded action by the other player).
  const nav = dbg.nav;
  if (nav.acts.length || !dbg.position || dbg.position.debug.imported) return null;
  const acts = dbgRecordedActs();
  const stm = dbg.position.current_player;
  if (!acts || !stm) return null;
  for (let j = nav.ply; j < acts.length && j < nav.ply + 4; j++) {
    const pl = dbg.gamePlacements.find(p => p.index === j + 1);
    if (pl && pl.player && pl.player !== stm) return acts[j];
  }
  return null;
}

function dbgOppListHtml(a) {
  const opp = a.opp_policy;
  if (!opp || !opp.length) return `<div class="dbg-empty-note">No opponent-policy head</div>`;
  const reply = dbgRecordedReplyActionId();
  const rows = opp.slice(0, 8).map(row => {
    const hit = reply != null && row.action_id === reply;
    return `<div class="dbg-move-row${hit ? " dbg-move-best" : ""}"${hit ? ` title="actual recorded reply"` : ""}><span></span><span>${row.q},${row.r}${hit ? " ●" : ""}</span><span>${(row.p * 100).toFixed(1)}%</span><span></span><span></span><span></span></div>`;
  }).join("");
  return `<div class="dbg-subhead">Opponent policy</div>${rows}`;
}

// ---- SEARCH / COMPARE / CKPT panels ---------------------------------------------

function dbgRenderSearchPanel() {
  const panel = dbgEl("dbgTabSearch");
  if (!panel) return;
  dbgRenderTree();
  dbgRenderScatter();
  dbgRenderLadder();
  const sum = panel.querySelector(".dbg-search-summary");
  if (!sum) return;
  const s = dbg.search;
  if (!s) {
    sum.className = "dbg-search-summary dbg-empty-note";
    sum.textContent = "No search yet";
    return;
  }
  sum.className = "dbg-search-summary";
  // Key AGREEMENT (not freshness): Δ-vs-raw and the argmax compare are only
  // meaningful when search and analysis describe the same position. Both stale
  // to the SAME key stays visible (the M14 dot already flags it).
  const a = dbg.keys.analysis === dbg.keys.search ? dbg.analysis : null;
  const priorTop = a && a.policy && a.policy[0];
  const agree = priorTop && s.best && priorTop.q === s.best.q && priorTop.r === s.best.r;
  const delta = a ? s.root_value - a.value : null;
  sum.innerHTML = `
    <span>visits <strong>${s.visits}</strong></span>
    <span>root value <strong class="${s.root_value >= 0 ? "pos" : "neg"}">${s.root_value.toFixed(3)}</strong></span>
    <span>best <strong>${s.best ? `${s.best.q},${s.best.r}` : "—"}</strong></span>
    ${priorTop ? `<span class="dbg-muted">prior argmax ${priorTop.q},${priorTop.r} ${agree ? "=" : "≠"}</span>` : ""}
    ${delta != null ? `<span class="dbg-muted">Δ vs raw ${delta >= 0 ? "+" : ""}${delta.toFixed(3)}</span>` : ""}`;
}

function dbgRenderComparePanel() {
  const body = document.querySelector("#dbgTabCompare .dbg-cmp-body");
  if (!body) return;
  if (!dbg.nav.ckptB) {
    body.innerHTML = `<div class="dbg-empty-note">Select checkpoint B in the context strip to compare</div>`;
    return;
  }
  const a = dbg.analysis;
  const b = dbg.analysisB;
  if (!b) {
    // Three-way: in flight / failed / not started — a failed B analyze must
    // surface as an error, not sit on "Evaluating…" forever.
    const entryB0 = dbg.cache.get(dbgCacheKey(dbg.nav, dbg.nav.ckptB));
    if (entryB0 && entryB0.analysisPending) {
      body.innerHTML = `<div class="dbg-empty-note">Evaluating checkpoint B…</div>`;
    } else if (entryB0 && entryB0.analysisError) {
      body.innerHTML = `<div class="dbg-empty-note">B analyze failed: ${escapeText(entryB0.analysisError)} — use ↻ to retry</div>`;
    } else {
      body.innerHTML = `<div class="dbg-empty-note">Checkpoint B not evaluated yet</div>`;
    }
    return;
  }
  const topA = a && a.policy && a.policy[0];
  const topB = b.policy && b.policy[0];
  const dv = a ? b.value - a.value : null;
  const rows = [
    ["Value A", a ? a.value.toFixed(3) : "—"],
    ["Value B", b.value.toFixed(3)],
    ["Δ (B−A)", dv != null ? `${dv >= 0 ? "+" : ""}${dv.toFixed(3)}` : "—"],
  ];
  if (a) {
    for (const h of Object.keys(a.stvalue || {}).sort((x, y) => Number(x) - Number(y))) {
      const sb = b.stvalue && b.stvalue[h];
      if (!sb) continue;
      const d = sb.scalar - a.stvalue[h].scalar;
      rows.push([`STV+${h} Δ`, `${d >= 0 ? "+" : ""}${d.toFixed(3)}`]);
    }
  }
  rows.push(["Top A", topA ? `${topA.q},${topA.r} (${(topA.p * 100).toFixed(0)}%)` : "—"]);
  rows.push(["Top B", topB ? `${topB.q},${topB.r} (${(topB.p * 100).toFixed(0)}%)` : "—"]);
  rows.push(["Agree", topA && topB ? (topA.action_id === topB.action_id ? "yes" : "no") : "—"]);
  // PV divergence: first differing ply of the A vs B search_tree PVs (when both run).
  const treeA = dbgFreshData("tree");
  const entryB = dbg.cache.get(dbgCacheKey(dbg.nav, dbg.nav.ckptB));
  const treeB = entryB && entryB.tree;
  let pvRow;
  if (treeA && treeB) {
    const pa = treeA.pv || [];
    const pb = treeB.pv || [];
    let i = 0;
    while (i < pa.length && i < pb.length && pa[i] === pb[i]) i++;
    if (i >= pa.length && i >= pb.length) pvRow = "identical PVs";
    else {
      const ca = pa[i] != null ? dbgUnpackActionId(pa[i]) : null;
      const cb = pb[i] != null ? dbgUnpackActionId(pb[i]) : null;
      pvRow = `ply +${i + 1}: A ${ca ? `${ca.q},${ca.r}` : "(end)"} vs B ${cb ? `${cb.q},${cb.r}` : "(end)"}`;
    }
  } else {
    pvRow = `needs debug trees (A ${treeA ? "✓" : "—"} · B ${treeB ? "✓" : "—"})`;
  }
  rows.push(["PV divergence", pvRow]);
  body.innerHTML = rows.map(([k, v]) => `<div class="info-row"><span class="label">${k}</span><span class="value">${v}</span></div>`).join("")
    + (treeB ? "" : `<button type="button" id="dbgCmpTreeB" class="dbg-mini-btn" style="margin-top:6px"${dbg.treeBusy ? " disabled" : ""}>Run B debug tree</button>`);
}

function dbgRenderCkptPanel() {
  const el = dbgEl("dbgCkptInfo");
  if (!el) return;
  if (!dbg.nav.ckptA) {
    el.innerHTML = `<div class="dbg-empty-note">No checkpoint selected</div>`;
    return;
  }
  let html = dbgCkptInfoHtml(dbg.nav.ckptA);
  if (dbg.nav.ckptB) {
    html += `<div class="dbg-subhead" style="margin-top:10px">Checkpoint B</div>` + dbgCkptInfoHtml(dbg.nav.ckptB);
  }
  el.innerHTML = html;
}

function dbgCkptInfoHtml(name) {
  const info = dbg.ckptInfo.get(`${dbg.nav.run}|${name}`);
  const ck = dbg.checkpoints.find(c => c.name === name);
  const rows = [["Checkpoint", escapeText(name)]];
  if (ck && ck.epoch != null) rows.push(["Epoch", String(ck.epoch)]);
  if (!info) {
    rows.push(["Provenance", dbg.nav.tab === "ckpt" ? "loading…" : "open this tab to load"]);
  } else if (info.error) {
    rows.push(["Error", escapeText(info.error)]);
  } else {
    const m = info.meta || {};
    if (m.lineage) rows.push(["Lineage", escapeText(m.lineage)]);
    // §3.8 contract: meta.arch is a flattened "key=val, …" display STRING (web.py
    // converts the worker's dict). Parse blocks_type out of it for the Trunk row;
    // stay tolerant of a raw dict in case an older/other backend returns one.
    const archStr = m.arch && typeof m.arch === "object"
      ? Object.keys(m.arch).sort().map(k => `${k}=${m.arch[k]}`).join(", ")
      : (m.arch ? String(m.arch) : "");
    const trunkMatch = archStr.match(/(?:^|,\s*)blocks_type=([^,]+)/);
    if (trunkMatch) rows.push(["Trunk", escapeText(trunkMatch[1].trim())]);
    if (archStr) rows.push(["Arch", escapeText(archStr)]);
    if (m.rl_epoch != null) rows.push(["RL epoch", String(m.rl_epoch)]);
    if (m.step != null) rows.push(["Step", String(m.step)]);
    if (m.graft) rows.push(["Graft", m.graft === "pre" ? "pre (≤e6, expanded)" : "post (≥e7)"]);
    if (m.stv_horizons && m.stv_horizons.length) rows.push(["STV horizons", m.stv_horizons.join(", ")]);
    rows.push(["Moves-left", m.has_moves_left ? `yes (cap ${m.moves_left_cap != null ? m.moves_left_cap : "?"})` : "no"]);
    if (m.expanded_value) rows.push(["Expanded value", "yes"]);
    if (m.expanded_stv && m.expanded_stv.length) rows.push(["Expanded STV", `${m.expanded_stv.length} heads`]);
    if (m.zeroed_feature_cols && m.zeroed_feature_cols.length) rows.push(["Zeroed cols", m.zeroed_feature_cols.join(", ")]);
    if (m.candidate_radius != null) rows.push(["Cand. radius", String(m.candidate_radius)]);
    if (m.param_count != null) rows.push(["Params", `${(m.param_count / 1e6).toFixed(2)}M`]);
    if (info.size != null) rows.push(["File", formatBytes(info.size)]);
    if (info.mtime != null) rows.push(["Modified", new Date(info.mtime * 1000).toLocaleString()]);
    if (m.load_warnings && m.load_warnings.length) rows.push(["Warnings", escapeText(m.load_warnings.join("; "))]);
  }
  return rows.map(([k, v]) => `<div class="info-row"><span class="label">${k}</span><span class="value">${v}</span></div>`).join("");
}

// ---- bottom dock ------------------------------------------------------------------

function dbgSetDockTab(tab) {
  dbg.dockTab = tab;
  document.querySelectorAll("#dbgDockTabs [data-dock-tab]").forEach(b => b.classList.toggle("active", b.dataset.dockTab === tab));
  const chart = dbgEl("dbgDockChart");
  if (chart) chart.classList.toggle("active", tab === "trajectory");
  const sweep = dbgEl("dbgCkptSweep");
  if (sweep) sweep.classList.toggle("active", tab === "ckptsweep");
}

function dbgRenderDockChart() {
  // Trajectory dock (M9/S11): sweep-fed when a Game Error Sweep exists for the
  // current (game, ckptA) — value_p0 + KL second axis + mismatch ticks +
  // clickable blunder markers; otherwise the legacy /trajectory plot.
  const el = dbgEl("dbgDockChart");
  if (!el) return;
  const nav = dbg.nav;
  const sweep = dbgSweepData();
  const t = dbg.trajectoryKey === `${nav.run}|${nav.path}|${nav.rec}|${nav.ckptA}` ? dbg.trajectory : null;
  const reeval = sweep ? sweep.plies : ((t && t.reeval) || []);
  if (!reeval.length) {
    el.innerHTML = `<div class="dbg-empty-note">Sweep the game (or Plot) to fill the per-ply value / KL chart</div>`;
    el.__dbgChart = null;
    return;
  }
  const W = 1000, H = 220, padL = 36, padR = 34, padT = 14, padB = 22;
  const total = (sweep && sweep.total) || (t && t.total) || (reeval[reeval.length - 1].ply + 1) || 1;
  const x = ply => padL + (W - padL - padR) * (total ? ply / total : 0);
  const y = v => padT + (H - padT - padB) * (1 - (v + 1) / 2);  // v in [-1,1] -> top=+1
  const linePath = (pts, key) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.ply).toFixed(1)},${y(p[key]).toFixed(1)}`).join("");

  let html = "";
  html += `<line x1="${padL}" y1="${y(0)}" x2="${W - padR}" y2="${y(0)}" stroke="#2c3d50" stroke-width="1"></line>`;
  html += `<text x="4" y="${y(1) + 4}" fill="#5a6b7a" font-size="11">+1</text>`;
  html += `<text x="6" y="${y(0) + 4}" fill="#5a6b7a" font-size="11">0</text>`;
  html += `<text x="4" y="${y(-1) + 2}" fill="#5a6b7a" font-size="11">−1</text>`;
  if (nav.ply != null) {
    const px = x(Math.min(nav.ply, total));
    html += `<line x1="${px}" y1="${padT}" x2="${px}" y2="${H - padB}" stroke="var(--accent)" stroke-width="1" opacity="0.4" stroke-dasharray="3 3"></line>`;
  }
  if (t && t.recorded && t.recorded.length) {
    html += `<path d="${linePath(t.recorded, "root_value_p0")}" fill="none" stroke="var(--yellow)" stroke-width="1.6" opacity="0.85"></path>`;
  }
  // Policy-KL series on a second (right) axis — sweep only (S11; hidden without).
  let maxKl = 0;
  const klPts = sweep ? sweep.plies.filter(p => p.kl != null) : [];
  for (const p of klPts) maxKl = Math.max(maxKl, p.kl);
  if (klPts.length && maxKl > 0) {
    const yk = v => padT + (H - padT - padB) * (1 - v / maxKl);
    html += `<path d="${klPts.map((p, i) => `${i ? "L" : "M"}${x(p.ply).toFixed(1)},${yk(p.kl).toFixed(1)}`).join("")}" fill="none" stroke="#b07ce8" stroke-width="1.4" opacity="0.8"></path>`;
    html += `<text x="${W - padR + 3}" y="${padT + 8}" fill="#b07ce8" font-size="10">KL ${maxKl.toFixed(2)}</text>`;
    html += `<text x="${W - padR + 3}" y="${H - padB}" fill="#b07ce8" font-size="10">0</text>`;
  }
  html += `<path d="${linePath(reeval, "value_p0")}" fill="none" stroke="var(--accent)" stroke-width="2"></path>`;
  if (sweep) {
    for (const p of sweep.plies) {
      if (p.top1_match === false) {
        html += `<line x1="${x(p.ply).toFixed(1)}" y1="${H - padB}" x2="${x(p.ply).toFixed(1)}" y2="${H - padB - 6}" stroke="var(--yellow)" stroke-width="1.5" opacity="0.8"></line>`;
      }
    }
    for (const bp of dbgBlunders(sweep)) {
      const row = sweep.byPly.get(bp);
      if (!row) continue;
      html += `<circle data-dbg-jump="${bp}" cx="${x(bp).toFixed(1)}" cy="${y(row.value_p0).toFixed(1)}" r="5" fill="var(--p1)" opacity="0.85" style="cursor:pointer"><title>blunder ply ${bp} — click to jump</title></circle>`;
    }
  }
  html += `<line class="dbg-dock-cross" x1="0" y1="${padT}" x2="0" y2="${H - padB}" stroke="#dfe9f3" stroke-width="1" opacity="0" pointer-events="none"></line>`;
  const legend = sweep
    ? "value_p0 accent · KL purple (right axis) · top-1 miss ticks yellow · blunders red (click to jump)"
    : (t && t.stride > 1 ? `(every ${t.stride} plies · re-eval accent, recorded yellow)` : "re-eval accent · recorded yellow");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${html}</svg>`
    + `<div class="dbg-dock-readout dbg-muted">${legend}</div>`;
  const byPly = new Map();
  for (const p of reeval) byPly.set(p.ply, p);
  el.__dbgChart = { W, padL, padR, total, byPly, legend };
}

// ---- stale dots (spec M14) -----------------------------------------------------

function dbgSetStale(panelId, stale) {
  const panel = dbgEl(panelId);
  if (!panel) return;
  const dot = panel.querySelector(".dbg-stale");
  if (dot) dot.hidden = !stale;
}

function dbgUpdateStaleDots() {
  const nav = dbg.nav;
  const ready = nav.run && nav.path && nav.ply != null;
  const cur = ready ? dbgCurrentKey() : "";
  const curB = ready && nav.ckptB ? dbgCacheKey(nav, nav.ckptB) : "";
  const posStale = Boolean(dbg.position) && dbg.keys.position !== cur;
  dbgSetStale("dbgTabHeads", Boolean(dbg.analysis) && (dbg.keys.analysis !== cur || posStale));
  dbgSetStale("dbgTabSearch", (Boolean(dbg.search) && dbg.keys.search !== cur) || (Boolean(dbg.tree) && dbg.keys.tree !== cur));
  dbgSetStale("dbgTabTargets", posStale);  // targets render per-key, so only a stale position misleads
  dbgSetStale("dbgTabCompare", Boolean(nav.ckptB) && Boolean(dbg.analysisB) && dbg.keys.analysisB !== curB);
  dbgSetStale("dbgTabInputs", posStale);
  dbgSetStale("dbgTabCkpt", false);     // checkpoint provenance is position-independent
}

function dbgRefreshPanel(panelId) {
  if (panelId === "dbgTabHeads") dbgAnalyzeNow(true);
  else if (panelId === "dbgTabSearch") dbgRunSearch();
  else if (panelId === "dbgTabCompare") {
    if (!dbg.nav.ckptB) {
      dbgStub("Select checkpoint B in the context strip first.");
      return;
    }
    const entryB = dbgCacheEntry(dbgCacheKey(dbg.nav, dbg.nav.ckptB));
    delete entryB.analysis;
    delete entryB.analysisError;
    dbgEnsureAnalysisB();
  } else if (panelId === "dbgTabCkpt") dbgEnsureCkptInfo(true);
  else if (panelId === "dbgTabTargets") dbgEnsureRecordRow(true);
  else if (panelId === "dbgTabInputs") dbgEnsureInputs(true);
}

// ---- board view (pan/zoom, ported from the Match board on debug-local state) ----

function dbgApplyView() {
  if (dbg.view && debugBoardSvg) {
    debugBoardSvg.setAttribute("viewBox", `${dbg.view.x} ${dbg.view.y} ${dbg.view.width} ${dbg.view.height}`);
    const svgB = dbg.split ? dbgEl("dbgBoardSvgB") : null;
    if (svgB) svgB.setAttribute("viewBox", `${dbg.view.x} ${dbg.view.y} ${dbg.view.width} ${dbg.view.height}`);
  }
}

function dbgSyncView(base) {
  dbg.baseView = base;
  if (!dbg.view || !dbg.viewDirty) dbg.view = { ...base };
  dbgApplyView();
}

function dbgZoom(factor, anchor) {
  if (!dbg.view) return;
  const base = dbg.baseView || dbg.view;
  const nextWidth = clamp(dbg.view.width * factor, base.width * 0.14, base.width * 4.2);
  const scale = nextWidth / dbg.view.width;
  const nextHeight = dbg.view.height * scale;
  const point = anchor || {
    x: dbg.view.x + dbg.view.width / 2,
    y: dbg.view.y + dbg.view.height / 2,
  };
  dbg.view = {
    x: point.x - (point.x - dbg.view.x) * scale,
    y: point.y - (point.y - dbg.view.y) * scale,
    width: nextWidth,
    height: nextHeight,
  };
  dbg.viewDirty = true;
  dbgApplyView();
}

function dbgZoomAtCenter(factor) {
  if (!dbg.view) return;
  dbgZoom(factor, {
    x: dbg.view.x + dbg.view.width / 2,
    y: dbg.view.y + dbg.view.height / 2,
  });
}

function dbgZoomReset() {
  dbg.viewDirty = false;
  if (dbg.baseView) dbg.view = { ...dbg.baseView };
  dbgApplyView();
}

function dbgClientToBoard(clientX, clientY) {
  const matrix = debugBoardSvg && debugBoardSvg.getScreenCTM();
  if (!matrix || !dbg.view) {
    return {
      x: dbg.view ? dbg.view.x + dbg.view.width / 2 : 0,
      y: dbg.view ? dbg.view.y + dbg.view.height / 2 : 0,
    };
  }
  const point = debugBoardSvg.createSVGPoint();
  point.x = clientX;
  point.y = clientY;
  return point.matrixTransform(matrix.inverse());
}

// ---- navigation helpers -------------------------------------------------------

function dbgStepPly(delta) {
  // If a press does nothing, say WHY (the usual cause is the selected game
  // failing to load) instead of silently returning.
  if (!dbg.position && dbg.nav.ply == null) {
    debugSetStatus("No position loaded — pick a game with completed records.", "error");
    return;
  }
  const total = dbgRecordedTotal();
  const cur = dbg.nav.ply != null ? dbg.nav.ply : (total || 0);
  dbgGotoPly(cur + delta);
}

function dbgGotoPly(ply, { replace = false } = {}) {
  const total = dbgRecordedTotal();
  const clamped = Math.max(0, total != null ? Math.min(ply, total) : ply);
  if (clamped === dbg.nav.ply && !dbg.nav.acts.length) return;
  // Explicit ply navigation returns to the RECORDED game: re-basing the injected
  // tail onto a different prefix is never meaningful (the injected cells may be
  // occupied/illegal there) — chips, U and G are the way back out of a branch.
  dbgNavigate({ ply: clamped, acts: [] }, { replace });
}

function dbgStepRecord(delta) {
  if (!dbg.records.length) return;
  const next = Math.max(0, Math.min(dbg.nav.rec + delta, dbg.records.length - 1));
  if (next !== dbg.nav.rec) dbgNavigate({ rec: next, ply: null, acts: [] });
}

function dbgStepFile(delta) {
  if (!dbg.games.length) return;
  const idx = dbg.games.findIndex(g => g.path === dbg.nav.path);
  const next = Math.max(0, Math.min((idx === -1 ? 0 : idx) + delta, dbg.games.length - 1));
  const g = dbg.games[next];
  if (g && g.path !== dbg.nav.path) dbgNavigate({ path: g.path, rec: 0, ply: null, acts: [] });
}

function dbgInjectCell(q, r) {
  // WHAT-IF INJECTION (spec M4): clicking an empty legal cell appends its action
  // to the branch prefix; the server replays + re-analyzes the new position.
  if (dbg.suppressClick) return;
  const cell = dbg.cellIndex.get(`${q},${r}`);
  if (!cell || cell.placement || !dbg.position) return;
  if (!cell.legal) {
    debugSetStatus(`${q},${r} is not a legal cell here.`, "error");
    return;
  }
  diagTap(`inject ${q},${r}`);
  dbgNavigate({ acts: dbg.nav.acts.concat([dbgPackActionId(q, r)]) });
}

function dbgClearToggles() {
  dbg.overlays = { threats: false, numbers: false, last: false, legalDim: false };
  for (const [id, key] of [["dbgTglThreats", "threats"], ["dbgTglNumbers", "numbers"], ["dbgTglLast", "last"], ["dbgTglLegalDim", "legalDim"]]) {
    const el = dbgEl(id);
    if (el) el.checked = dbg.overlays[key];
  }
  // Repaint here: the follow-up dbgNavigate is a same-hash no-paint when the
  // base mode is unchanged (Shift+digit solo / 0 with the mode already set).
  dbgRenderBoard();
}

function dbgToggleLog() {
  dbg.logScale = !dbg.logScale;
  const el = dbgEl("dbgLegendLog");
  if (el) el.checked = dbg.logScale;
  dbgRenderBoard();
}

function dbgToggleCollapse(id) {
  const el = dbgEl(id);
  if (el) el.classList.toggle("dbg-collapsed");
}

// ---- keyboard layer (spec §1.7 / M13) -------------------------------------------

function dbgHandleKey(e) {
  if (activeScreen !== "debug") return;
  const help = dbgEl("dbgHelp");
  const palette = dbgEl("dbgPalette");
  if (e.key === "Escape") {
    if (palette && !palette.hidden) {
      palette.hidden = true;
      e.preventDefault();
      return;
    }
    if (help && !help.hidden) {
      help.hidden = true;
      e.preventDefault();
      return;
    }
    return;
  }
  if (palette && !palette.hidden) return;  // palette owns the keyboard while open
  const ae = document.activeElement;
  if (ae && /^(INPUT|SELECT|TEXTAREA)$/.test(ae.tagName)) return;  // M13: arrows edit the input, never the ply
  const nav = dbg.nav;
  const k = e.key;
  // Never hijack browser/system chords (Ctrl+C copy, Ctrl+A select-all, …);
  // Ctrl+K is the one modified shortcut the spec assigns (command palette).
  if ((e.ctrlKey || e.metaKey || e.altKey) && k.toLowerCase() !== "k") return;
  let handled = true;
  if (k === "?") {
    if (help) help.hidden = !help.hidden;
  } else if (k.toLowerCase() === "k" && !e.altKey && !e.metaKey) {
    dbgOpenPalette();
  } else if (e.code && e.code.startsWith("Digit") && !e.ctrlKey && !e.altKey && !e.metaKey) {
    const num = Number(e.code.slice(5));
    if (num === 0) {
      dbgClearToggles();  // 0 = mode none + clear additive toggles
      dbgClearCmpHeat();
      dbgNavigate({ mode: "none" }, { replace: true });
    } else if (num >= 1 && num <= 8) {
      if (e.shiftKey) dbgClearToggles();  // Shift+digit = solo
      dbgClearCmpHeat();
      dbgNavigate({ mode: DBG_MODE_ORDER[num - 1] }, { replace: true });
    } else {
      handled = false;
    }
  } else if (k === "ArrowLeft" || k === "ArrowRight") {
    if (e.shiftKey) dbgStepBlunder(k === "ArrowLeft" ? -1 : 1);
    else dbgStepPly(k === "ArrowLeft" ? -1 : 1);
  } else if (k === "Home") {
    dbgGotoPly(0);
  } else if (k === "End") {
    dbgGotoPly(1e9);
  } else if (k === "[") {
    dbgStepRecord(-1);
  } else if (k === "]") {
    dbgStepRecord(1);
  } else if (k === "{") {
    dbgStepFile(-1);
  } else if (k === "}") {
    dbgStepFile(1);
  } else if (k === "a" || k === "A") {
    dbgAnalyzeNow(true);
  } else if (k === "s" || k === "S") {
    dbgRunSearch();
  } else if (k === "t" || k === "T") {
    // §1.7 "T  run / toggle debug tree": with a fresh tree for this key, toggle
    // its PV ghosts; otherwise run (dbgRunTree stubs while one is in flight).
    if (!dbg.treeBusy && dbgFreshData("tree")) {
      dbg.treeGhostsOff = !dbg.treeGhostsOff;
      dbg.treePreview = null;
      dbgRenderBoard();
    } else {
      dbgRunTree();
    }
  } else if (k === "l" || k === "L") {
    dbgToggleLog();
  } else if (k === "u" || k === "U") {
    if (nav.acts.length) dbgNavigate({ acts: nav.acts.slice(0, -1) });
  } else if (k === "g" || k === "G") {
    if (nav.acts.length) dbgNavigate({ acts: [] });
  } else if (k === "p" || k === "P") {
    dbgAddPin();
  } else if (k === "c" || k === "C") {
    dbgNavigate({ tab: "compare" }, { replace: true });
  } else if (k === "i" || k === "I") {
    dbgNavigate({ tab: "inputs" }, { replace: true });
  } else {
    handled = false;
  }
  if (handled) e.preventDefault();
}

// ---- events --------------------------------------------------------------------

function debugBindEvents() {
  const root = dbgEl("debugScreen");
  if (!root || root.__dbgDelegated) return;  // bind once on the stable ancestor
  root.__dbgDelegated = true;

  // The ply strip renders row geometry for the current orientation (vertical
  // rail on desktop, horizontal 22px strip <=900px) — rebuild it on breakpoint
  // crossings so the inline top/height vs left/width styles stay correct.
  if (window.matchMedia) {
    const mq = window.matchMedia("(max-width: 900px)");
    const onOrientation = () => {
      try {
        dbgRenderPlyRail();
      } catch (e) {
        reportError("dbgRenderPlyRail: " + (e && (e.stack || e.message) || e));
      }
    };
    if (mq.addEventListener) mq.addEventListener("change", onOrientation);
    else if (mq.addListener) mq.addListener(onOrientation);
  }

  // BUTTONS via event DELEGATION on #debugScreen (which is never rebuilt), plus a
  // touchend fallback. Per-button click listeners failed on the owner's phone:
  // controls trapped in a fixed-height overflow:hidden panel micro-scrolled under
  // the finger, so the browser classified the tap as a scroll and never promoted
  // touchend -> click. Delegation + an explicit touchend handler guarantees the
  // action fires on tap; everything routes through reportError so a throw is
  // visible on-device.
  const ACTIONS = {
    dbgPlyFirst: { fn: () => dbgGotoPly(0), tap: "|< first" },
    dbgPlyPrev: { fn: () => dbgStepPly(-1), tap: "< prev" },
    dbgPlyNext: { fn: () => dbgStepPly(1), tap: "> next" },
    dbgPlyLast: { fn: () => dbgGotoPly(1e9), tap: ">| last" },
    dbgPlySweepBtn: { fn: () => dbgRunSweep(), tap: "sweep" },
    dbgSweepBtn: { fn: () => dbgRunSweep(), tap: "sweep" },
    dbgTrajBtn: { fn: () => dbgPlotTrajectory(), tap: "plot" },
    dbgSearchRun: { fn: () => dbgRunSearch(), tap: "search" },
    dbgTreeRun: { fn: () => dbgRunTree(), tap: "tree" },
    dbgLadderRun: { fn: () => dbgRunLadder(), tap: "ladder" },
    dbgCmpTreeB: { fn: () => dbgRunTree({ checkpoint: dbg.nav.ckptB }), tap: "tree B" },
    dbgCkptSweepRun: { fn: () => dbgRunCkptSweep(), tap: "ckpt sweep" },
    dbgCtxRefresh: { fn: () => dbgRefreshSources(), tap: "refresh" },
    dbgCtxCollapse: { fn: () => dbgToggleCollapse("dbgCtx"), tap: "ctx collapse" },
    dbgDockToggle: { fn: () => dbgToggleCollapse("dbgDock"), tap: "dock collapse" },
    dbgZoomIn: { fn: () => dbgZoomAtCenter(0.82), tap: "zoom in" },
    dbgZoomOut: { fn: () => dbgZoomAtCenter(1.22), tap: "zoom out" },
    dbgZoomReset: { fn: () => dbgZoomReset(), tap: "zoom reset" },
    dbgBranchReturn: { fn: () => dbgNavigate({ acts: [] }), tap: "return to game" },
    dbgPinAdd: { fn: () => dbgAddPin(), tap: "pin" },
    dbgPinExport: { fn: () => dbgExportPins(), tap: "pin export" },
    dbgPaletteBtn: { fn: () => dbgOpenPalette(), tap: "palette" },
    dbgHelpBtn: { fn: () => { const h = dbgEl("dbgHelp"); if (h) h.hidden = !h.hidden; }, tap: "help" },
    dbgBlunderPrev: { fn: () => dbgStepBlunder(-1), tap: "blunder prev" },
    dbgBlunderNext: { fn: () => dbgStepBlunder(1), tap: "blunder next" },
    dbgCmpHeat: { fn: () => dbgToggleCmpHeat(), tap: "cmp heat" },
    dbgCkptSweepStop: {
      fn: () => {
        if (dbg.ckptSweep && dbg.ckptSweep.running) dbg.ckptSweep.abort = true;
        else dbgStub("No checkpoint sweep running.");
      },
      tap: "ckpt sweep stop",
    },
  };
  const fire = ev => {
    const t = ev.target;
    if (!t || !t.closest) return;
    const idBtn = t.closest("button[id]");
    const act = idBtn && ACTIONS[idBtn.id];
    if (act) {
      ev.preventDefault();
      diagTap(act.tap);
      try {
        act.fn();
      } catch (e) {
        reportError("dbg " + idBtn.id + ": " + (e && (e.stack || e.message) || e));
      }
      return;
    }
    const chipX = t.closest(".dbg-chip-x");
    if (chipX) {
      ev.preventDefault();
      diagTap("undo chip");
      dbgNavigate({ acts: dbg.nav.acts.slice(0, Number(chipX.dataset.actCut) || 0) });
      return;
    }
    const treeStep = t.closest(".dbg-tree-step");
    if (treeStep) {
      ev.preventDefault();
      diagTap("tree step-into");
      // M11: re-base via injection — each line move becomes a branch chip.
      dbgNavigate({ acts: dbg.nav.acts.concat(dbgTreePathIds(treeStep.dataset.treePath)) });
      return;
    }
    const treeExpand = t.closest(".dbg-tree-expand");
    if (treeExpand) {
      ev.preventDefault();
      diagTap("tree expand");
      const p = treeExpand.dataset.treePath;
      dbgRunTree({ rootActions: dbgTreePathIds(p), graftPath: p });
      return;
    }
    const treeRow = t.closest(".dbg-tree-row");
    if (treeRow) {
      ev.preventDefault();
      dbgTreeRowClick(treeRow.dataset.treePath);
      return;
    }
    const jump = t.closest("[data-dbg-jump]");
    if (jump) {
      ev.preventDefault();
      diagTap("jump ply " + jump.dataset.dbgJump);
      dbgGotoPly(Number(jump.dataset.dbgJump));
      return;
    }
    const pinX = t.closest(".dbg-pin-x");
    if (pinX) {
      ev.preventDefault();
      dbgRemovePin(Number(pinX.dataset.pin));
      return;
    }
    const pinNote = t.closest(".dbg-pin-note");
    if (pinNote) {
      ev.preventDefault();
      dbgNotePin(Number(pinNote.dataset.pin));
      return;
    }
    const pinChip = t.closest(".dbg-pin-chip");
    if (pinChip) {
      ev.preventDefault();
      dbgRestorePin(Number(pinChip.dataset.pin));
      return;
    }
    const journalRow = t.closest("[data-journal]");
    if (journalRow) {
      ev.preventDefault();
      const e2 = dbg.journal[Number(journalRow.dataset.journal)];
      if (e2 && e2.nav) dbgNavigate(Object.assign({}, e2.nav, { acts: ((e2.nav && e2.nav.acts) || []).slice() }));
      return;
    }
    const palItem = t.closest("[data-pal]");
    if (palItem) {
      ev.preventDefault();
      dbgPaletteExec(Number(palItem.dataset.pal));
      return;
    }
    const refresh = t.closest(".dbg-panel-refresh");
    if (refresh) {
      ev.preventDefault();
      const panel = refresh.closest(".dbg-tab-panel");
      if (panel) dbgRefreshPanel(panel.id);
      return;
    }
    const modeBtn = t.closest("#dbgModeBar [data-mode]");
    if (modeBtn) {
      ev.preventDefault();
      diagTap("mode " + modeBtn.dataset.mode);
      // Parity with the 0 key (§1.7): None = mode none + clear additive toggles.
      if (modeBtn.dataset.mode === "none") dbgClearToggles();
      dbgClearCmpHeat();  // picking a base mode dismisses the cmp Δ overlay
      dbgNavigate({ mode: modeBtn.dataset.mode }, { replace: true });
      return;
    }
    const tabBtn = t.closest("#dbgTabs [data-tab]");
    if (tabBtn) {
      ev.preventDefault();
      diagTap("tab " + tabBtn.dataset.tab);
      dbgNavigate({ tab: tabBtn.dataset.tab }, { replace: true });
      return;
    }
    const dockTab = t.closest("#dbgDockTabs [data-dock-tab]");
    if (dockTab) {
      ev.preventDefault();
      dbgSetDockTab(dockTab.dataset.dockTab);
      return;
    }
    const journalClear = t.closest(".dbg-journal-clear");
    if (journalClear) {
      ev.preventDefault();
      diagTap("journal clear");
      dbgClearJournal();
      return;
    }
    const sortBtn = t.closest("[data-dbg-sort]");
    if (sortBtn) {
      ev.preventDefault();
      dbg.sortKey = sortBtn.dataset.dbgSort;
      dbgRenderHeads();
      return;
    }
    const plyRow = t.closest(".dbg-ply-row");
    if (plyRow) {
      ev.preventDefault();
      diagTap("strip ply " + plyRow.dataset.ply);
      dbgGotoPly(Number(plyRow.dataset.ply));
      return;
    }
    const cellEl = t.closest(".dbg-cell");
    if (cellEl) {
      ev.preventDefault();
      dbgInjectCell(Number(cellEl.dataset.q), Number(cellEl.dataset.r));
      return;
    }
    if (t.classList && t.classList.contains("dbg-overlay")) {
      ev.preventDefault();
      t.hidden = true;  // click outside the card closes help/palette
    }
  };
  let lastTouchFire = 0;
  root.addEventListener("touchend", ev => {
    lastTouchFire = Date.now();
    fire(ev);
  }, { passive: false });
  root.addEventListener("click", ev => {
    if (Date.now() - lastTouchFire < 700) return;  // touchend already handled this tap
    fire(ev);
  });

  // Hover HUD: ONE delegated mousemove on the SVG + the per-render metrics Map.
  if (debugBoardSvg) {
    debugBoardSvg.addEventListener("mousemove", ev => {
      const cellEl = ev.target && ev.target.closest && ev.target.closest(".dbg-cell");
      if (cellEl) dbgHoverCell(Number(cellEl.dataset.q), Number(cellEl.dataset.r));
    });
    debugBoardSvg.addEventListener("mouseleave", () => {
      const hud = dbgEl("debugBoardHud");
      if (hud) hud.innerHTML = `<div>Hover a cell</div>`;
    });
  }

  // Pan / pinch / wheel on the debug board area (debug-local state only).
  const area = debugBoardSvg && debugBoardSvg.closest(".board-area");
  if (area && !area.__dbgViewBound) {
    area.__dbgViewBound = true;
    const pointers = new Map();
    let drag = null;
    let pinch = null;
    const beginPan = ev => {
      const rect = debugBoardSvg.getBoundingClientRect();
      drag = {
        pointerId: ev.pointerId,
        clientX: ev.clientX,
        clientY: ev.clientY,
        scaleX: dbg.view.width / Math.max(1, rect.width),
        scaleY: dbg.view.height / Math.max(1, rect.height),
        view: { ...dbg.view },
        moved: false,
      };
    };
    area.addEventListener("wheel", ev => {
      if (!dbg.view || ev.target.closest(".board-view-controls")) return;
      ev.preventDefault();
      dbgZoom(ev.deltaY < 0 ? 0.88 : 1.14, dbgClientToBoard(ev.clientX, ev.clientY));
    }, { passive: false });
    area.addEventListener("pointerdown", ev => {
      if (!dbg.view || (ev.pointerType === "mouse" && ev.button !== 0)) return;
      if (ev.target.closest(".board-view-controls")) return;
      pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
      area.setPointerCapture(ev.pointerId);
      if (pointers.size >= 2) {
        drag = null;  // a second finger upgrades the gesture to pinch-zoom
        const pts = [...pointers.values()].slice(0, 2);
        const mid = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };
        pinch = {
          rect: debugBoardSvg.getBoundingClientRect(),
          startDist: Math.max(1, Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y)),
          startView: { ...dbg.view },
          anchorBoard: dbgClientToBoard(mid.x, mid.y),
        };
        dbg.suppressClick = true;
      } else {
        beginPan(ev);
      }
    });
    area.addEventListener("pointermove", ev => {
      if (!pointers.has(ev.pointerId)) return;
      ev.preventDefault();
      pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
      if (pinch && pointers.size >= 2) {
        const pts = [...pointers.values()].slice(0, 2);
        const dist = Math.max(1, Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y));
        const mid = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };
        const base = dbg.baseView || pinch.startView;
        const nextWidth = clamp(pinch.startView.width * (pinch.startDist / dist), base.width * 0.14, base.width * 4.2);
        const nextHeight = pinch.startView.height * (nextWidth / pinch.startView.width);
        const rect = pinch.rect;
        dbg.view = {
          width: nextWidth,
          height: nextHeight,
          x: pinch.anchorBoard.x - (mid.x - rect.left) * (nextWidth / Math.max(1, rect.width)),
          y: pinch.anchorBoard.y - (mid.y - rect.top) * (nextHeight / Math.max(1, rect.height)),
        };
        dbg.viewDirty = true;
        dbgApplyView();
      } else if (drag && ev.pointerId === drag.pointerId) {
        const dx = (ev.clientX - drag.clientX) * drag.scaleX;
        const dy = (ev.clientY - drag.clientY) * drag.scaleY;
        if (Math.hypot(ev.clientX - drag.clientX, ev.clientY - drag.clientY) > 4) drag.moved = true;
        dbg.view = { ...drag.view, x: drag.view.x - dx, y: drag.view.y - dy };
        dbg.viewDirty = true;
        dbgApplyView();
      }
    }, { passive: false });
    const endPointer = ev => {
      if (!pointers.has(ev.pointerId)) return;
      pointers.delete(ev.pointerId);
      if (area.hasPointerCapture(ev.pointerId)) area.releasePointerCapture(ev.pointerId);
      const moved = (drag && drag.moved) || Boolean(pinch);
      if (pointers.size < 2) pinch = null;
      if (pointers.size === 1) {
        // Dropped from pinch to a single finger — resume panning from it.
        const [pointerId, point] = [...pointers.entries()][0];
        beginPan({ pointerId, clientX: point.x, clientY: point.y });
        drag.moved = true;
      } else if (pointers.size === 0) {
        drag = null;
        if (moved) {
          dbg.suppressClick = true;
          window.setTimeout(() => {
            dbg.suppressClick = false;
          }, 80);
        } else {
          dbg.suppressClick = false;
        }
      }
    };
    area.addEventListener("pointerup", endPointer);
    area.addEventListener("pointercancel", endPointer);
  }

  // Selects / sliders / checkboxes keep direct change/input listeners.
  const on = (id, ev, fn) => {
    const el = dbgEl(id);
    if (el) el.addEventListener(ev, fn);
  };
  on("dbgCtxRun", "change", e => dbgNavigate({ run: e.target.value, path: "", rec: 0, ply: null, acts: [], ckptA: "", ckptB: "" }));
  on("dbgCtxSource", "change", e => dbgNavigate({ src: e.target.value, path: "", rec: 0, ply: null, acts: [] }));
  on("dbgCtxFile", "change", e => dbgNavigate({ path: e.target.value, rec: 0, ply: null, acts: [] }));
  on("dbgCtxRecord", "change", e => dbgNavigate({ rec: Number(e.target.value) || 0, ply: null, acts: [] }));
  on("dbgCtxCkptA", "change", e => dbgNavigate({ ckptA: e.target.value }));
  on("dbgCtxCkptB", "change", e => dbgNavigate({ ckptB: e.target.value }));
  // Slider: first input of a drag PUSHES (so Back returns to the pre-drag ply),
  // the rest REPLACE (no history spam); change ends the drag.
  let sliderDragging = false;
  on("dbgPlySlider", "input", e => {
    diagTap("slide " + e.target.value);
    dbgGotoPly(Number(e.target.value), { replace: sliderDragging });
    sliderDragging = true;
  });
  on("dbgPlySlider", "change", e => {
    sliderDragging = false;
    dbgGotoPly(Number(e.target.value), { replace: true });
  });
  on("dbgTglThreats", "change", e => {
    dbg.overlays.threats = e.target.checked;
    dbgRenderBoard();
  });
  on("dbgTglNumbers", "change", e => {
    dbg.overlays.numbers = e.target.checked;
    dbgRenderBoard();
  });
  on("dbgTglLast", "change", e => {
    dbg.overlays.last = e.target.checked;
    dbgRenderBoard();
  });
  on("dbgTglLegalDim", "change", e => {
    dbg.overlays.legalDim = e.target.checked;
    dbgRenderBoard();
  });
  on("dbgLegendLog", "change", e => {
    dbg.logScale = e.target.checked;
    dbgRenderBoard();
  });
  on("dbgOpacity", "input", e => {
    dbg.opacity = Number(e.target.value) || 0.9;
    dbgRenderBoard();
  });
  on("dbgPlaneSelect", "change", () => dbgRenderBoard());
  on("dbgCmpSplit", "change", e => dbgToggleSplit(e.target.checked));
  on("dbgPinImport", "change", e => {
    const file = e.target.files && e.target.files[0];
    e.target.value = "";
    dbgImportPins(file);
  });

  // Command palette (S1): the input owns the keyboard while open.
  on("dbgPaletteInput", "input", e => dbgRenderPaletteList(e.target.value));
  on("dbgPaletteInput", "keydown", e => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const n = (dbg.paletteShown || []).length;
      if (n) {
        dbg.paletteIdx = (dbg.paletteIdx + (e.key === "ArrowDown" ? 1 : n - 1)) % n;
        dbgRenderPaletteList(e.target.value);
      }
    } else if (e.key === "Enter") {
      e.preventDefault();
      dbgPaletteExec(dbg.paletteIdx);
    }
  });

  // Split-board (S5) shared hover: the B half reuses the same delegated handler.
  const svgB = dbgEl("dbgBoardSvgB");
  if (svgB) {
    svgB.addEventListener("mousemove", ev => {
      const cellEl = ev.target && ev.target.closest && ev.target.closest(".dbg-cell");
      if (cellEl) dbgHoverCell(Number(cellEl.dataset.q), Number(cellEl.dataset.r));
    });
  }

  // Trajectory dock hover crosshair (S11): per-render chart state on the
  // container; attribute updates only — no re-render in mousemove.
  const dock = dbgEl("dbgDockChart");
  if (dock) {
    dock.addEventListener("mousemove", ev => {
      const st = dock.__dbgChart;
      const svg = dock.querySelector("svg");
      if (!st || !svg) return;
      const rect = svg.getBoundingClientRect();
      const xv = (ev.clientX - rect.left) / Math.max(1, rect.width) * st.W;
      const ply = Math.max(0, Math.min(st.total, Math.round((xv - st.padL) / Math.max(1, st.W - st.padL - st.padR) * st.total)));
      const cross = svg.querySelector(".dbg-dock-cross");
      if (cross) {
        cross.setAttribute("x1", xv.toFixed(1));
        cross.setAttribute("x2", xv.toFixed(1));
        cross.setAttribute("opacity", "0.35");
      }
      const out = dock.querySelector(".dbg-dock-readout");
      const row = st.byPly.get(ply);
      if (out) {
        out.textContent = row
          ? `ply ${ply} · v_p0 ${row.value_p0 != null ? row.value_p0.toFixed(3) : "—"}`
            + `${row.kl != null ? ` · KL ${row.kl.toFixed(3)}` : ""}`
            + `${row.top1_match != null ? ` · top1 ${row.top1_match ? "✓" : "✗"}` : ""}`
            + `${row.value_err_z != null ? ` · err_z ${row.value_err_z >= 0 ? "+" : ""}${row.value_err_z.toFixed(2)}` : ""}`
          : `ply ${ply}`;
      }
    });
    dock.addEventListener("mouseleave", () => {
      const cross = dock.querySelector(".dbg-dock-cross");
      if (cross) cross.setAttribute("opacity", "0");
      const out = dock.querySelector(".dbg-dock-readout");
      if (out && dock.__dbgChart) out.textContent = dock.__dbgChart.legend || "";
    });
  }

  document.addEventListener("keydown", dbgHandleKey);
}

// ---- screen entry ----------------------------------------------------------------

async function enterDebugScreen() {
  showVersionTag();
  if (!dbg.inited) {
    dbg.inited = true;
    // Bind handlers in their OWN try so a later failure can never leave the nav
    // buttons unwired, surfaced to the on-screen banner (the real-phone failure
    // mode we otherwise can't see).
    try {
      debugBindEvents();
    } catch (e) {
      reportError("debugBindEvents: " + (e && (e.stack || e.message) || e));
    }
  }
  if (dbg.pendingDeepLink) {
    // navigateScreen() is mid-flight: when it is about to WRITE `#debug` the
    // resulting hashchange routes through dbgApplyHash (which consumes the deep
    // link); when the hash already IS `#debug...` no event will come — apply now.
    if (String(window.location.hash || "").startsWith("#debug")) dbgApplyHash();
    return;
  }
  dbgApplyHash();
}

// Debug-internal hash navigation (ply steps, injections, checkpoint flips). The
// global listener above (setScreen) handles screen routing; this one applies the
// nav state. dbgApplyHash dedupes, so the double delivery is harmless.
window.addEventListener("hashchange", () => {
  if (screenFromHash() === "debug") dbgApplyHash();
});

async function init() {
  setScreen(activeScreen, { preserveHash: true });
  await Promise.allSettled([loadAdapters(), loadState(), loadTrainingRuns()]);
  render();
}

init();
