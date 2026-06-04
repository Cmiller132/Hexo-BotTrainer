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
  return SCREENS.includes(hash) ? hash : "match";
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
  // Game-length stats (mean/median/max/stdev), appended when the producer emits
  // them (omitted otherwise, so dense_cnn runs are unaffected).
  const lenMean = asFinite(selfplay.game_length_mean);
  const lenMed = asFinite(selfplay.game_length_median);
  if (lenMean !== null || lenMed !== null) {
    selfplayText += ` | len μ${formatDecimal(lenMean, 1)} med ${formatDecimal(lenMed, 0)}`
      + ` max ${asFinite(selfplay.game_length_max) ?? "--"} σ${formatDecimal(selfplay.game_length_stdev, 1)}`;
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
// Debug tab — model inspection.
//
// Self-contained from the Match/History screens: its own board SVG and state,
// reusing only the pure geometry helpers (center/path/playerColor/HEX). All
// model outputs come from the CPU inference worker via /api/debug/* and are
// rendered here; nothing in this module touches the live-match polling.
// ===========================================================================

const dbg = {
  inited: false,
  loading: false,
  run: "",
  source: "selfplay",
  games: [],
  gameFile: "",
  records: [],
  record: 0,
  checkpoints: [],
  checkpoint: "",
  position: null,   // /api/debug/position payload
  analysis: null,   // /api/debug/analyze result
  search: null,     // /api/debug/search result
  compare: null,    // { checkpoint, analysis } for A/B
  compareCheckpoint: "",
  trajectory: null, // /api/debug/trajectory result
  imported: null,   // imported action-id list (overrides game when set)
  overlays: { policy: true, visits: false, opp: false, threats: false, numbers: true },
  pendingDeepLink: null,
};

const dbgEl = id => document.getElementById(id);
const debugBoardSvg = dbgEl("debugBoardSvg");

function debugSetStatus(message, kind = "info") {
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
    throw new Error((data && data.error) || `HTTP ${res.status}`);
  }
  return data;
}

async function enterDebugScreen() {
  if (!dbg.inited) {
    dbg.inited = true;
    debugBindEvents();
    await debugInit();  // consumes any pendingDeepLink itself
  } else if (dbg.pendingDeepLink) {
    const link = dbg.pendingDeepLink;
    dbg.pendingDeepLink = null;
    await debugApplyDeepLink(link);
  } else {
    debugRenderAll();
  }
}

async function debugInit() {
  debugSetStatus("Loading runs…");
  try {
    if (!trainingRuns.length) await loadTrainingRuns();
  } catch (_e) { /* fall through with whatever we have */ }
  const runNames = trainingRuns.map(r => r.name);
  // Prefer a deep-link target, else the run already selected on History, else the
  // first run that has a hexgt checkpoints dir, else the first run.
  let preferred = dbg.pendingDeepLink && dbg.pendingDeepLink.run;
  if (!preferred && historySelectedRun && runNames.includes(historySelectedRun)) preferred = historySelectedRun;
  dbg.run = preferred && runNames.includes(preferred) ? preferred : (runNames[0] || "");
  debugSyncRunSelect();
  await debugLoadRun();
  if (dbg.pendingDeepLink) {
    const link = dbg.pendingDeepLink;
    dbg.pendingDeepLink = null;
    await debugApplyDeepLink(link);
  }
  debugSetStatus("");
}

function debugSyncRunSelect() {
  const sel = dbgEl("debugRunSelect");
  if (!sel) return;
  sel.innerHTML = trainingRuns.length
    ? trainingRuns.map(r => `<option value="${escapeAttr(r.name)}">${escapeText(r.name)}</option>`).join("")
    : `<option value="">No runs</option>`;
  sel.value = dbg.run;
}

async function debugLoadRun() {
  await Promise.allSettled([debugLoadCheckpoints(), debugLoadGames()]);
}

async function debugLoadCheckpoints() {
  const sel = dbgEl("debugCheckpointSelect");
  if (!dbg.run) { dbg.checkpoints = []; if (sel) sel.innerHTML = ""; return; }
  try {
    const data = await debugFetchJson(`/api/debug/checkpoints?run=${encodeURIComponent(dbg.run)}`);
    dbg.checkpoints = data.checkpoints || [];
    if (sel) {
      sel.innerHTML = dbg.checkpoints.map(c => {
        const label = c.epoch != null ? `epoch ${c.epoch}` : c.name.replace(/\.pt$/, "");
        const graft = c.graft ? ` · ${c.graft}` : "";
        return `<option value="${escapeAttr(c.name)}">${escapeText(label + graft)}</option>`;
      }).join("");
    }
    if (!dbg.checkpoint || !dbg.checkpoints.some(c => c.name === dbg.checkpoint)) {
      const latest = dbg.checkpoints.find(c => c.latest) || dbg.checkpoints[0];
      dbg.checkpoint = latest ? latest.name : "";
    }
    if (sel) sel.value = dbg.checkpoint;
    const cmp = dbgEl("debugCompareSelect");
    if (cmp) {
      const prev = cmp.value;
      cmp.innerHTML = `<option value="">— none —</option>` + dbg.checkpoints.map(c => {
        const label = c.epoch != null ? `epoch ${c.epoch}` : c.name.replace(/\.pt$/, "");
        return `<option value="${escapeAttr(c.name)}">${escapeText(label)}</option>`;
      }).join("");
      cmp.value = dbg.checkpoints.some(c => c.name === prev) ? prev : "";
    }
    debugRenderCheckpointInfo();
  } catch (e) {
    debugSetStatus(`Checkpoints: ${e.message}`, "error");
  }
}

async function debugLoadGames() {
  const sel = dbgEl("debugGameSelect");
  if (!dbg.run) { dbg.games = []; if (sel) sel.innerHTML = ""; return; }
  try {
    const data = await debugFetchJson(`/api/debug/games?run=${encodeURIComponent(dbg.run)}&source=${encodeURIComponent(dbg.source)}`);
    dbg.games = data.games || [];
    if (sel) {
      sel.innerHTML = dbg.games.length
        ? dbg.games.map(g => `<option value="${escapeAttr(g.path)}">${escapeText(g.path)}</option>`).join("")
        : `<option value="">No ${dbg.source} games</option>`;
    }
    if (!dbg.gameFile || !dbg.games.some(g => g.path === dbg.gameFile)) {
      dbg.gameFile = dbg.games.length ? dbg.games[0].path : "";
      dbg.record = 0;
    }
    if (sel) sel.value = dbg.gameFile;
    if (dbg.gameFile) await debugLoadPosition({ resetPly: true });
    else { dbg.position = null; debugRenderAll(); }
  } catch (e) {
    debugSetStatus(`Games: ${e.message}`, "error");
  }
}

async function debugLoadPosition({ resetPly = false, ply = null } = {}) {
  if (resetPly) { dbg.trajectory = null; const note = dbgEl("debugTrajectoryNote"); if (note) note.textContent = ""; }
  if (dbg.imported) return debugLoadImported(ply);
  if (!dbg.run || !dbg.gameFile) return;
  const targetPly = ply != null ? ply : (resetPly ? null : (dbg.position && dbg.position.debug.ply));
  const params = new URLSearchParams({ run: dbg.run, path: dbg.gameFile, record: String(dbg.record) });
  if (targetPly != null) params.set("ply", String(targetPly));
  else params.set("ply", "0");
  try {
    const data = await debugFetchJson(`/api/debug/position?${params.toString()}`);
    if (targetPly == null && resetPly) {
      // Default to the final position so the whole game is visible at a glance.
      const total = data.debug.total;
      if (total > 0 && data.debug.ply !== total) return debugLoadPosition({ ply: total });
    }
    dbg.position = data;
    dbg.records = data.record_games || [];
    dbg.analysis = null;
    dbg.search = null;
    debugSyncRecordSelect();
    debugRenderAll();
    await debugAnalyze();
  } catch (e) {
    debugSetStatus(`Position: ${e.message}`, "error");
  }
}

async function debugLoadImported(ply = null) {
  // Build a synthetic position payload from an imported action-id list by asking
  // the position endpoint? Imported lists have no .hxr, so we render a minimal
  // board client-side via the analyze result's candidates + reconstructed stones.
  const ids = dbg.imported;
  const targetPly = ply != null ? Math.max(0, Math.min(ply, ids.length)) : ids.length;
  dbg.position = {
    placements: debugStonesFromActions(ids.slice(0, targetPly)),
    legal: [],
    mode: "debug-import",
    debug: { ply: targetPly, total: ids.length, action_ids: ids, last_action_id: targetPly > 0 ? ids[targetPly - 1] : null, winner: null, imported: true },
    record_games: [],
  };
  dbg.records = [];
  dbg.analysis = null;
  dbg.search = null;
  debugRenderAll();
  await debugAnalyze();
}

function debugStonesFromActions(ids) {
  // action_id -> (q,r) via the same packing the engine uses (low 16 bits each,
  // sign-extended). Mirrors hexo_engine.types.unpack_coord_id for display only.
  const stones = [];
  ids.forEach((aid, i) => {
    const { q, r } = debugUnpackCoord(aid);
    stones.push({ q, r, player: i % 2 === 0 ? "player0" : "player1", index: i + 1 });
  });
  return stones;
}

function debugUnpackCoord(actionId) {
  const raw = Number(actionId) >>> 0;
  const toS16 = v => (v & 0x8000) ? v - 0x10000 : v;
  return { q: toS16(raw & 0xffff), r: toS16((raw >>> 16) & 0xffff) };
}

function debugActionPrefix() {
  // The move prefix corresponding to the displayed ply (for analyze/search).
  if (!dbg.position) return null;
  const dbgInfo = dbg.position.debug;
  return (dbgInfo.action_ids || []).slice(0, dbgInfo.ply);
}

function debugRequestBody() {
  const prefix = debugActionPrefix();
  if (prefix == null) return null;
  const body = { run: dbg.run, checkpoint: dbg.checkpoint };
  if (dbg.imported || (dbg.position && dbg.position.debug.imported)) {
    body.action_ids = prefix;
  } else {
    body.path = dbg.gameFile;
    body.record = dbg.record;
    body.ply = dbg.position.debug.ply;
  }
  return body;
}

async function debugAnalyze() {
  const body = debugRequestBody();
  if (!body || !dbg.checkpoint) { debugRenderAll(); return; }
  dbg.loading = true;
  debugSetStatus("Evaluating position on CPU…", "busy");
  debugRenderAll();
  try {
    dbg.analysis = await debugFetchJson("/api/debug/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    debugSetStatus("");
  } catch (e) {
    dbg.analysis = null;
    debugSetStatus(`Analyze: ${e.message}`, "error");
  } finally {
    dbg.loading = false;
    debugRenderAll();
  }
  if (dbg.compareCheckpoint) debugRunCompare();
}

async function debugRunSearch() {
  const body = debugRequestBody();
  if (!body || !dbg.checkpoint) return;
  const visitsEl = dbgEl("debugSearchVisits");
  body.visits = Math.max(1, Math.min(20000, parseInt(visitsEl && visitsEl.value, 10) || 512));
  dbg.loading = true;
  debugSetStatus(`Running ${body.visits}-visit CPU search…`, "busy");
  debugRenderSearch();
  try {
    dbg.search = await debugFetchJson("/api/debug/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    debugSetStatus("");
  } catch (e) {
    dbg.search = null;
    debugSetStatus(`Search: ${e.message}`, "error");
  } finally {
    dbg.loading = false;
    debugRenderAll();
  }
}

// ---- deep link (from History "Open in Debug") -----------------------------

async function debugApplyDeepLink({ run, path, record, ply }) {
  if (run && trainingRuns.some(r => r.name === run)) {
    dbg.run = run;
    debugSyncRunSelect();
    await debugLoadRun();
  }
  // Pick the source (selfplay/evaluation) that contains the file.
  if (path) {
    const inSource = src => dbg.games.some(g => g.path === path);
    if (!inSource()) {
      dbg.source = path.startsWith("eval") ? "evaluation" : "selfplay";
      const sel = dbgEl("debugSourceSelect"); if (sel) sel.value = dbg.source;
      await debugLoadGames();
    }
    dbg.gameFile = path;
    dbg.record = Number(record) || 0;
    const gsel = dbgEl("debugGameSelect"); if (gsel) gsel.value = dbg.gameFile;
    await debugLoadPosition({ ply: ply != null ? Number(ply) : null });
  }
}

function debugOpenFromHistory(detail) {
  // detail: { run, path, record, ply }
  dbg.pendingDeepLink = detail;
  navigateScreen("debug");
}

// ---- rendering ------------------------------------------------------------

function debugRenderAll() {
  debugRenderBoard();
  debugRenderPlyBar();
  debugRenderPositionInfo();
  debugRenderValue();
  debugRenderMoves();
  debugRenderSearch();
  debugRenderCheckpointInfo();
  debugRenderCompare();
  debugRenderTrajectory();
}

function debugHeatMaps() {
  // Build q,r -> normalized weight maps for the active overlays.
  const maps = { policy: new Map(), visits: new Map(), opp: new Map() };
  const fill = (rows, key, field) => {
    if (!rows || !rows.length) return;
    const max = rows.reduce((m, r) => Math.max(m, r[field]), 0) || 1;
    for (const r of rows) maps[key].set(`${r.q},${r.r}`, r[field] / max);
  };
  if (dbg.analysis) {
    if (dbg.overlays.policy) fill(dbg.analysis.policy, "policy", "p");
    if (dbg.overlays.opp) fill(dbg.analysis.opp_policy, "opp", "p");
  }
  if (dbg.search && dbg.overlays.visits) fill(dbg.search.visit_policy, "visits", "p");
  return maps;
}

function debugRenderBoard() {
  if (!debugBoardSvg) return;
  const pos = dbg.position;
  if (!pos) { debugBoardSvg.innerHTML = ""; return; }

  const heat = debugHeatMaps();
  const cells = new Map();
  const addCell = (q, r, extra) => {
    const key = `${q},${r}`;
    const existing = cells.get(key) || { q, r, key };
    cells.set(key, Object.assign(existing, extra));
  };
  for (const c of (pos.legal || [])) addCell(c.q, c.r, { legal: true });
  for (const p of (pos.placements || [])) addCell(p.q, p.r, { placement: p });
  // Candidate cells from analysis (so heat shows even off the legal set).
  if (dbg.analysis) for (const r of dbg.analysis.policy || []) addCell(r.q, r.r, { candidate: true });
  for (const key of heat.policy.keys()) { const [q, r] = key.split(",").map(Number); addCell(q, r, {}); }

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const data = [];
  for (const cell of cells.values()) {
    const c = center(cell.q, cell.r);
    minX = Math.min(minX, c.x - HEX * 1.6); maxX = Math.max(maxX, c.x + HEX * 1.6);
    minY = Math.min(minY, c.y - HEX * 1.6); maxY = Math.max(maxY, c.y + HEX * 1.6);
    data.push(Object.assign(cell, { x: c.x, y: c.y }));
  }
  if (!Number.isFinite(minX)) { debugBoardSvg.innerHTML = ""; return; }
  debugBoardSvg.setAttribute("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`);
  debugBoardSvg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  const lastId = pos.debug.last_action_id;
  const lastCoord = lastId != null ? debugUnpackCoord(lastId) : null;
  data.sort((a, b) => (a.placement ? 1 : 0) - (b.placement ? 1 : 0));
  let html = "";

  // Threat windows (count >= 4) drawn under the cells as colored connectors.
  if (dbg.overlays.threats && pos.tactics && Array.isArray(pos.tactics.threats)) {
    for (const w of pos.tactics.threats) {
      const cells = (w.cells || []).map(c => center(c.q, c.r));
      if (cells.length < 2) continue;
      const owner = (w.threat_player || w.active_player || "");
      const color = owner.endsWith("1") ? "var(--p1)" : "var(--p0)";
      const d = cells.map((c, i) => `${i ? "L" : "M"}${c.x.toFixed(1)},${c.y.toFixed(1)}`).join("");
      html += `<path d="${d}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" opacity="0.5" pointer-events="none"></path>`;
    }
  }
  for (const h of data) {
    const isStone = Boolean(h.placement);
    const pol = heat.policy.get(h.key) || 0;
    const vis = heat.visits.get(h.key) || 0;
    const opp = heat.opp.get(h.key) || 0;
    let fill = isStone ? playerColor(h.placement.player) : "#101924";
    const stroke = isStone ? "#708296" : "#2c3d50";
    let opacity = isStone ? "1" : (h.legal || h.candidate ? "0.7" : "0.45");
    html += `<path class="dbg-cell" d="${path(h.x, h.y, HEX - 1)}" fill="${fill}" stroke="${stroke}" stroke-width="1" opacity="${opacity}" data-q="${h.q}" data-r="${h.r}"></path>`;
    if (!isStone) {
      if (pol > 0.02) html += `<path d="${path(h.x, h.y, HEX - 2)}" fill="var(--accent)" opacity="${(0.15 + 0.75 * pol).toFixed(3)}" pointer-events="none"></path>`;
      if (vis > 0.02) html += `<path d="${path(h.x, h.y, (HEX - 2) * Math.sqrt(vis))}" fill="var(--yellow)" opacity="0.75" pointer-events="none"></path>`;
      if (opp > 0.04) html += `<circle cx="${h.x}" cy="${h.y}" r="${(HEX * 0.28)}" fill="none" stroke="var(--p1)" stroke-width="1.4" opacity="${(0.3 + 0.6 * opp).toFixed(3)}" pointer-events="none"></circle>`;
    }
    if (lastCoord && h.q === lastCoord.q && h.r === lastCoord.r) {
      html += `<path class="dbg-last" d="${path(h.x, h.y, HEX - 0.5)}" fill="none" stroke="var(--accent)" stroke-width="2.2" pointer-events="none"></path>`;
    }
    if (isStone && dbg.overlays.numbers) {
      html += `<text class="dbg-stone-label" x="${h.x}" y="${h.y}" text-anchor="middle" dominant-baseline="central">${h.placement.index}</text>`;
    }
  }
  debugBoardSvg.innerHTML = html;
  debugBoardSvg.querySelectorAll(".dbg-cell").forEach(el => {
    el.addEventListener("mousemove", () => debugHoverCell(Number(el.dataset.q), Number(el.dataset.r)));
  });
}

function debugHoverCell(q, r) {
  const hud = dbgEl("debugBoardHud");
  if (!hud) return;
  const key = `${q},${r}`;
  const pol = dbg.analysis && (dbg.analysis.policy || []).find(x => x.q === q && x.r === r);
  const vis = dbg.search && (dbg.search.visit_policy || []).find(x => x.q === q && x.r === r);
  const parts = [`Q ${q} · R ${r}`];
  if (pol) parts.push(`prior ${(pol.p * 100).toFixed(1)}%`);
  if (vis) parts.push(`visits ${(vis.p * 100).toFixed(1)}%`);
  hud.innerHTML = `<div>${escapeText(parts.join("  ·  "))}</div>`;
}

function debugRenderPlyBar() {
  const pos = dbg.position;
  const total = pos ? pos.debug.total : 0;
  const ply = pos ? pos.debug.ply : 0;
  const slider = dbgEl("debugPlySlider");
  if (slider) { slider.max = String(total); slider.value = String(ply); slider.disabled = !pos; }
  const label = dbgEl("debugPlyLabel");
  if (label) label.textContent = `${ply} / ${total}`;
  const sub = dbgEl("debugPlySub");
  if (sub) {
    if (!pos) sub.textContent = "No game";
    else sub.textContent = ply === 0 ? "Opening" : `Move ${ply}`;
  }
}

function debugRenderPositionInfo() {
  const el = dbgEl("debugPositionInfo");
  if (!el) return;
  const pos = dbg.position;
  if (!pos) { el.innerHTML = `<div class="debug-empty">Pick a game or import a move list.</div>`; return; }
  const d = pos.debug;
  const a = dbg.analysis;
  const rows = [
    ["Game", pos.game_id ? escapeText(String(pos.game_id).split(":").pop()) : "—"],
    ["Ply", `${d.ply} / ${d.total}`],
    ["To play", a ? `P${a.current_player} (${escapeText(a.current_role || "")})` : "—"],
    ["Winner", d.winner ? escapeText(d.winner) : (d.imported ? "imported" : "—")],
    ["Candidates", a ? String(a.candidate_count) : "—"],
    ["Legal cells", a ? String(a.legal_count) : "—"],
  ];
  el.innerHTML = rows.map(([k, v]) => `<div class="info-row"><span class="label">${k}</span><span class="value">${v}</span></div>`).join("");
}

function debugRenderValue() {
  const el = dbgEl("debugValuePanel");
  const chip = dbgEl("debugValueChip");
  if (!el) return;
  const a = dbg.analysis;
  if (!a) { el.innerHTML = `<div class="debug-empty">${dbg.loading ? "Evaluating…" : "No analysis yet."}</div>`; if (chip) chip.textContent = ""; return; }
  if (chip) { chip.textContent = a.value.toFixed(3); chip.className = `debug-chip ${a.value >= 0 ? "pos" : "neg"}`; }

  const dist = a.value_dist || [];
  const bins = a.value_bins || [];
  const maxP = dist.reduce((m, x) => Math.max(m, x), 0) || 1;
  const bars = dist.map((p, i) => {
    const h = Math.max(1, (p / maxP) * 100);
    const center = bins[i] != null ? bins[i] : 0;
    const cls = center >= 0 ? "pos" : "neg";
    return `<span class="dbg-vbar ${cls}" style="height:${h.toFixed(1)}%" title="v=${center.toFixed(3)} p=${(p * 100).toFixed(1)}%"></span>`;
  }).join("");

  const stvRows = Object.keys(a.stvalue || {}).sort((x, y) => Number(x) - Number(y)).map(h => {
    const s = a.stvalue[h].scalar;
    return `<div class="dbg-stv-row"><span class="label">STV+${h}</span><span class="dbg-bar-track"><span class="dbg-bar-fill ${s >= 0 ? "pos" : "neg"}" style="width:${(Math.abs(s) * 50).toFixed(1)}%;${s >= 0 ? "left:50%" : "right:50%"}"></span></span><span class="value">${s.toFixed(3)}</span></div>`;
  }).join("");

  // Both-perspectives / optimism probe: the same board scored from each side.
  let optimismHtml = "";
  if (typeof a.value_swapped === "number") {
    const sum = a.optimism != null ? a.optimism : (a.value + a.value_swapped);
    const cal = Math.abs(sum) < 0.1 ? "ok" : (sum > 0 ? "warn" : "warn");
    optimismHtml = `
      <div class="dbg-perspectives">
        <div><span class="label">Side to move</span><span class="value ${a.value >= 0 ? "pos" : "neg"}">${a.value.toFixed(3)}</span></div>
        <div><span class="label">Opponent view</span><span class="value ${a.value_swapped >= 0 ? "pos" : "neg"}">${a.value_swapped.toFixed(3)}</span></div>
        <div title="v_self + v_opp; 0 = zero-sum consistent, >0 = both sides optimistic"><span class="label">Optimism Σ</span><span class="value dbg-opt-${cal}">${sum >= 0 ? "+" : ""}${sum.toFixed(3)}</span></div>
      </div>`;
  }

  el.innerHTML = `
    <div class="dbg-value-scalar">value <strong class="${a.value >= 0 ? "pos" : "neg"}">${a.value.toFixed(4)}</strong> <span class="dbg-muted">(side to move)</span></div>
    <div class="dbg-vdist" aria-label="65-bin value distribution">${bars}</div>
    <div class="dbg-vdist-axis"><span>loss −1</span><span>0</span><span>+1 win</span></div>
    ${optimismHtml}
    ${stvRows ? `<div class="dbg-stv">${stvRows}</div>` : ""}
  `;
}

function debugRenderMoves() {
  const el = dbgEl("debugMovesPanel");
  if (!el) return;
  const a = dbg.analysis;
  if (!a) { el.innerHTML = `<div class="debug-empty">—</div>`; return; }
  const visitsByKey = new Map();
  if (dbg.search) for (const r of dbg.search.visit_policy || []) visitsByKey.set(`${r.q},${r.r}`, r.p);
  const top = (a.policy || []).slice(0, 12);
  const head = `<div class="dbg-move-row dbg-move-head"><span>#</span><span>cell</span><span>prior</span><span>visits</span></div>`;
  const rows = top.map((r, i) => {
    const key = `${r.q},${r.r}`;
    const v = visitsByKey.has(key) ? `${(visitsByKey.get(key) * 100).toFixed(1)}%` : "—";
    const best = dbg.search && dbg.search.best && dbg.search.best.q === r.q && dbg.search.best.r === r.r;
    return `<div class="dbg-move-row${best ? " dbg-move-best" : ""}"><span>${i + 1}</span><span>${r.q},${r.r}</span><span>${(r.p * 100).toFixed(1)}%</span><span>${v}</span></div>`;
  }).join("");
  el.innerHTML = head + rows;
}

function debugRenderSearch() {
  const el = dbgEl("debugSearchPanel");
  if (!el) return;
  const s = dbg.search;
  if (!s) { el.innerHTML = `<div class="debug-empty">Run a search to compare visit distribution vs the raw prior.</div>`; return; }
  const a = dbg.analysis;
  const delta = a ? (s.root_value - a.value) : null;
  const top = (s.visit_policy || []).slice(0, 8);
  const priorByKey = new Map((s.root_prior || []).map(r => [`${r.q},${r.r}`, r.p]));
  const rows = top.map(r => {
    const pr = priorByKey.get(`${r.q},${r.r}`);
    return `<div class="dbg-move-row"><span>${r.q},${r.r}</span><span>${(r.p * 100).toFixed(1)}%</span><span class="dbg-muted">${pr != null ? (pr * 100).toFixed(1) + "%" : "—"}</span></div>`;
  }).join("");
  el.innerHTML = `
    <div class="dbg-search-summary">
      <span>visits <strong>${s.visits}</strong></span>
      <span>root value <strong class="${s.root_value >= 0 ? "pos" : "neg"}">${s.root_value.toFixed(3)}</strong></span>
      ${delta != null ? `<span class="dbg-muted">Δ vs raw ${delta >= 0 ? "+" : ""}${delta.toFixed(3)}</span>` : ""}
    </div>
    <div class="dbg-move-row dbg-move-head"><span>cell</span><span>visits</span><span>prior</span></div>
    ${rows}
  `;
}

async function debugRunCompare() {
  const body = debugRequestBody();
  if (!body || !dbg.compareCheckpoint) { dbg.compare = null; debugRenderCompare(); return; }
  const compareBody = Object.assign({}, body, { checkpoint: dbg.compareCheckpoint });
  debugSetStatus("Evaluating comparison checkpoint…", "busy");
  try {
    const analysis = await debugFetchJson("/api/debug/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(compareBody),
    });
    dbg.compare = { checkpoint: dbg.compareCheckpoint, analysis };
    debugSetStatus("");
  } catch (e) {
    dbg.compare = null;
    debugSetStatus(`Compare: ${e.message}`, "error");
  }
  debugRenderCompare();
}

function debugRenderCompare() {
  const el = dbgEl("debugComparePanel");
  if (!el) return;
  if (!dbg.compareCheckpoint) { el.innerHTML = `<div class="debug-empty">Pick a second checkpoint to compare value &amp; top move.</div>`; return; }
  const a = dbg.analysis, c = dbg.compare && dbg.compare.analysis;
  if (!c) { el.innerHTML = `<div class="debug-empty">Evaluating…</div>`; return; }
  const topA = a && a.policy && a.policy[0];
  const topC = c.policy && c.policy[0];
  const dv = a ? (c.value - a.value) : null;
  const rows = [
    ["Value A", a ? a.value.toFixed(3) : "—"],
    ["Value B", c.value.toFixed(3)],
    ["Δ (B−A)", dv != null ? `${dv >= 0 ? "+" : ""}${dv.toFixed(3)}` : "—"],
    ["Top A", topA ? `${topA.q},${topA.r} (${(topA.p * 100).toFixed(0)}%)` : "—"],
    ["Top B", topC ? `${topC.q},${topC.r} (${(topC.p * 100).toFixed(0)}%)` : "—"],
    ["Agree", (topA && topC) ? (topA.q === topC.q && topA.r === topC.r ? "yes" : "no") : "—"],
  ];
  el.innerHTML = rows.map(([k, v]) => `<div class="info-row"><span class="label">${k}</span><span class="value">${v}</span></div>`).join("");
}

async function debugPlotTrajectory() {
  if (!dbg.run || !dbg.gameFile || !dbg.checkpoint || (dbg.position && dbg.position.debug.imported)) {
    debugSetStatus("Trajectory needs a recorded game + checkpoint.", "error");
    return;
  }
  debugSetStatus("Re-evaluating the whole game on CPU…", "busy");
  const note = dbgEl("debugTrajectoryNote");
  try {
    const params = new URLSearchParams({ run: dbg.run, path: dbg.gameFile, record: String(dbg.record), checkpoint: dbg.checkpoint });
    dbg.trajectory = await debugFetchJson(`/api/debug/trajectory?${params.toString()}`);
    if (note) note.textContent = dbg.trajectory.stride > 1 ? `(every ${dbg.trajectory.stride} plies)` : "";
    debugSetStatus("");
  } catch (e) {
    dbg.trajectory = null;
    debugSetStatus(`Trajectory: ${e.message}`, "error");
  }
  debugRenderTrajectory();
}

function debugRenderTrajectory() {
  const svg = dbgEl("debugTrajectorySvg");
  if (!svg) return;
  const t = dbg.trajectory;
  if (!t || !t.reeval || !t.reeval.length) { svg.innerHTML = `<text x="500" y="110" text-anchor="middle" fill="#5a6b7a">Plot to re-evaluate value across the game.</text>`; return; }
  const W = 1000, H = 220, padL = 36, padR = 12, padT = 14, padB = 22;
  const total = t.total || 1;
  const x = ply => padL + (W - padL - padR) * (total ? ply / total : 0);
  const y = v => padT + (H - padT - padB) * (1 - (v + 1) / 2); // v in [-1,1] -> top=+1
  const linePath = (pts, key) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.ply).toFixed(1)},${y(p[key]).toFixed(1)}`).join("");

  let html = "";
  // grid: zero line + ±1
  html += `<line x1="${padL}" y1="${y(0)}" x2="${W - padR}" y2="${y(0)}" stroke="#2c3d50" stroke-width="1"></line>`;
  html += `<text x="4" y="${y(1) + 4}" fill="#5a6b7a" font-size="11">+1</text>`;
  html += `<text x="6" y="${y(0) + 4}" fill="#5a6b7a" font-size="11">0</text>`;
  html += `<text x="4" y="${y(-1) + 2}" fill="#5a6b7a" font-size="11">−1</text>`;
  // current ply marker
  if (dbg.position) {
    const px = x(dbg.position.debug.ply);
    html += `<line x1="${px}" y1="${padT}" x2="${px}" y2="${H - padB}" stroke="var(--accent)" stroke-width="1" opacity="0.4" stroke-dasharray="3 3"></line>`;
  }
  if (t.recorded && t.recorded.length) {
    html += `<path d="${linePath(t.recorded, "root_value_p0")}" fill="none" stroke="var(--yellow)" stroke-width="1.6" opacity="0.85"></path>`;
  }
  html += `<path d="${linePath(t.reeval, "value_p0")}" fill="none" stroke="var(--accent)" stroke-width="2"></path>`;
  svg.innerHTML = html;
}

function debugRenderCheckpointInfo() {
  const el = dbgEl("debugCheckpointInfo");
  if (!el) return;
  const ck = dbg.checkpoints.find(c => c.name === dbg.checkpoint);
  const meta = dbg.analysis && dbg.analysis.meta;
  if (!ck && !meta) { el.innerHTML = `<div class="debug-empty">—</div>`; return; }
  const rows = [];
  if (ck) {
    rows.push(["Checkpoint", escapeText(ck.name)]);
    if (ck.epoch != null) rows.push(["Epoch", String(ck.epoch)]);
    if (ck.graft) rows.push(["Graft", ck.graft === "pre" ? "pre (≤e6, expanded)" : "post (≥e7)"]);
  }
  if (meta) {
    if (meta.rl_epoch != null) rows.push(["RL epoch", String(meta.rl_epoch)]);
    if (meta.expanded_stv && meta.expanded_stv.length) rows.push(["STV expand", `${meta.expanded_stv.length} heads`]);
    if (meta.load_warnings && meta.load_warnings.length) rows.push(["Warnings", escapeText(meta.load_warnings.join("; "))]);
  }
  el.innerHTML = rows.map(([k, v]) => `<div class="info-row"><span class="label">${k}</span><span class="value">${v}</span></div>`).join("");
}

// ---- events ---------------------------------------------------------------

function debugBindEvents() {
  const on = (id, ev, fn) => { const el = dbgEl(id); if (el) el.addEventListener(ev, fn); };
  on("debugRunSelect", "change", async e => { dbg.run = e.target.value; dbg.gameFile = ""; dbg.checkpoint = ""; await debugLoadRun(); });
  on("debugSourceSelect", "change", async e => { dbg.source = e.target.value; dbg.gameFile = ""; await debugLoadGames(); });
  on("debugGameSelect", "change", async e => { dbg.gameFile = e.target.value; dbg.record = 0; dbg.imported = null; await debugLoadPosition({ resetPly: true }); });
  on("debugRecordSelect", "change", async e => { dbg.record = Number(e.target.value) || 0; await debugLoadPosition({ resetPly: true }); });
  on("debugCheckpointSelect", "change", async e => { dbg.checkpoint = e.target.value; dbg.search = null; debugRenderCheckpointInfo(); await debugAnalyze(); });
  on("debugRefreshBtn", "click", async () => { await debugLoadRun(); });
  on("debugAnalyzeBtn", "click", () => debugAnalyze());
  on("debugSearchBtn", "click", () => debugRunSearch());
  on("debugCompareSelect", "change", e => { dbg.compareCheckpoint = e.target.value; dbg.compare = null; debugRunCompare(); });
  on("debugTrajectoryBtn", "click", () => debugPlotTrajectory());
  on("debugPlyStart", "click", () => debugStep(-1e9));
  on("debugPlyPrev", "click", () => debugStep(-1));
  on("debugPlyNext", "click", () => debugStep(1));
  on("debugPlyEnd", "click", () => debugStep(1e9));
  on("debugPlySlider", "input", e => debugGotoPly(Number(e.target.value)));
  on("debugImportBtn", "click", () => debugImport());
  on("debugMoveInput", "keydown", e => { if (e.key === "Enter") debugImport(); });
  ["Policy", "Visits", "Opp", "Threats", "Numbers"].forEach(name => {
    on(`debugOv${name}`, "change", e => { dbg.overlays[name.toLowerCase()] = e.target.checked; debugRenderBoard(); });
  });
}

let debugPlyTimer = null;
function debugGotoPly(ply) {
  if (!dbg.position) return;
  dbg.position.debug.ply = Math.max(0, Math.min(ply, dbg.position.debug.total));
  dbg.position.debug.last_action_id = dbg.position.debug.ply > 0 ? dbg.position.debug.action_ids[dbg.position.debug.ply - 1] : null;
  debugRenderPlyBar();
  // Debounce the position+analyze fetch while dragging the slider.
  window.clearTimeout(debugPlyTimer);
  debugPlyTimer = window.setTimeout(() => debugLoadPosition({ ply: dbg.position.debug.ply }), 180);
}

function debugStep(delta) {
  if (!dbg.position) return;
  const ply = Math.max(0, Math.min(dbg.position.debug.ply + delta, dbg.position.debug.total));
  debugGotoPly(ply);
}

function debugImport() {
  const input = dbgEl("debugMoveInput");
  if (!input) return;
  const ids = (input.value || "").split(/[\s,]+/).map(s => s.trim()).filter(Boolean).map(Number).filter(n => Number.isFinite(n));
  if (!ids.length) { debugSetStatus("Paste a comma/space-separated action-id list.", "error"); return; }
  dbg.imported = ids;
  debugLoadImported(ids.length);
}

function debugSyncRecordSelect() {
  const sel = dbgEl("debugRecordSelect");
  if (!sel) return;
  if (!dbg.records.length) { sel.innerHTML = `<option value="0">0</option>`; sel.value = "0"; return; }
  sel.innerHTML = dbg.records.map(r => {
    const win = r.winner ? ` (${r.winner.replace("player", "P")})` : "";
    return `<option value="${r.index}">#${r.index} · ${r.actions}mv${escapeText(win)}</option>`;
  }).join("");
  sel.value = String(dbg.record);
}

async function init() {
  setScreen(activeScreen, { preserveHash: true });
  await Promise.allSettled([loadAdapters(), loadState(), loadTrainingRuns()]);
  render();
}

init();
