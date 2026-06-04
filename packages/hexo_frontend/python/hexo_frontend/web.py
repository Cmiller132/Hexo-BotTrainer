"""Tiny stdlib web app for manually playing a Hexo match through the runner."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from email.utils import formatdate
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Condition, Lock, RLock, Thread
from time import monotonic, perf_counter, time as wall_clock
from typing import Any, ClassVar
from urllib.parse import parse_qs, unquote, urlparse

import hexo_engine as engine
from hexo_runner.adapters.sealbot import (
    DEFAULT_SEALBOT_TIME_LIMIT,
    SealBotConfig,
    SealBotPlayer,
    discover_sealbot_adapters,
)
from hexo_runner.modes.match import run_match
from hexo_runner.player import DecisionResult, FinalSummary, PlayerIdentity, TransitionEvent, WorkerContext, GameContext
from hexo_runner.records import GameResult, HexoRecordFile
from hexo_runner.session import GameSpec
from hexo_engine.types import unpack_coord_id

from .dashboard import dashboard_state
from . import debug_service


STATIC_ROOT = files("hexo_frontend").joinpath("static")
STATIC_TYPES = {
    "css": "text/css; charset=utf-8",
    "html": "text/html; charset=utf-8",
    "js": "text/javascript; charset=utf-8",
}
ARTIFACT_TYPES = {
    ".json": "application/json; charset=utf-8",
    ".jsonl": "application/x-ndjson; charset=utf-8",
    ".png": "image/png",
    ".hxr": "application/octet-stream",
}
ARTIFACT_SUFFIXES = frozenset(ARTIFACT_TYPES)
TRAINING_SCAN_EXCLUDED_DIRS = frozenset({"archive", "quarantine", "__pycache__"})
HISTORY_ALL_RUNS = "__all__"
TRAINING_OVERVIEW_HISTORY_LIMIT = 50
TRAINING_OVERVIEW_ARTIFACT_LIMIT = 50
HISTORY_PAGE_DEFAULT_LIMIT = 400
HISTORY_PAGE_MAX_LIMIT = 500
ARTIFACT_PAGE_DEFAULT_LIMIT = 50
ARTIFACT_PAGE_MAX_LIMIT = 200
BotFactory = Callable[[str, float], object]
PLAYER_ROLES = ("player0", "player1")
MANUAL_KIND = "manual"
SEALBOT_PREFIX = "sealbot-"

# --- Transfer efficiency (gzip + caching) -----------------------------------
# The dashboard is often viewed over LAN/VPN, where uncompressed payloads and
# per-request connections are slow. Responses below this size aren't worth
# gzipping (the ~20-byte gzip header/overhead dominates).
GZIP_MIN_BYTES = 600
# Browser cache window for static assets. index.html cache-busts app.js/styles.css
# with a ?v= query, so a few minutes of caching is safe and makes reloads ~free;
# the ETag still forces revalidation (304, no body) once it expires.
STATIC_MAX_AGE_SECONDS = 300
# The training-run/list scans walk the run tree and open many .hxr files. The
# history screen re-polls every 15s, so memoize the built payload briefly to bound
# that work to at most once per interval regardless of poll/client count.
TRAINING_CACHE_TTL_SECONDS = 3.0

_static_lock = Lock()
# name -> (mtime, (raw_bytes, gzipped_bytes, etag, last_modified, content_type))
_static_cache: dict[str, tuple[float, tuple[bytes, bytes, str, str, str]]] = {}
_training_cache_lock = Lock()
_training_run_cache: dict[str, tuple[float, dict[str, object]]] = {}
_training_runs_cache: list[tuple[float, dict[str, object]]] = []
_hxr_history_cache: dict[str, tuple[int, int, list[dict[str, object]]]] = {}
_hxr_count_cache: dict[str, tuple[int, int, int]] = {}


def _strong_etag(data: bytes) -> str:
    return '"' + hashlib.sha1(data).hexdigest()[:20] + '"'


def _static_entry(name: str) -> tuple[bytes, bytes, str, str, str] | None:
    """Return (raw, gzipped, etag, last_modified, content_type) for a static asset.

    Memoized by (name, mtime): the file is read, gzipped, and hashed at most once
    per version, so serving it is just a dict lookup + socket write. Returns None
    when the asset is missing.
    """

    resource = STATIC_ROOT.joinpath(name)
    mtime: float | None
    try:
        mtime = Path(str(resource)).stat().st_mtime
    except (OSError, TypeError, ValueError):
        mtime = None
    if mtime is not None:
        with _static_lock:
            hit = _static_cache.get(name)
            if hit is not None and hit[0] == mtime:
                return hit[1]
    try:
        raw = resource.read_bytes()
    except (FileNotFoundError, IsADirectoryError, OSError):
        return None
    extension = name.rsplit(".", 1)[-1]
    content_type = STATIC_TYPES.get(extension, "application/octet-stream")
    entry = (
        raw,
        gzip.compress(raw, 6),
        _strong_etag(raw),
        formatdate(mtime, usegmt=True) if mtime else formatdate(usegmt=True),
        content_type,
    )
    if mtime is not None:
        with _static_lock:
            _static_cache[name] = (mtime, entry)
    return entry


def _training_run_cached(name: str) -> dict[str, object]:
    now = monotonic()
    with _training_cache_lock:
        hit = _training_run_cache.get(name)
        if hit is not None and now - hit[0] < TRAINING_CACHE_TTL_SECONDS:
            return hit[1]
    payload = _training_run(name)  # heavy scan, kept outside the lock
    with _training_cache_lock:
        _training_run_cache[name] = (monotonic(), payload)
    return payload


def _training_runs_cached() -> dict[str, object]:
    now = monotonic()
    with _training_cache_lock:
        if _training_runs_cache and now - _training_runs_cache[0][0] < TRAINING_CACHE_TTL_SECONDS:
            return _training_runs_cache[0][1]
    payload = _training_runs()
    with _training_cache_lock:
        _training_runs_cache[:] = [(monotonic(), payload)]
    return payload


class MoveConflict(ValueError):
    """Raised when a browser move arrives while the human cannot act."""


class ManualMatchController:
    """Frontend-owned bridge between HTTP clicks and generic runner players."""

    def __init__(self, *, sealbot_path: str | Path | None = None, bot_factory: BotFactory | None = None) -> None:
        self._condition = Condition(RLock())
        self._sealbot_path = Path(sealbot_path).expanduser().resolve() if sealbot_path else None
        self._bot_factory = bot_factory
        self._thread: Thread | None = None
        self._game_number = 0
        self._cancelled = False
        self._state: engine.HexoState | None = None
        self._python_state: engine.PythonHexoState | None = None
        self._pending_action: engine.Action | None = None
        self._version = 0
        self._result: GameResult | None = None
        self._error: BaseException | None = None
        self._mode = "manual"
        self._player_setup: dict[str, str] = {"player0": MANUAL_KIND, "player1": MANUAL_KIND}
        self._bot_time_limit = DEFAULT_SEALBOT_TIME_LIMIT
        self._seed: int | None = None
        self._thinking_player: str | None = None
        self._last_bot_decision: dict[str, object] | None = None
        self._observed_transition: tuple[str, int] | None = None
        self.reset()

    def reset(self, config: dict[str, Any] | None = None) -> dict[str, object]:
        match = self._parse_match_config(config or {})
        self.close()
        with self._condition:
            self._game_number += 1
            self._mode = match["mode"]
            self._player_setup = dict(match["players"])
            self._bot_time_limit = match["time_limit"]
            self._seed = match["seed"]
            game_id = f"{self._mode}-{self._game_number}"
            self._cancelled = False
            self._state = None
            self._python_state = None
            self._pending_action = None
            self._version = 0
            self._result = None
            self._error = None
            self._thinking_player = None
            self._last_bot_decision = None
            self._observed_transition = None
            players = self._players_for_match()
            spec = GameSpec(game_id=game_id, seed=self._seed, mode=self._mode)
            self._thread = Thread(target=self._run_match, args=(spec, players), daemon=True)
            self._thread.start()
            self._wait_for_state_locked()
            return self._payload_locked()

    def adapters(self) -> dict[str, object]:
        return {"sealbot": discover_sealbot_adapters(self._sealbot_path)}

    def state(self, *, since: int | None = None, timeout_ms: int = 0) -> dict[str, object]:
        with self._condition:
            self._wait_for_state_locked()
            if since is not None and self._version <= since and timeout_ms > 0:
                deadline = monotonic() + max(0.0, min(timeout_ms, 30000) / 1000.0)
                while self._version <= since and self._error is None:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
            return self._payload_locked()

    def submit_move(self, q: int, r: int) -> dict[str, object]:
        with self._condition:
            self._wait_for_state_locked()
            state = self._state
            if state is None or self._result is not None:
                raise MoveConflict("No move is currently pending.")
            if not self._can_submit_locked():
                raise MoveConflict("It is not the human player's turn.")
            action = engine.PlacementAction(engine.AxialCoord(q=q, r=r))
            if not engine.is_legal_action(state, action):
                raise ValueError(f"{q},{r} is not legal.")

            start_version = self._version
            self._pending_action = action
            self._condition.notify_all()
            while self._version == start_version and self._error is None and self._result is None:
                self._condition.wait(timeout=0.25)
            if self._error is not None:
                raise RuntimeError(str(self._error)) from self._error
            return self._payload_locked()

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError("Timed out waiting for the current match to stop.")
        self._thread = None

    def decide(self, player_index: int, state: engine.HexoState) -> DecisionResult:
        with self._condition:
            if self._cancelled:
                raise RuntimeError("manual match reset")
            self._set_state_locked(state)
            self._version += 1
            self._condition.notify_all()

            while self._pending_action is None and not self._cancelled:
                self._condition.wait()
            if self._cancelled:
                raise RuntimeError("manual match reset")

            action = self._pending_action
            self._pending_action = None
            return DecisionResult(action=action, diagnostics={"manual_player": player_index})

    def bot_decision_started(self, player_index: int, state: engine.HexoState) -> None:
        with self._condition:
            if self._cancelled:
                raise RuntimeError("manual match reset")
            self._set_state_locked(state)
            self._thinking_player = _player_role(player_index)
            self._version += 1
            self._condition.notify_all()

    def bot_decision_finished(self, player_index: int, result: DecisionResult, duration_ms: float) -> None:
        action = result.action
        payload: dict[str, object] = {
            "player": _player_role(player_index),
            "duration_ms": round(duration_ms, 3),
            "diagnostics": dict(result.diagnostics),
        }
        if isinstance(action, engine.PlacementAction):
            payload.update({"q": action.coord.q, "r": action.coord.r})
        with self._condition:
            self._thinking_player = None
            self._last_bot_decision = payload
            self._version += 1
            self._condition.notify_all()

    def bot_decision_failed(self, player_index: int, exc: BaseException, duration_ms: float) -> None:
        with self._condition:
            self._thinking_player = None
            self._last_bot_decision = {
                "player": _player_role(player_index),
                "duration_ms": round(duration_ms, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._version += 1
            self._condition.notify_all()

    def observe_transition(self, transition: TransitionEvent) -> None:
        with self._condition:
            key = (transition.game_id, transition.action_index)
            if self._observed_transition == key:
                return
            self._observed_transition = key
            self._set_state_locked(transition.state)
            self._version += 1
            self._condition.notify_all()

    def _run_match(self, spec: GameSpec, players: tuple[object, object]) -> None:
        try:
            with tempfile.TemporaryDirectory(prefix="hexo_manual_records_") as tmp:
                result = run_match(spec, players, tmp)
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._thinking_player = None
                self._condition.notify_all()
            return
        with self._condition:
            self._result = result
            self._thinking_player = None
            self._version += 1
            self._condition.notify_all()

    def _players_for_match(self) -> tuple[object, object]:
        return (
            self._make_player(0, self._player_setup["player0"]),
            self._make_player(1, self._player_setup["player1"]),
        )

    def _make_player(self, player_index: int, kind: str) -> object:
        role = _player_role(player_index)
        if kind == MANUAL_KIND:
            return _ManualPlayer(self, player_index, label=f"{_player_label(role)} Manual")

        variant = _sealbot_variant(kind)
        if self._bot_factory is not None:
            bot = self._bot_factory(variant, self._bot_time_limit)
        else:
            bot = SealBotPlayer(
                SealBotConfig(
                    path=self._sealbot_path,
                    variant=variant,
                    time_limit=self._bot_time_limit,
                )
            )
        return _ObservedBotPlayer(self, player_index, bot)

    def _parse_match_config(self, config: dict[str, Any]) -> dict[str, Any]:
        bot = config.get("bot") if isinstance(config.get("bot"), dict) else {}
        time_limit = float(bot.get("time_limit") or self._bot_time_limit or DEFAULT_SEALBOT_TIME_LIMIT)
        if "time_limit" in config and config["time_limit"] not in {"", None}:
            time_limit = float(config["time_limit"])
        if time_limit <= 0:
            raise ValueError("SealBot time_limit must be positive.")
        seed = config.get("seed")
        players = self._normalize_player_setup(config)
        mode = "sealbot" if any(_is_sealbot_kind(kind) for kind in players.values()) else "manual"
        return {
            "mode": mode,
            "players": players,
            "time_limit": time_limit,
            "seed": None if seed in {"", None} else int(seed),
        }

    def _normalize_player_setup(self, config: dict[str, Any]) -> dict[str, str]:
        raw_players = config.get("players")
        if isinstance(raw_players, dict):
            return {
                "player0": _normalize_player_kind(raw_players.get("player0", MANUAL_KIND)),
                "player1": _normalize_player_kind(raw_players.get("player1", MANUAL_KIND)),
            }

        mode = str(config.get("mode") or "manual")
        if mode not in {"manual", "sealbot"}:
            raise ValueError(f"Unknown match mode: {mode}")
        if mode == "manual":
            return {"player0": MANUAL_KIND, "player1": MANUAL_KIND}

        human_player = str(config.get("human_player") or "player0")
        if human_player not in PLAYER_ROLES:
            raise ValueError("human_player must be player0 or player1.")
        bot = config.get("bot") if isinstance(config.get("bot"), dict) else {}
        variant = str(bot.get("variant") or "current")
        bot_kind = _normalize_player_kind({"kind": "sealbot", "variant": variant})
        return {
            "player0": MANUAL_KIND if human_player == "player0" else bot_kind,
            "player1": MANUAL_KIND if human_player == "player1" else bot_kind,
        }

    def _wait_for_state_locked(self, timeout: float = 5.0) -> None:
        deadline = monotonic() + timeout
        while self._python_state is None and self._error is None:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise RuntimeError("Timed out waiting for match state.")
            self._condition.wait(timeout=remaining)
        if self._python_state is None and self._error is not None:
            raise RuntimeError(str(self._error)) from self._error

    def _payload_locked(self) -> dict[str, object]:
        payload = dashboard_state(self._require_state_locked())
        payload.update(
            {
                "version": self._version,
                "game_id": f"{self._mode}-{self._game_number}",
                "mode": self._mode,
                "players": self._players_payload_locked(),
                "turn_status": self._turn_status_locked(payload),
                "can_submit": self._can_submit_locked(),
                "thinking_player": self._thinking_player,
                "last_bot_decision": self._last_bot_decision,
                "error": self._error_message_locked(),
                "match": {
                    "players": dict(self._player_setup),
                    "time_limit": self._bot_time_limit,
                    "seed": self._seed,
                },
            }
        )
        return payload

    def _players_payload_locked(self) -> dict[str, dict[str, object]]:
        return {
            role: _player_payload(index, self._player_setup[role])
            for index, role in enumerate(PLAYER_ROLES)
        }

    def _turn_status_locked(self, payload: dict[str, object]) -> str:
        if self._error is not None or (self._result is not None and self._result.abort is not None):
            return "error"
        if self._result is not None or payload.get("winner") is not None:
            return "terminal"
        if self._thinking_player is not None:
            return "bot_thinking"
        current = str(payload.get("current_player") or "")
        return "bot_thinking" if _is_sealbot_kind(self._player_setup.get(current, MANUAL_KIND)) else "human_turn"

    def _can_submit_locked(self) -> bool:
        if self._state is None or self._result is not None or self._pending_action is not None:
            return False
        if self._thinking_player is not None:
            return False
        if self._python_state is not None and self._python_state.terminal is not None:
            return False
        current = str(engine.current_player(self._state))
        if _is_sealbot_kind(self._player_setup.get(current, MANUAL_KIND)):
            return False
        return True

    def _error_message_locked(self) -> str | None:
        if self._error is not None:
            return str(self._error)
        if self._result is not None and self._result.abort is not None:
            return self._result.abort.message
        return None

    def _set_state_locked(self, state: engine.HexoState) -> None:
        self._state = state
        self._python_state = engine.to_python_state(state)

    def _require_state_locked(self) -> engine.PythonHexoState:
        if self._python_state is None:
            raise RuntimeError("Match state is unavailable.")
        return self._python_state


class _ManualPlayer:
    def __init__(self, controller: ManualMatchController, player_index: int, *, label: str) -> None:
        self._controller = controller
        self._player_index = player_index
        self.identity = PlayerIdentity(player_id=f"manual-player-{player_index}", label=label)

    def setup_worker(self, context: WorkerContext) -> None:
        return

    def start_game(self, context: GameContext) -> None:
        return

    def decide(self, state: engine.HexoState) -> DecisionResult:
        return self._controller.decide(self._player_index, state)

    def observe_transition(self, transition: TransitionEvent) -> None:
        self._controller.observe_transition(transition)

    def finish_game(self, final_summary: FinalSummary) -> None:
        return

    def close(self) -> None:
        return


class _ObservedBotPlayer:
    def __init__(self, controller: ManualMatchController, player_index: int, delegate: object) -> None:
        self._controller = controller
        self._player_index = player_index
        self._delegate = delegate
        self.identity = delegate.identity

    def setup_worker(self, context: WorkerContext) -> None:
        self._delegate.setup_worker(context)

    def start_game(self, context: GameContext) -> None:
        self._delegate.start_game(context)

    def decide(self, state: engine.HexoState) -> DecisionResult:
        self._controller.bot_decision_started(self._player_index, state)
        started = perf_counter()
        try:
            result = self._delegate.decide(state)
        except BaseException as exc:
            self._controller.bot_decision_failed(self._player_index, exc, (perf_counter() - started) * 1000.0)
            raise
        self._controller.bot_decision_finished(self._player_index, result, (perf_counter() - started) * 1000.0)
        return result

    def observe_transition(self, transition: TransitionEvent) -> None:
        self._delegate.observe_transition(transition)
        self._controller.observe_transition(transition)

    def finish_game(self, final_summary: FinalSummary) -> None:
        self._delegate.finish_game(final_summary)

    def close(self) -> None:
        self._delegate.close()


def _player_role(player_index: int) -> str:
    return "player0" if player_index == 0 else "player1"


def _player_label(role: str) -> str:
    return "P0" if role == "player0" else "P1"


def _is_sealbot_kind(kind: str) -> bool:
    return kind.startswith(SEALBOT_PREFIX)


def _sealbot_variant(kind: str) -> str:
    if not _is_sealbot_kind(kind):
        raise ValueError(f"Player kind is not SealBot: {kind}")
    return kind.removeprefix(SEALBOT_PREFIX)


def _normalize_player_kind(value: object) -> str:
    if isinstance(value, dict):
        kind = str(value.get("kind") or value.get("adapter") or value.get("id") or MANUAL_KIND)
        variant = str(value.get("variant") or "current")
        if kind in {"manual", "human"}:
            return MANUAL_KIND
        if kind in {"bot", "sealbot"}:
            return _normalize_player_kind(f"sealbot-{variant}")
        return _normalize_player_kind(kind)

    kind = str(value or MANUAL_KIND).strip().lower()
    if kind in {"manual", "human"}:
        return MANUAL_KIND
    if kind in {"bot", "sealbot"}:
        return "sealbot-current"
    if kind in {"sealbot-current", "sealbot-best"}:
        return kind
    raise ValueError(f"Unknown player kind: {kind}")


def _player_payload(player_index: int, kind: str) -> dict[str, object]:
    role = _player_role(player_index)
    if kind == MANUAL_KIND:
        return {"role": role, "kind": kind, "label": "Manual"}
    variant = _sealbot_variant(kind)
    return {
        "role": role,
        "kind": kind,
        "label": f"SealBot {variant}",
        "adapter_id": "sealbot",
        "variant": variant,
    }


class HexoPlayHandler(BaseHTTPRequestHandler):
    server_version = "hexo-frontend-play/0.1"
    # HTTP/1.1 keep-alive: reuse one TCP connection for index.html + app.js +
    # styles.css + the API calls instead of a fresh connection per request -- the
    # big win over high-latency LAN/VPN links. Every response sets Content-Length
    # (required for keep-alive); 304s carry no body.
    protocol_version = "HTTP/1.1"
    timeout = 30  # reap idle keep-alive connections so handler threads don't pile up
    controller: ClassVar[ManualMatchController]

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/state":
                query = parse_qs(parsed.query)
                since = _query_int(query.get("since", [None])[0])
                timeout_ms = _query_int(query.get("timeout_ms", [None])[0]) or 0
                self._send_json(self.controller.state(since=since, timeout_ms=timeout_ms))
            elif path == "/api/adapters":
                self._send_json(self.controller.adapters())
            elif path == "/api/training/runs":
                self._send_json(_training_runs_cached())
            elif path == "/api/training/run":
                query = parse_qs(parsed.query)
                self._send_json(_training_run_cached(str(query.get("name", [""])[0])))
            elif path == "/api/training/history-page":
                query = parse_qs(parsed.query)
                self._send_json(
                    _training_history_page(
                        run_name=str(query.get("run", [""])[0]),
                        limit=_query_limit(
                            query.get("limit", [None])[0],
                            default=HISTORY_PAGE_DEFAULT_LIMIT,
                            maximum=HISTORY_PAGE_MAX_LIMIT,
                        ),
                        cursor=str(query.get("cursor", [""])[0] or ""),
                        source=str(query.get("source", ["all"])[0] or "all"),
                        winner=str(query.get("winner", ["all"])[0] or "all"),
                        sort=str(query.get("sort", ["newest"])[0] or "newest"),
                        query_text=str(query.get("query", [""])[0] or ""),
                        include_total=_query_bool(query.get("include_total", ["1"])[0], default=True),
                    )
                )
            elif path == "/api/training/history-count":
                query = parse_qs(parsed.query)
                self._send_json(
                    _training_history_count(
                        run_name=str(query.get("run", [""])[0]),
                        source=str(query.get("source", ["all"])[0] or "all"),
                        winner=str(query.get("winner", ["all"])[0] or "all"),
                        query_text=str(query.get("query", [""])[0] or ""),
                    )
                )
            elif path == "/api/training/artifacts-page":
                query = parse_qs(parsed.query)
                self._send_json(
                    _training_artifacts_page(
                        run_name=str(query.get("run", [""])[0]),
                        limit=_query_limit(
                            query.get("limit", [None])[0],
                            default=ARTIFACT_PAGE_DEFAULT_LIMIT,
                            maximum=ARTIFACT_PAGE_MAX_LIMIT,
                        ),
                        cursor=str(query.get("cursor", [""])[0] or ""),
                        kind=str(query.get("kind", ["all"])[0] or "all"),
                    )
                )
            elif path == "/api/training/file":
                query = parse_qs(parsed.query)
                self._send_training_file(
                    str(query.get("run", [""])[0]),
                    str(query.get("path", [""])[0]),
                )
            elif path == "/api/training/history":
                query = parse_qs(parsed.query)
                self._send_json(
                    _training_history(
                        str(query.get("run", [""])[0]),
                        str(query.get("path", [""])[0]),
                        _query_int(query.get("record", [None])[0]) or 0,
                    )
                )
            elif path == "/api/debug/checkpoints":
                query = parse_qs(parsed.query)
                self._send_json(_debug_checkpoints(str(query.get("run", [""])[0])))
            elif path == "/api/debug/games":
                query = parse_qs(parsed.query)
                self._send_json(
                    _debug_games(
                        str(query.get("run", [""])[0]),
                        str(query.get("source", ["selfplay"])[0] or "selfplay"),
                    )
                )
            elif path == "/api/debug/trajectory":
                query = parse_qs(parsed.query)
                self._send_json(
                    _debug_trajectory(
                        str(query.get("run", [""])[0]),
                        str(query.get("path", [""])[0]),
                        _query_int(query.get("record", [None])[0]) or 0,
                        str(query.get("checkpoint", [""])[0]),
                    )
                )
            elif path == "/api/debug/position":
                query = parse_qs(parsed.query)
                self._send_json(
                    _debug_position(
                        str(query.get("run", [""])[0]),
                        str(query.get("path", [""])[0]),
                        _query_int(query.get("record", [None])[0]) or 0,
                        _query_int(query.get("ply", [None])[0]) or 0,
                    )
                )
            elif path == "/" or path == "/index.html":
                self._send_static("index.html")
            elif path.startswith("/static/"):
                self._send_static(unquote(path.removeprefix("/static/")))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (TypeError, ValueError, RuntimeError) as exc:
            self._send_json(self._error_payload(str(exc)), HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/new":
                self._send_json(self.controller.reset(self._read_json()))
            elif path == "/api/move":
                body = self._read_json()
                self._send_json(self.controller.submit_move(int(body["q"]), int(body["r"])))
            elif path == "/api/debug/analyze":
                self._send_json(_debug_analyze(self._read_json()))
            elif path == "/api/debug/search":
                self._send_json(_debug_search(self._read_json()))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except MoveConflict as exc:
            self._send_json({"error": str(exc), "state": self.controller.state()}, HTTPStatus.CONFLICT)
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json({"error": str(exc), "state": self.controller.state()}, HTTPStatus.BAD_REQUEST)
        except RuntimeError as exc:
            self._send_json(self._error_payload(str(exc)), HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _accepts_gzip(self) -> bool:
        return "gzip" in (self.headers.get("Accept-Encoding") or "").lower()

    def _send_body(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        *,
        cache_control: str | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
        gzip_body: bytes | None = None,
        allow_gzip: bool = True,
    ) -> None:
        """Write one response with conditional-GET (304), gzip, and Content-Length.

        ``gzip_body`` lets callers pass pre-compressed bytes (static assets) so they
        are not re-gzipped per request; otherwise the body is gzipped on the fly when
        the client accepts it and it is large enough to be worth it.
        """

        if etag is not None and status == HTTPStatus.OK:
            inm = self.headers.get("If-None-Match")
            if inm and any(etag == token.strip() for token in inm.split(",")):
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("ETag", etag)
                if cache_control:
                    self.send_header("Cache-Control", cache_control)
                self.end_headers()
                return

        encoding: str | None = None
        if allow_gzip and self._accepts_gzip():
            if gzip_body is not None:
                body, encoding = gzip_body, "gzip"
            elif len(body) >= GZIP_MIN_BYTES:
                body, encoding = gzip.compress(body, 6), "gzip"

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if encoding is not None:
            self.send_header("Content-Encoding", encoding)
            self.send_header("Vary", "Accept-Encoding")
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        if etag:
            self.send_header("ETag", etag)
        if last_modified:
            self.send_header("Last-Modified", last_modified)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        # ETag + revalidation lets the history screen's periodic re-poll receive a 304
        # (no body) whenever the run data is unchanged since the last fetch.
        etag = _strong_etag(encoded) if status == HTTPStatus.OK else None
        self._send_body(
            encoded,
            "application/json; charset=utf-8",
            status,
            cache_control="no-cache" if etag else None,
            etag=etag,
        )

    def _error_payload(self, message: str) -> dict[str, object]:
        try:
            return {"error": message, "state": self.controller.state()}
        except Exception:
            return {"error": message}

    def _send_static(self, name: str) -> None:
        if (not name) or ("/" in name) or ("\\" in name) or name.startswith("."):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        entry = _static_entry(name)
        if entry is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        raw, gz, etag, last_modified, content_type = entry
        # Versioned assets are cacheable. index.html must revalidate so a changed
        # app.js query string is picked up immediately after a frontend restart.
        cache_control = "no-cache" if name == "index.html" else f"public, max-age={STATIC_MAX_AGE_SECONDS}"
        self._send_body(
            raw,
            content_type,
            cache_control=cache_control,
            etag=etag,
            last_modified=last_modified,
            gzip_body=gz,
        )

    def _send_training_file(self, run_name: str, artifact_path: str) -> None:
        path = _resolve_run_path(run_name, artifact_path)
        if path is None or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        encoded = path.read_bytes()
        suffix = path.suffix.lower()
        # Only gzip text artifacts (.json/.jsonl); .png/.hxr are already compact and
        # would just waste CPU.
        self._send_body(
            encoded,
            ARTIFACT_TYPES.get(suffix, "application/octet-stream"),
            allow_gzip=suffix in {".json", ".jsonl"},
        )


def _query_int(value: str | None) -> int | None:
    if value in {"", None}:
        return None
    return int(value)


def _query_bool(value: str | None, *, default: bool) -> bool:
    if value in {"", None}:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _query_limit(value: str | None, *, default: int, maximum: int) -> int:
    raw = _query_int(value)
    if raw is None:
        return default
    return max(1, min(int(raw), maximum))


def _training_roots() -> tuple[Path, ...]:
    cwd = Path.cwd()
    candidates = (cwd / "runs", cwd / "configs" / "runs")
    roots: list[Path] = []
    seen: set[str] = set()
    for root in candidates:
        resolved = str(root.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(root)
    return tuple(roots)


def _training_runs() -> dict[str, object]:
    runs_by_name: dict[str, dict[str, object]] = {}
    for root in _training_roots():
        if not root.exists():
            continue
        for path in sorted(
            root.iterdir(),
            key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)),
            reverse=True,
        ):
            if not path.is_dir():
                continue
            diagnostics = path / "diagnostics"
            selfplay = path / "selfplay"
            if not diagnostics.exists() and not selfplay.exists():
                continue
            stat = _safe_stat(path)
            current = {
                "name": path.name,
                "path": str(path),
                "diagnostics": str(diagnostics),
                "selfplay": str(selfplay),
                "modified": stat.st_mtime if stat is not None else 0,
            }
            existing = runs_by_name.get(path.name)
            if existing is None or float(current["modified"]) > float(existing["modified"]):
                runs_by_name[path.name] = current
    runs = sorted(runs_by_name.values(), key=lambda item: float(item["modified"]), reverse=True)
    return {"roots": [str(root) for root in _training_roots()], "runs": runs}


def _training_run(name: str) -> dict[str, object]:
    run_dir = _resolve_run_dir(name)
    if run_dir is None:
        raise ValueError("Unknown training run")
    diagnostics_by_epoch = _diagnostics_by_epoch(run_dir)
    live_status = _training_live_status(run_dir)
    epoch_history = _epoch_history(run_dir)
    evaluation_history = _evaluation_history(run_dir)
    artifacts_page = _training_artifacts_overview_page(
        run_dir,
        limit=TRAINING_OVERVIEW_ARTIFACT_LIMIT,
    )
    history_page = _training_history_page_for_runs(
        [(run_dir.name, run_dir)],
        limit=TRAINING_OVERVIEW_HISTORY_LIMIT,
        cursor="",
        source="all",
        winner="all",
        sort="newest",
        query_text="",
        include_total=False,
        diagnostics_cache={run_dir.name: diagnostics_by_epoch},
        live_status_cache={run_dir.name: live_status},
    )
    histories = list(history_page["items"])
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "artifacts": artifacts_page["items"],
        "artifacts_page": {
            "limit": TRAINING_OVERVIEW_ARTIFACT_LIMIT,
            "next_cursor": artifacts_page["next_cursor"],
            "complete": artifacts_page["complete"],
        },
        "histories": histories,
        "history_page": {
            "limit": TRAINING_OVERVIEW_HISTORY_LIMIT,
            "next_cursor": history_page["next_cursor"],
            "complete": history_page["complete"],
            "history_complete": False,
            "recent_history_count": len(histories),
        },
        "diagnostics_by_epoch": diagnostics_by_epoch,
        "epoch_history": epoch_history,
        "evaluation_history": evaluation_history,
        "learning_health": _learning_health(epoch_history, evaluation_history, live_status),
        "status": _training_run_status(run_dir, histories, live_status),
    }


def _training_history(run_name: str, artifact_path: str, record_index: int = 0) -> dict[str, object]:
    path = _resolve_run_path(run_name, artifact_path)
    if path is None or not path.is_file() or path.suffix.lower() != ".hxr":
        raise ValueError("Unknown game history artifact")
    stat = _safe_stat(path)
    if stat is None or stat.st_size <= 0:
        raise ValueError("Game history artifact is empty")

    with HexoRecordFile.open(path) as record_file:
        players = [_record_player_payload(player) for player in record_file.players]
        records = list(record_file.iter_records())

    if not records:
        raise ValueError("Game history artifact contains no games")
    if record_index < 0 or record_index >= len(records):
        raise ValueError(f"Game history record index out of range: {record_index}")

    record = records[record_index]
    state = engine.new_game(seed=record.seed)
    applied_actions: list[int] = []
    for action_id in record.action_ids:
        action_id = int(action_id)
        engine.apply_action(state, engine.PlacementAction(unpack_coord_id(action_id)))
        applied_actions.append(action_id)

    payload = dashboard_state(engine.to_python_state(state))
    payload.update(
        {
            "version": int(stat.st_mtime_ns % 9_000_000_000_000_000),
            "game_id": f"{run_name}:{record.game_id}",
            "mode": "history",
            "players": _players_by_role(players),
            "turn_status": "history",
            "can_submit": False,
            "thinking_player": None,
            "last_bot_decision": None,
            "error": None,
            "match": {
                "players": {item["role"]: item["kind"] for item in players},
                "time_limit": None,
                "seed": record.seed,
            },
            "history": {
                "run": run_name,
                "path": artifact_path,
                "record_index": record_index,
                "record_count": len(records),
                "status": record.status,
                "winner": record.winner,
                "placements": record.placements,
                "action_ids": applied_actions,
                "abort": _abort_payload(record.abort),
            },
            "record_games": [
                {
                    "index": index,
                    "game_id": item.game_id,
                    "status": item.status,
                    "actions": len(item.action_ids),
                    "winner": item.winner,
                }
                for index, item in enumerate(records)
            ],
        }
    )
    return payload


# ---------------------------------------------------------------------------
# Debug tab: position inspection via the CPU inference worker (debug_service).
#
# These endpoints reconstruct a board position from a recorded game (or a raw
# move list) and ask the out-of-process, CPU-only worker what the model thinks —
# policy prior, value distribution, opponent-policy + STV heads, and on-demand
# MCTS. The worker is launched with CUDA_VISIBLE_DEVICES="" so it never contends
# for the training GPU, and results are cached so re-opening a view is instant.
# ---------------------------------------------------------------------------

_DEBUG_CKPT_EPOCH_RE = re.compile(r"epoch(\d+)\.pt$")
# The STV graft widened the value/STV readout heads at RL epoch 7 (also visible
# as a ~29 MB -> ~31 MB checkpoint-size jump); used only for a display hint.
_DEBUG_GRAFT_EPOCH = 7
_DEBUG_GRAFT_SIZE_BYTES = 30_500_000


def _debug_worker() -> "debug_service.DebugWorker":
    return debug_service.get_worker()


def _debug_checkpoints(run_name: str) -> dict[str, object]:
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None:
        raise ValueError("Unknown training run")
    ckpt_dir = run_dir / "checkpoints"
    items: list[dict[str, object]] = []
    if ckpt_dir.is_dir():
        for entry in os.scandir(ckpt_dir):
            if not entry.is_file() or not entry.name.endswith(".pt"):
                continue
            match = _DEBUG_CKPT_EPOCH_RE.search(entry.name)
            epoch = int(match.group(1)) if match else None
            stat = _safe_stat(Path(entry.path))
            size = int(stat.st_size) if stat else 0
            if epoch is not None:
                graft = "post" if epoch >= _DEBUG_GRAFT_EPOCH else "pre"
            elif size:
                graft = "post" if size > _DEBUG_GRAFT_SIZE_BYTES else "pre"
            else:
                graft = None
            items.append(
                {
                    "name": entry.name,
                    "epoch": epoch,
                    "size": size,
                    "mtime": int(stat.st_mtime) if stat else 0,
                    "latest": entry.name == "hexgt_rl_latest.pt",
                    "graft": graft,
                }
            )
    items.sort(key=lambda x: (not x["latest"], -(x["epoch"] if x["epoch"] is not None else -1), str(x["name"])))
    return {"run": run_name, "checkpoints": items, "worker": _debug_worker().status()}


def _debug_games(run_name: str, source: str) -> dict[str, object]:
    """List the recorded game files (``.hxr``) available for inspection. Self-play
    files are one-per-epoch; evaluation files live in ``eval*/`` subdirectories."""

    run_dir = _resolve_run_dir(run_name)
    if run_dir is None:
        raise ValueError("Unknown training run")

    def rel(p: Path) -> str:
        return p.relative_to(run_dir).as_posix()

    def hxr_in(directory: Path, recurse: bool) -> list[Path]:
        found: list[Path] = []
        if not directory.is_dir():
            return found
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.endswith(".hxr"):
                found.append(Path(entry.path))
            elif recurse and entry.is_dir():
                for sub in os.scandir(entry.path):
                    if sub.is_file() and sub.name.endswith(".hxr"):
                        found.append(Path(sub.path))
        return found

    files: list[Path] = []
    if source in ("selfplay", "all"):
        files += hxr_in(run_dir / "selfplay", recurse=False)
    if source in ("evaluation", "all"):
        files += hxr_in(run_dir / "evaluation", recurse=True)
        files += hxr_in(run_dir / "eval", recurse=True)

    items = []
    for path in files:
        stat = _safe_stat(path)
        items.append(
            {
                "path": rel(path),
                "name": path.name,
                "size": int(stat.st_size) if stat else 0,
                "mtime": int(stat.st_mtime) if stat else 0,
            }
        )
    items.sort(key=lambda x: str(x["path"]), reverse=True)
    return {"run": run_name, "source": source, "games": items}


def _debug_resolve_checkpoint(run_name: str, checkpoint: str) -> Path:
    name = checkpoint.strip()
    if not name:
        raise ValueError("checkpoint is required")
    if "/" in name or "\\" in name:  # accept a bare filename only, resolve under the run
        name = Path(name).name
    path = _resolve_run_path(run_name, f"checkpoints/{name}")
    if path is None or not path.is_file():
        raise ValueError(f"Unknown checkpoint: {checkpoint}")
    return path


def _debug_open_record(run_name: str, artifact_path: str, record_index: int):
    path = _resolve_run_path(run_name, artifact_path)
    if path is None or not path.is_file() or path.suffix.lower() != ".hxr":
        raise ValueError("Unknown game history artifact")
    with HexoRecordFile.open(path) as record_file:
        players = [_record_player_payload(player) for player in record_file.players]
        records = list(record_file.iter_records())
    if not records:
        raise ValueError("Game history artifact contains no games")
    if record_index < 0 or record_index >= len(records):
        raise ValueError(f"Game history record index out of range: {record_index}")
    return records[record_index], players, records


def _debug_position(run_name: str, artifact_path: str, record_index: int, ply: int) -> dict[str, object]:
    record, players, records = _debug_open_record(run_name, artifact_path, record_index)
    action_ids = [int(a) for a in record.action_ids]
    total = len(action_ids)
    ply = max(0, min(int(ply), total))
    state = engine.new_game(seed=record.seed)
    for action_id in action_ids[:ply]:
        engine.apply_action(state, engine.PlacementAction(unpack_coord_id(action_id)))

    payload = dashboard_state(engine.to_python_state(state))
    payload.update(
        {
            "mode": "debug",
            "game_id": f"{run_name}:{record.game_id}",
            "players": _players_by_role(players),
            "debug": {
                "run": run_name,
                "path": artifact_path,
                "record_index": record_index,
                "record_count": len(records),
                "ply": ply,
                "total": total,
                "action_ids": action_ids,
                "last_action_id": action_ids[ply - 1] if ply > 0 else None,
                "winner": record.winner,
                "status": record.status,
                "seed": record.seed,
            },
            "record_games": [
                {
                    "index": index,
                    "game_id": item.game_id,
                    "status": item.status,
                    "actions": len(item.action_ids),
                    "winner": item.winner,
                }
                for index, item in enumerate(records)
            ],
        }
    )
    return payload


def _debug_action_prefix(body: dict[str, Any]) -> tuple[str, list[int]]:
    """Resolve (run, action_id prefix) from a debug request body. Either an
    explicit ``action_ids`` list (paste/import) or a recorded game + ``ply``."""

    run = str(body.get("run", ""))
    raw = body.get("action_ids")
    if raw is not None:
        return run, [int(a) for a in raw]
    record, _players, _records = _debug_open_record(run, str(body.get("path", "")), int(body.get("record", 0) or 0))
    full = [int(a) for a in record.action_ids]
    ply = int(body.get("ply", len(full)))
    ply = max(0, min(ply, len(full)))
    return run, full[:ply]


def _debug_signature(prefix: str, ckpt_path: Path, action_ids: list[int], n: object) -> str:
    return json.dumps([prefix, str(ckpt_path), action_ids, n], separators=(",", ":"))


def _debug_analyze(body: dict[str, Any]) -> dict[str, object]:
    run, action_ids = _debug_action_prefix(body)
    ckpt_path = _debug_resolve_checkpoint(run, str(body.get("checkpoint", "")))
    n = body.get("n")
    signature = _debug_signature("analyze", ckpt_path, action_ids, n)
    return _debug_worker().cached(
        signature, "analyze", checkpoint=str(ckpt_path), action_ids=action_ids, n=n
    )


def _debug_search(body: dict[str, Any]) -> dict[str, object]:
    run, action_ids = _debug_action_prefix(body)
    ckpt_path = _debug_resolve_checkpoint(run, str(body.get("checkpoint", "")))
    n = body.get("n")
    visits = int(body.get("visits", 512))
    c_puct = float(body.get("c_puct", 1.5))
    visits = max(1, min(visits, 20_000))  # bound CPU work per request
    signature = _debug_signature(f"search:{visits}:{c_puct}", ckpt_path, action_ids, n)
    return _debug_worker().cached(
        signature,
        "search",
        timeout=debug_service.DEFAULT_TIMEOUT,
        checkpoint=str(ckpt_path),
        action_ids=action_ids,
        visits=visits,
        c_puct=c_puct,
        n=n,
    )


def _debug_recorded_trajectory(run_dir: Path, artifact_path: str, game_id: object) -> list[dict[str, object]]:
    """Best-effort recorded root_value per move from ``eval/epoch_*_examples.json``.

    Only self-play example games carry per-move traces, so this returns ``[]`` when
    no matching trace exists. Values are normalized to player-0's perspective."""

    match = re.search(r"epoch_(\d+)", Path(artifact_path).name)
    if match is None:
        return []
    epoch = int(match.group(1))
    examples_path = run_dir / "eval" / f"epoch_{epoch:06d}_examples.json"
    if not examples_path.is_file():
        return []
    try:
        with examples_path.open("r", encoding="utf-8") as handle:
            games = json.load(handle)
    except (OSError, ValueError):
        return []
    trace = None
    for game in games if isinstance(games, list) else []:
        if str(game.get("game_id")) == str(game_id):
            trace = game.get("moves") or []
            break
    if not trace:
        return []
    out = []
    for move in trace:
        rv = move.get("root_value")
        if rv is None:
            continue
        ply = int(move.get("move", 0))
        player0 = str(move.get("player", "player0")).endswith("0")
        out.append({"ply": ply, "root_value": float(rv), "root_value_p0": float(rv) if player0 else -float(rv)})
    return out


def _debug_trajectory(run_name: str, artifact_path: str, record_index: int, checkpoint: str, max_points: int = 160) -> dict[str, object]:
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None:
        raise ValueError("Unknown training run")
    record, _players, _records = _debug_open_record(run_name, artifact_path, record_index)
    action_ids = [int(a) for a in record.action_ids]
    total = len(action_ids)
    ckpt_path = _debug_resolve_checkpoint(run_name, checkpoint)

    # Evaluate plies 0..total, strided so a long game stays bounded (one forward
    # per point). The stride is surfaced so the UI never implies full coverage.
    stride = max(1, -(-(total + 1) // max_points))
    plies = list(range(0, total + 1, stride))
    if plies[-1] != total:
        plies.append(total)
    sequences = [action_ids[:p] for p in plies]

    signature = _debug_signature(f"trajectory:{stride}", ckpt_path, action_ids, max_points)
    raw = _debug_worker().cached(
        signature, "reeval", checkpoint=str(ckpt_path), sequences=sequences, timeout=debug_service.DEFAULT_TIMEOUT
    )
    reeval = []
    for entry in raw.get("values", []):
        cp = int(entry.get("current_player", 0))
        value = float(entry.get("value", 0.0))
        reeval.append({"ply": int(entry["ply"]), "value": value, "current_player": cp,
                       "value_p0": value if cp == 0 else -value})

    return {
        "run": run_name,
        "path": artifact_path,
        "record": record_index,
        "total": total,
        "stride": stride,
        "checkpoint": ckpt_path.name,
        "winner": record.winner,
        "reeval": reeval,
        "recorded": _debug_recorded_trajectory(run_dir, artifact_path, record.game_id),
    }


def _training_artifacts_page(
    *,
    run_name: str,
    limit: int = ARTIFACT_PAGE_DEFAULT_LIMIT,
    cursor: str = "",
    kind: str = "all",
) -> dict[str, object]:
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None:
        raise ValueError("Unknown training run")
    return _training_artifacts_page_for_run(run_dir, limit=limit, cursor=cursor, kind=kind)


def _training_artifacts_overview_page(run_dir: Path, *, limit: int) -> dict[str, object]:
    paths: list[Path] = []

    def add_direct(root: Path, suffixes: set[str] | frozenset[str] = ARTIFACT_SUFFIXES) -> None:
        try:
            entries = list(os.scandir(root))
        except OSError:
            return
        for entry in entries:
            if entry.is_file() and Path(entry.name).suffix.lower() in suffixes:
                paths.append(Path(entry.path))

    def add_recent_child_dirs(
        root: Path,
        suffixes: set[str] | frozenset[str],
        *,
        max_dirs: int = 2,
    ) -> None:
        try:
            entries = list(os.scandir(root))
        except OSError:
            return
        dirs: list[tuple[float, Path]] = []
        for entry in entries:
            if entry.is_file() and Path(entry.name).suffix.lower() in suffixes:
                paths.append(Path(entry.path))
            elif entry.is_dir():
                try:
                    dirs.append((entry.stat().st_mtime, Path(entry.path)))
                except OSError:
                    continue
        for _, directory in sorted(dirs, reverse=True)[:max_dirs]:
            add_direct(directory, suffixes)

    def add_recursive_limited(
        root: Path,
        suffixes: set[str] | frozenset[str] = ARTIFACT_SUFFIXES,
        *,
        max_files: int = 100,
    ) -> None:
        if not root.is_dir():
            return
        for path in _iter_training_files(root, suffix=None):
            if path.is_file() and path.suffix.lower() in suffixes:
                paths.append(path)
                if len(paths) >= max_files:
                    return

    add_direct(run_dir)
    add_direct(run_dir / "diagnostics")
    add_direct(run_dir / "selfplay", {".hxr"})
    add_recent_child_dirs(run_dir / "evaluation", {".hxr"})
    add_direct(run_dir / "checkpoints")
    add_recursive_limited(run_dir / "bootstrap")

    unique = {str(path.resolve()): path for path in paths}
    paths = list(unique.values())
    paths.sort(key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)), reverse=True)
    selected = paths[: limit + 1]
    return {
        "run": run_dir.name,
        "items": [_artifact_payload(run_dir, path) for path in selected[:limit]],
        "next_cursor": str(limit) if len(selected) > limit else None,
        "complete": len(selected) <= limit,
        "scanned_files": len(paths),
    }


def _training_artifacts_page_for_run(
    run_dir: Path,
    *,
    limit: int,
    cursor: str,
    kind: str,
) -> dict[str, object]:
    offset = max(0, _query_int(cursor) or 0)
    wanted_kind = str(kind or "all").lower()
    paths = [
        path
        for path in _iter_training_files(run_dir)
        if path.is_file()
        and path.suffix.lower() in ARTIFACT_SUFFIXES
        and (wanted_kind == "all" or path.suffix.lower().lstrip(".") == wanted_kind)
    ]
    paths.sort(key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)), reverse=True)
    selected = paths[offset : offset + limit + 1]
    items = [_artifact_payload(run_dir, path) for path in selected[:limit]]
    next_offset = offset + limit
    return {
        "run": run_dir.name,
        "items": items,
        "next_cursor": str(next_offset) if len(selected) > limit else None,
        "complete": len(selected) <= limit,
        "scanned_files": len(paths),
    }


def _artifact_payload(run_dir: Path, path: Path) -> dict[str, object]:
    rel = path.relative_to(run_dir).as_posix()
    stat = _safe_stat(path)
    suffix = path.suffix.lower()
    artifact: dict[str, object] = {
        "path": rel,
        "name": path.name,
        "bytes": stat.st_size if stat is not None else 0,
        "modified": stat.st_mtime if stat is not None else 0,
        "kind": suffix.lstrip(".") or "file",
        "loadable_history": False,
        "history_count": 0,
    }
    if suffix == ".json":
        payload = _read_json_file(path)
        artifact["summary"] = _artifact_summary(payload)
    elif suffix == ".hxr" and _is_loadable_history_path(rel) and stat is not None and stat.st_size > 0:
        rows = _hxr_base_rows(path, run_dir)
        history_count = len(rows)
        artifact["loadable_history"] = history_count > 0
        artifact["history_count"] = history_count
    return artifact


def _training_history_page(
    *,
    run_name: str,
    limit: int = HISTORY_PAGE_DEFAULT_LIMIT,
    cursor: str = "",
    source: str = "all",
    winner: str = "all",
    sort: str = "newest",
    query_text: str = "",
    include_total: bool = True,
) -> dict[str, object]:
    run_infos = _history_run_infos(run_name)
    if not run_infos:
        raise ValueError("Unknown training run")
    return _training_history_page_for_runs(
        run_infos,
        limit=limit,
        cursor=cursor,
        source=source,
        winner=winner,
        sort=sort,
        query_text=query_text,
        include_total=include_total,
    )


def _training_history_count(
    *,
    run_name: str,
    source: str = "all",
    winner: str = "all",
    query_text: str = "",
) -> dict[str, object]:
    run_infos = _history_run_infos(run_name)
    if not run_infos:
        raise ValueError("Unknown training run")
    total_matches, scanned_files, scanned_games = _training_history_count_for_runs(
        run_infos,
        source=source,
        winner=winner,
        query_text=query_text,
    )
    return {
        "total_matches": total_matches,
        "scanned_files": scanned_files,
        "scanned_games": scanned_games,
    }


def _training_history_count_for_runs(
    run_infos: list[tuple[str, Path]],
    *,
    source: str,
    winner: str,
    query_text: str,
) -> tuple[int, int, int]:
    total_matches = 0
    scanned_files = 0
    scanned_games = 0
    can_count_without_rows = _history_filter_matches_all(winner=winner, query_text=query_text)
    diagnostics_cache: dict[str, dict[str, object]] | None = None
    live_status_cache: dict[str, dict[str, object]] | None = None
    if not can_count_without_rows:
        diagnostics_cache = {
            run_name: _diagnostics_by_epoch(run_dir)
            for run_name, run_dir in run_infos
        }
        live_status_cache = {
            run_name: _training_live_status(run_dir)
            for run_name, run_dir in run_infos
        }

    for run_name, run_dir, path, _stat in _history_files_for_runs(run_infos, source=source, reverse=True):
        scanned_files += 1
        if can_count_without_rows:
            record_count = _hxr_record_count(path, run_dir)
            total_matches += record_count
            scanned_games += record_count
            continue
        rows = _history_rows_for_file(
            run_name,
            run_dir,
            path,
            diagnostics_cache=diagnostics_cache,
            live_status_cache=live_status_cache,
            reverse_records=False,
        )
        scanned_games += len(rows)
        total_matches += sum(1 for row in rows if _history_row_matches(row, winner=winner, query_text=query_text))

    return total_matches, scanned_files, scanned_games


def _training_history_page_for_runs(
    run_infos: list[tuple[str, Path]],
    *,
    limit: int,
    cursor: str,
    source: str,
    winner: str,
    sort: str,
    query_text: str,
    include_total: bool = True,
    diagnostics_cache: dict[str, dict[str, object]] | None = None,
    live_status_cache: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    diagnostics_cache = diagnostics_cache or {
        run_name: _diagnostics_by_epoch(run_dir)
        for run_name, run_dir in run_infos
    }
    live_status_cache = live_status_cache or {
        run_name: _training_live_status(run_dir)
        for run_name, run_dir in run_infos
    }
    normalized_sort = sort if sort in {"newest", "oldest", "longest", "shortest", "winner"} else "newest"
    if normalized_sort in {"longest", "shortest", "winner"}:
        rows, scanned_files, scanned_games = _collect_history_rows(
            run_infos,
            source=source,
            diagnostics_cache=diagnostics_cache,
            live_status_cache=live_status_cache,
        )
        rows = [row for row in rows if _history_row_matches(row, winner=winner, query_text=query_text)]
        rows.sort(key=lambda item: _history_complete_sort_key(item, normalized_sort))
        offset = max(0, _query_int(cursor) or 0)
        selected = rows[offset : offset + limit]
        next_offset = offset + limit
        return {
            "items": selected,
            "next_cursor": str(next_offset) if next_offset < len(rows) else None,
            "complete": next_offset >= len(rows),
            "total_matches": len(rows),
            "scanned_files": scanned_files,
            "scanned_games": scanned_games,
            "sort": normalized_sort,
        }

    return _training_history_streaming_page(
        run_infos,
        limit=limit,
        cursor=cursor,
        source=source,
        winner=winner,
        sort=normalized_sort,
        query_text=query_text,
        include_total=include_total,
        diagnostics_cache=diagnostics_cache,
        live_status_cache=live_status_cache,
    )


def _training_history_streaming_page(
    run_infos: list[tuple[str, Path]],
    *,
    limit: int,
    cursor: str,
    source: str,
    winner: str,
    sort: str,
    query_text: str,
    include_total: bool,
    diagnostics_cache: dict[str, dict[str, object]] | None,
    live_status_cache: dict[str, dict[str, object]] | None,
) -> dict[str, object]:
    reverse = sort != "oldest"
    cursor_key = _decode_history_cursor(cursor)
    passed_cursor = cursor_key is None
    selected: list[dict[str, object]] = []
    has_more = False
    total_matches: int | None = 0 if include_total else None
    can_count_without_rows = include_total and _history_filter_matches_all(winner=winner, query_text=query_text)
    scanned_files = 0
    scanned_games = 0

    for run_name, run_dir, path, _stat in _history_files_for_runs(run_infos, source=source, reverse=reverse):
        scanned_files += 1
        if can_count_without_rows and has_more:
            record_count = _hxr_record_count(path, run_dir)
            total_matches = (total_matches or 0) + record_count
            scanned_games += record_count
            continue
        rows = _history_rows_for_file(
            run_name,
            run_dir,
            path,
            diagnostics_cache=diagnostics_cache,
            live_status_cache=live_status_cache,
            reverse_records=reverse,
        )
        scanned_games += len(rows)
        if can_count_without_rows:
            total_matches = (total_matches or 0) + len(rows)
        for row in rows:
            matches = True if can_count_without_rows else _history_row_matches(row, winner=winner, query_text=query_text)
            if include_total and not can_count_without_rows and matches:
                total_matches = (total_matches or 0) + 1
            row_key = _history_cursor_key(row)
            if not passed_cursor:
                if row_key == cursor_key:
                    passed_cursor = True
                continue
            if not matches:
                continue
            if len(selected) >= limit:
                has_more = True
                if include_total:
                    if can_count_without_rows:
                        break
                    continue
                break
            selected.append(row)
        if has_more and not include_total:
            break

    return {
        "items": selected,
        "next_cursor": _encode_history_cursor(_history_cursor_key(selected[-1])) if has_more and selected else None,
        "complete": not has_more,
        "total_matches": total_matches,
        "scanned_files": scanned_files,
        "scanned_games": scanned_games,
        "sort": sort,
    }


def _history_run_infos(run_name: str) -> list[tuple[str, Path]]:
    if run_name == HISTORY_ALL_RUNS:
        infos: list[tuple[str, Path]] = []
        for item in _training_runs()["runs"]:
            resolved = _resolve_run_dir(str(item.get("name") or ""))
            if resolved is not None:
                infos.append((resolved.name, resolved))
        return infos
    run_dir = _resolve_run_dir(run_name)
    return [] if run_dir is None else [(run_dir.name, run_dir)]


def _collect_history_rows(
    run_infos: list[tuple[str, Path]],
    *,
    source: str,
    diagnostics_cache: dict[str, dict[str, object]] | None,
    live_status_cache: dict[str, dict[str, object]] | None,
) -> tuple[list[dict[str, object]], int, int]:
    rows: list[dict[str, object]] = []
    scanned_files = 0
    scanned_games = 0
    for run_name, run_dir, path, _stat in _history_files_for_runs(run_infos, source=source, reverse=True):
        scanned_files += 1
        file_rows = _history_rows_for_file(
            run_name,
            run_dir,
            path,
            diagnostics_cache=diagnostics_cache,
            live_status_cache=live_status_cache,
            reverse_records=False,
        )
        rows.extend(file_rows)
        scanned_games += len(file_rows)
    return rows, scanned_files, scanned_games


def _history_files_for_runs(
    run_infos: list[tuple[str, Path]],
    *,
    source: str,
    reverse: bool,
) -> list[tuple[str, Path, Path, os.stat_result]]:
    files: list[tuple[str, Path, Path, os.stat_result]] = []
    for run_name, run_dir in run_infos:
        for path, stat in _iter_history_artifact_files(run_dir, source=source):
            if stat.st_size <= 0:
                continue
            rel = path.relative_to(run_dir).as_posix()
            if not _is_loadable_history_path(rel):
                continue
            files.append((run_name, run_dir, path, stat))
    files.sort(
        key=lambda item: (
            _epoch_from_artifact_path(item[2].relative_to(item[1]).as_posix()) or 0,
            item[3].st_mtime,
            str(item[0]),
            item[2].relative_to(item[1]).as_posix(),
        ),
        reverse=reverse,
    )
    return files


def _iter_history_artifact_files(
    run_dir: Path,
    *,
    source: str,
) -> list[tuple[Path, os.stat_result]]:
    normalized_source = str(source or "all").lower()
    roots: list[Path] = []
    if normalized_source in {"", "all", "selfplay"}:
        roots.append(run_dir / "selfplay")
    if normalized_source in {"", "all", "evaluation"}:
        roots.append(run_dir / "evaluation")

    files: list[tuple[Path, os.stat_result]] = []
    for root in roots:
        if not root.is_dir():
            continue
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                entries = list(os.scandir(current))
            except OSError:
                continue
            for entry in entries:
                name = entry.name
                if name.startswith(".") or name in TRAINING_SCAN_EXCLUDED_DIRS:
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False) and name.endswith(".hxr"):
                        files.append((Path(entry.path), entry.stat(follow_symlinks=False)))
                except OSError:
                    continue
    return files


def _history_rows_for_file(
    run_name: str,
    run_dir: Path,
    path: Path,
    *,
    diagnostics_cache: dict[str, dict[str, object]] | None,
    live_status_cache: dict[str, dict[str, object]] | None,
    reverse_records: bool,
) -> list[dict[str, object]]:
    rel = path.relative_to(run_dir).as_posix()
    base_rows = _hxr_base_rows(path, run_dir)
    if reverse_records:
        base_rows = list(reversed(base_rows))
    epoch = _epoch_from_artifact_path(rel)
    source = _history_source(rel)
    diagnostics_by_epoch = (
        diagnostics_cache.get(run_name)
        if diagnostics_cache is not None and run_name in diagnostics_cache
        else _diagnostics_by_epoch(run_dir)
    )
    live_status = (
        live_status_cache.get(run_name)
        if live_status_cache is not None and run_name in live_status_cache
        else _training_live_status(run_dir)
    )
    diagnostics = dict(diagnostics_by_epoch.get(str(epoch), {})) if epoch is not None else {}
    if (
        live_status
        and source == "selfplay"
        and epoch is not None
        and int(live_status.get("current_epoch") or -1) == int(epoch)
        and "selfplay" not in diagnostics
    ):
        diagnostics["live"] = {
            "path": rel,
            "summary": _live_history_diagnostic_summary(live_status),
        }
    brief = _history_diagnostics_brief(diagnostics)
    rows: list[dict[str, object]] = []
    for row in base_rows:
        item = dict(row)
        item["run"] = run_name
        item["diagnostics"] = brief
        rows.append(item)
    return rows


def _hxr_base_rows(path: Path, run_dir: Path) -> list[dict[str, object]]:
    stat = _safe_stat(path)
    if stat is None or stat.st_size <= 0:
        return []
    cache_key = str(path.resolve())
    with _training_cache_lock:
        hit = _hxr_history_cache.get(cache_key)
        if hit is not None and hit[0] == stat.st_mtime_ns and hit[1] == stat.st_size:
            return [dict(row) for row in hit[2]]

    rel = path.relative_to(run_dir).as_posix()
    try:
        with HexoRecordFile.open(path) as record_file:
            players = [_record_player_payload(player) for player in record_file.players]
            records = list(record_file.iter_records())
    except Exception:
        return []

    rows: list[dict[str, object]] = []
    epoch = _epoch_from_artifact_path(rel)
    source = _history_source(rel)
    players_by_role = _players_by_role(players)
    for index, record in enumerate(records):
        length = int(record.placements or len(record.action_ids))
        rows.append(
            {
                "path": rel,
                "record_index": index,
                "game_id": record.game_id,
                "status": record.status,
                "winner": record.winner,
                "winner_label": _winner_label(record.winner),
                "length": length,
                "actions": len(record.action_ids),
                "epoch": epoch,
                "source": source,
                "seed": record.seed,
                "players": players_by_role,
                "modified": stat.st_mtime,
                "modified_ns": stat.st_mtime_ns,
                "bytes": stat.st_size,
                "abort": _abort_payload(record.abort),
            }
        )
    with _training_cache_lock:
        _hxr_history_cache[cache_key] = (stat.st_mtime_ns, stat.st_size, [dict(row) for row in rows])
    return rows


def _hxr_record_count(path: Path, run_dir: Path) -> int:
    stat = _safe_stat(path)
    if stat is None or stat.st_size <= 0:
        return 0
    cache_key = str(path.resolve())
    with _training_cache_lock:
        hit = _hxr_history_cache.get(cache_key)
        if hit is not None and hit[0] == stat.st_mtime_ns and hit[1] == stat.st_size:
            return len(hit[2])
        count_hit = _hxr_count_cache.get(cache_key)
        if count_hit is not None and count_hit[0] == stat.st_mtime_ns and count_hit[1] == stat.st_size:
            return count_hit[2]

    try:
        with HexoRecordFile.open(path) as record_file:
            count = sum(1 for _ in record_file.iter_records())
    except Exception:
        return 0
    with _training_cache_lock:
        _hxr_count_cache[cache_key] = (stat.st_mtime_ns, stat.st_size, count)
    return count


def _is_loadable_history_path(rel: str) -> bool:
    return rel.split("/", 1)[0] in {"selfplay", "evaluation"}


def _history_filter_matches_all(*, winner: str, query_text: str) -> bool:
    return str(winner or "all").lower() in {"", "all"} and not str(query_text or "").strip()


def _history_row_matches(row: dict[str, object], *, winner: str, query_text: str) -> bool:
    normalized_winner = str(winner or "all").lower()
    if normalized_winner == "none":
        if row.get("winner") is not None:
            return False
    elif normalized_winner not in {"", "all"} and row.get("winner") != normalized_winner:
        return False

    query = str(query_text or "").strip().lower()
    if not query:
        return True
    players = row.get("players") if isinstance(row.get("players"), dict) else {}
    diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
    haystack = " ".join(
        str(value)
        for value in (
            row.get("game_id"),
            row.get("run"),
            row.get("path"),
            row.get("status"),
            row.get("source"),
            row.get("epoch"),
            row.get("seed"),
            row.get("winner_label"),
            row.get("length"),
            history_player_label(players.get("player0") if isinstance(players, dict) else None),
            history_player_label(players.get("player1") if isinstance(players, dict) else None),
            json.dumps(diagnostics, sort_keys=True) if diagnostics else "",
        )
        if value is not None
    ).lower()
    return query in haystack


def history_player_label(player: object) -> str:
    if not isinstance(player, dict):
        return "Unknown"
    return str(player.get("label") or player.get("kind") or "Unknown")


def _history_complete_sort_key(row: dict[str, object], sort: str) -> tuple[object, ...]:
    newest = (
        -float(row.get("modified") or 0.0),
        -int(row.get("epoch") or 0),
        str(row.get("run") or ""),
        str(row.get("path") or ""),
        -int(row.get("record_index") or 0),
    )
    if sort == "longest":
        return (-int(row.get("length") or row.get("actions") or 0),) + newest
    if sort == "shortest":
        return (int(row.get("length") or row.get("actions") or 0),) + newest
    if sort == "winner":
        return (str(row.get("winner_label") or _winner_label(row.get("winner"))),) + newest
    return newest


def _history_cursor_key(row: dict[str, object]) -> list[object]:
    return [
        row.get("run"),
        row.get("path"),
        int(row.get("record_index") or 0),
        int(row.get("modified_ns") or 0),
    ]


def _encode_history_cursor(key: list[object]) -> str:
    return json.dumps(key, separators=(",", ":"))


def _decode_history_cursor(cursor: str) -> list[object] | None:
    if not cursor:
        return None
    try:
        value = json.loads(cursor)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


def _record_player_payload(player: object) -> dict[str, object]:
    role = str(getattr(player, "role", ""))
    label = getattr(player, "label", None)
    player_id = str(getattr(player, "player_id", role or "player"))
    kind = "manual"
    lowered = player_id.lower()
    if "sealbot" in lowered:
        kind = "sealbot-best" if "best" in lowered else "sealbot-current"
    elif "dense" in lowered:
        kind = "dense-cnn"
    return {
        "role": role,
        "kind": kind,
        "label": str(label or player_id),
        "player_id": player_id,
    }


def _training_histories(
    run_dir: Path,
    diagnostics_by_epoch: dict[str, object],
    live_status: dict[str, object] | None = None,
) -> dict[str, list[dict[str, object]]]:
    histories: dict[str, list[dict[str, object]]] = {}
    for path in sorted(_iter_training_files(run_dir, suffix=".hxr")):
        stat = _safe_stat(path)
        if not path.is_file() or stat is None or stat.st_size <= 0:
            continue
        rel = path.relative_to(run_dir).as_posix()
        if rel.split("/", 1)[0] not in {"selfplay", "evaluation"}:
            continue
        try:
            with HexoRecordFile.open(path) as record_file:
                players = [_record_player_payload(player) for player in record_file.players]
                records = list(record_file.iter_records())
        except Exception:
            continue
        epoch = _epoch_from_artifact_path(rel)
        source = _history_source(rel)
        diagnostics = dict(diagnostics_by_epoch.get(str(epoch), {})) if epoch is not None else {}
        if (
            live_status
            and source == "selfplay"
            and epoch is not None
            and int(live_status.get("current_epoch") or -1) == int(epoch)
            and "selfplay" not in diagnostics
        ):
            diagnostics["live"] = {
                "path": rel,
                "summary": _live_history_diagnostic_summary(live_status),
            }
        entries: list[dict[str, object]] = []
        for index, record in enumerate(records):
            length = int(record.placements or len(record.action_ids))
            entries.append(
                {
                    "path": rel,
                    "record_index": index,
                    "game_id": record.game_id,
                    "status": record.status,
                    "winner": record.winner,
                    "winner_label": _winner_label(record.winner),
                    "length": length,
                    "actions": len(record.action_ids),
                    "epoch": epoch,
                    "source": source,
                    "seed": record.seed,
                    "players": _players_by_role(players),
                    "diagnostics": _history_diagnostics_brief(diagnostics),
                    "modified": stat.st_mtime,
                    "bytes": stat.st_size,
                    "abort": _abort_payload(record.abort),
                }
            )
        if entries:
            histories[rel] = entries
    return histories


def _history_diagnostics_brief(diagnostics: dict[str, object]) -> dict[str, object]:
    return {
        label: diagnostics[label]
        for label in ("selfplay", "evaluation")
        if label in diagnostics
    }


def _iter_training_files(run_dir: Path, *, suffix: str | None = None) -> list[Path]:
    files: list[Path] = []
    for root, dirs, names in os.walk(run_dir):
        dirs[:] = [
            name
            for name in dirs
            if name not in TRAINING_SCAN_EXCLUDED_DIRS and not name.startswith(".")
        ]
        root_path = Path(root)
        for name in names:
            if suffix is not None and not name.endswith(suffix):
                continue
            files.append(root_path / name)
    return files


def _diagnostics_by_epoch(run_dir: Path) -> dict[str, object]:
    by_epoch: dict[str, dict[str, object]] = {}
    diagnostics_dir = run_dir / "diagnostics"
    if not diagnostics_dir.exists():
        return by_epoch
    for path in sorted(diagnostics_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        epoch = _epoch_from_artifact_path(path.name)
        if epoch is None and isinstance(payload, dict) and payload.get("epoch") is not None:
            try:
                epoch = int(payload["epoch"])
            except (TypeError, ValueError):
                epoch = None
        if epoch is None:
            continue
        key = str(epoch)
        by_epoch.setdefault(key, {})
        label = _diagnostic_label(path.name)
        summary = _artifact_summary(payload)
        if summary:
            by_epoch[key][label] = {
                "path": f"diagnostics/{path.name}",
                "summary": summary,
            }
    return by_epoch


def _evaluation_history(run_dir: Path) -> list[dict[str, object]]:
    diagnostics_dir = run_dir / "diagnostics"
    if not diagnostics_dir.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(diagnostics_dir.glob("dense_cnn.evaluation.epoch_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        epoch = _epoch_from_artifact_path(path.name)
        if epoch is None and payload.get("epoch") is not None:
            try:
                epoch = int(payload["epoch"])
            except (TypeError, ValueError):
                epoch = None
        stat = _safe_stat(path)
        rows.append(
            {
                "epoch": epoch,
                "status": payload.get("status"),
                "games": payload.get("games"),
                "completed": payload.get("completed"),
                "wins": payload.get("wins"),
                "losses": payload.get("losses"),
                "mean_turns": payload.get("mean_turns"),
                "path": f"diagnostics/{path.name}",
                "modified": stat.st_mtime if stat is not None else 0,
            }
        )
    rows.sort(key=lambda item: int(item.get("epoch") or 0))
    return rows


def _epoch_history(run_dir: Path) -> list[dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    diagnostics_dir = run_dir / "diagnostics"

    if diagnostics_dir.exists():
        for path in sorted(diagnostics_dir.glob("epoch_*.json")):
            payload = _read_json_file(path)
            if not isinstance(payload, dict):
                continue
            result = payload.get("metadata", {}).get("result") if isinstance(payload.get("metadata"), dict) else None
            if not isinstance(result, dict):
                continue
            epoch = _coerce_epoch(result.get("epoch"), path.name)
            if epoch is None:
                continue
            row = rows.setdefault(epoch, {"epoch": epoch})
            row["status"] = payload.get("status")
            row["elapsed_seconds"] = payload.get("elapsed_seconds")
            _merge_epoch_result(row, result)

        for path in sorted(diagnostics_dir.glob("dense_cnn.selfplay.epoch_*.json")):
            payload = _read_json_file(path)
            if not isinstance(payload, dict):
                continue
            epoch = _coerce_epoch(payload.get("epoch"), path.name)
            if epoch is None:
                continue
            row = rows.setdefault(epoch, {"epoch": epoch})
            row["selfplay"] = _selfplay_epoch_summary(payload)

        for path in sorted(diagnostics_dir.glob("dense_cnn.evaluation.epoch_*.json")):
            payload = _read_json_file(path)
            if not isinstance(payload, dict):
                continue
            epoch = _coerce_epoch(payload.get("epoch"), path.name)
            if epoch is None:
                continue
            row = rows.setdefault(epoch, {"epoch": epoch})
            row["evaluation"] = _evaluation_epoch_summary(payload)

        # NOTE: no current producer emits this file; see dense_cnn/selfplay.py (kept for forward-compat / manual drops).
        for path in sorted(diagnostics_dir.glob("dense_cnn.policy_targets.epoch_*.json")):
            payload = _read_json_file(path)
            if not isinstance(payload, dict):
                continue
            epoch = _coerce_epoch(payload.get("epoch"), path.name)
            if epoch is None:
                continue
            preview = payload.get("preview") if isinstance(payload.get("preview"), list) else []
            row = rows.setdefault(epoch, {"epoch": epoch})
            row["d6"] = {
                "mode": payload.get("d6", {}).get("mode") if isinstance(payload.get("d6"), dict) else None,
                "preview_count": len(preview),
                "preview_symmetries": [
                    int(item.get("symmetry"))
                    for item in preview
                    if isinstance(item, dict) and item.get("symmetry") is not None
                ],
            }
            training = row.setdefault("training", {})
            if isinstance(training, dict):
                if isinstance(payload.get("source_summary"), dict):
                    training["source_summary"] = payload["source_summary"]
                if isinstance(payload.get("loss_components"), dict):
                    training["loss_components"] = payload["loss_components"]
                if isinstance(payload.get("policy_imitation"), dict):
                    training["policy_imitation"] = payload["policy_imitation"]

        # NOTE: no current producer emits this file; see dense_cnn/selfplay.py (kept for forward-compat / manual drops).
        for path in sorted(diagnostics_dir.glob("dense_cnn.training_progress.epoch_*.json")):
            payload = _read_json_file(path)
            if not isinstance(payload, dict):
                continue
            epoch = _coerce_epoch(payload.get("epoch"), path.name)
            if epoch is None:
                continue
            row = rows.setdefault(epoch, {"epoch": epoch})
            training = row.setdefault("training", {})
            if isinstance(training, dict):
                training["progress"] = _training_progress_summary(payload)

    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        for path in sorted(checkpoints_dir.glob("epoch_*.pt")):
            epoch = _coerce_epoch(None, path.name)
            if epoch is None:
                continue
            row = rows.setdefault(epoch, {"epoch": epoch})
            stat = _safe_stat(path)
            row["checkpoint"] = {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": stat.st_size if stat is not None else 0,
                "modified": stat.st_mtime if stat is not None else 0,
            }

    for row in rows.values():
        if "status" not in row:
            row["status"] = "partial"
    return [rows[key] for key in sorted(rows)]


def _learning_health(
    epoch_history: list[dict[str, object]],
    evaluation_history: list[dict[str, object]],
    live_status: dict[str, object],
) -> dict[str, object]:
    completed = [row for row in epoch_history if row.get("status") == "completed"]
    latest = completed[-1] if completed else (epoch_history[-1] if epoch_history else {})
    latest_epoch = int(latest.get("epoch") or 0)
    latest_training = latest.get("training") if isinstance(latest.get("training"), dict) else {}
    first_training = completed[0].get("training") if completed and isinstance(completed[0].get("training"), dict) else {}
    latest_loss = _optional_float(latest_training.get("loss"))
    first_loss = _optional_float(first_training.get("loss"))

    evals = [
        item
        for item in evaluation_history
        if _optional_float(item.get("mean_turns")) is not None
    ]
    latest_eval = evals[-1] if evals else {}
    first_eval = evals[0] if evals else {}
    best_eval = max(evals, key=lambda item: (_optional_float(item.get("mean_turns")) or 0.0, int(item.get("wins") or 0)), default={})
    latest_turns = _optional_float(latest_eval.get("mean_turns"))
    first_turns = _optional_float(first_eval.get("mean_turns"))
    best_turns = _optional_float(best_eval.get("mean_turns"))
    latest_wins = int(latest_eval.get("wins") or 0) if latest_eval else 0
    latest_games = int(latest_eval.get("games") or 0) if latest_eval else 0
    latest_selfplay = latest.get("selfplay") if isinstance(latest.get("selfplay"), dict) else {}
    latest_d6 = latest.get("d6") if isinstance(latest.get("d6"), dict) else {}
    latest_source_summary = (
        latest_training.get("source_summary")
        if isinstance(latest_training.get("source_summary"), dict)
        else {}
    )
    latest_source_counts = (
        latest_source_summary.get("source_counts")
        if isinstance(latest_source_summary.get("source_counts"), dict)
        else {}
    )
    latest_classical_fraction = _source_fraction(latest_source_counts, "classical")
    latest_policy_imitation = (
        latest_training.get("policy_imitation")
        if isinstance(latest_training.get("policy_imitation"), dict)
        else {}
    )
    latest_policy_overall = (
        latest_policy_imitation.get("overall")
        if isinstance(latest_policy_imitation.get("overall"), dict)
        else {}
    )
    latest_policy_top1 = _optional_float(latest_policy_overall.get("top1_accuracy"))
    latest_policy_target_mass = _optional_float(latest_policy_overall.get("mean_target_mass"))

    messages: list[str] = []
    status = "collecting"
    if latest_epoch > 0:
        status = "ok"
    if latest_loss is not None and first_loss is not None and latest_loss < first_loss:
        messages.append(f"Training loss improved from {first_loss:.3f} to {latest_loss:.3f}.")
    elif latest_loss is not None:
        messages.append(f"Latest training loss is {latest_loss:.3f}.")

    if latest_turns is None:
        status = "collecting"
        messages.append("No SealBot evaluation result yet for the completed epochs.")
    else:
        delta = latest_turns - (first_turns if first_turns is not None else latest_turns)
        if latest_wins > 0:
            status = "improving"
            messages.append(f"Latest SealBot eval has {latest_wins}/{latest_games} wins.")
        elif delta > 3.0:
            status = "improving"
            messages.append(f"SealBot survival improved by {delta:.1f} turns.")
        elif len(evals) >= 2:
            status = "watch"
            messages.append(f"SealBot survival is flat at {latest_turns:.1f} turns.")
        else:
            messages.append(f"Initial SealBot survival is {latest_turns:.1f} turns.")
        if latest_epoch >= 6 and latest_wins == 0 and (best_turns or 0.0) <= 30.0:
            status = "intervene"
            messages.append("Epoch 6+ is still under 30 turns with no wins; inspect games and training targets before continuing blindly.")
        elif status == "watch":
            messages.append("Keep training for now, but inspect previews if this remains flat near epoch 6.")

    exact_128 = abs((_optional_float(latest_selfplay.get("mcts_sims_per_searched_position")) or 0.0) - 128.0) < 1.0e-6
    speed = _optional_float(latest_selfplay.get("search_positions_per_second"))
    if speed is not None and speed >= 128.0 and exact_128:
        messages.append(f"Self-play speed is healthy at {speed:.0f} pos/s with exact 128 sims.")
    elif speed is not None:
        status = "watch" if status != "intervene" else status
        messages.append(f"Self-play speed needs attention: {speed:.0f} pos/s, exact128={exact_128}.")

    d6_mode = str(latest_d6.get("mode") or "")
    d6_preview = latest_d6.get("preview_symmetries") if isinstance(latest_d6.get("preview_symmetries"), list) else []
    if "random_per_training_expansion" in d6_mode or d6_preview:
        messages.append("D6 training augmentation previews are present.")
    elif latest_epoch > 0:
        status = "watch" if status != "intervene" else status
        messages.append("D6 augmentation preview is missing for the latest epoch.")

    if latest_classical_fraction is not None:
        messages.append(f"Training window classical replay is {latest_classical_fraction * 100.0:.0f}%.")
        if latest_epoch >= 7 and latest_classical_fraction < 0.5:
            status = "watch" if status != "intervene" else status
            messages.append("Classical replay is below the bootstrap floor; inspect sample selection.")
    if latest_policy_target_mass is not None and latest_policy_top1 is not None:
        messages.append(f"Policy imitation top-1 is {latest_policy_top1 * 100.0:.0f}% with {latest_policy_target_mass * 100.0:.1f}% target mass.")

    return {
        "status": status,
        "latest_epoch": latest_epoch or None,
        "current_stage": live_status.get("stage"),
        "latest_loss": latest_loss,
        "loss_delta_from_first": (latest_loss - first_loss) if latest_loss is not None and first_loss is not None else None,
        "latest_eval_mean_turns": latest_turns,
        "best_eval_mean_turns": best_turns,
        "eval_delta_from_first": (latest_turns - first_turns) if latest_turns is not None and first_turns is not None else None,
        "latest_eval_wins": latest_wins,
        "latest_eval_games": latest_games,
        "latest_selfplay_pos_s": speed,
        "latest_exact_128": exact_128,
        "latest_classical_fraction": latest_classical_fraction,
        "latest_policy_top1": latest_policy_top1,
        "latest_policy_target_mass": latest_policy_target_mass,
        "d6_preview_symmetries": d6_preview,
        "messages": messages,
    }


def _merge_epoch_result(row: dict[str, object], result: dict[str, object]) -> None:
    if isinstance(result.get("selfplay"), dict):
        row["selfplay"] = _selfplay_epoch_summary(result["selfplay"])
    if isinstance(result.get("training"), dict):
        row["training"] = _training_epoch_summary(result["training"])
    if isinstance(result.get("evaluation"), dict):
        row["evaluation"] = _evaluation_epoch_summary(result["evaluation"])
    if isinstance(result.get("checkpoint"), dict):
        checkpoint = result["checkpoint"]
        row["checkpoint"] = {
            "path": _run_relative_or_value(checkpoint.get("checkpoint_path")),
            "name": checkpoint.get("name"),
        }
    if isinstance(result.get("samples"), dict):
        samples = result["samples"]
        selection = samples.get("selection") if isinstance(samples.get("selection"), dict) else {}
        finalize = samples.get("finalize") if isinstance(samples.get("finalize"), dict) else {}
        row["samples"] = {
            "buffer_count": selection.get("sample_count") or finalize.get("buffer_count"),
            "window_size": selection.get("window_size"),
            "compressed_bytes": finalize.get("compressed_bytes"),
        }
    if isinstance(result.get("symmetries"), dict):
        metadata = result["symmetries"].get("metadata") if isinstance(result["symmetries"].get("metadata"), dict) else {}
        row["d6"] = {
            "mode": metadata.get("mode"),
            "group_size": metadata.get("d6_group_size"),
            "sample_count": metadata.get("sample_count"),
        }


def _selfplay_epoch_summary(payload: dict[str, object]) -> dict[str, object]:
    # Producer: dense_cnn/selfplay.py generate_selfplay_epoch (the summary dict).
    # Real keys: status, epoch, requested_games, games_started, completed_games,
    # truncated_games, games_finished, raw_samples, effective_samples,
    # searched_positions, mcts_simulations, search_visits, selfplay_npz_files,
    # record_path, elapsed_seconds, mcts_search_elapsed_seconds,
    # search_positions_per_second, positions_per_second, active_games,
    # mcts_virtual_batch_size, mcts_diagnostics, npz_writes.
    # Output key names are consumed by app.js and must stay unchanged; only the
    # source key each is populated FROM changes.
    completed_games = payload.get("completed_games")
    truncated_games = payload.get("truncated_games")

    # app.js reads selfplay.games; populate from games_finished, then fall back.
    games = payload.get("games_finished")
    if games is None:
        if completed_games is not None or truncated_games is not None:
            games = (completed_games or 0) + (truncated_games or 0)
        else:
            games = payload.get("games_started")

    # app.js reads selfplay.samples_added; producer emits effective_samples.
    samples_added = payload.get("effective_samples")
    if samples_added is None:
        samples_added = payload.get("raw_samples")

    # No producer key for per-searched-position sims; derive when both present.
    mcts_simulations = payload.get("mcts_simulations")
    searched_positions = payload.get("searched_positions")
    mcts_sims_per_searched_position: float | None = None
    sims = _optional_float(mcts_simulations)
    searched = _optional_float(searched_positions)
    if sims is not None and searched is not None and searched > 0.0:
        mcts_sims_per_searched_position = sims / searched

    return {
        "status": payload.get("status"),
        "games": games,
        "completed_games": completed_games,
        "truncated_games": truncated_games,
        "winner_counts": None,  # no producer key (separate worker handles display)
        "lengths": None,  # no producer key (separate worker handles display)
        "samples_added": samples_added,
        "searched_positions": searched_positions,
        "mcts_simulations": mcts_simulations,
        "search_positions_per_second": payload.get("search_positions_per_second"),
        "mcts_sims_per_searched_position": mcts_sims_per_searched_position,
        "elapsed_seconds": payload.get("elapsed_seconds"),
        # Game-length stats (None for producers that don't emit them, e.g. older
        # dense_cnn runs, so the frontend just omits them — additive, non-breaking).
        "game_length_mean": payload.get("game_length_mean"),
        "game_length_median": payload.get("game_length_median"),
        "game_length_max": payload.get("game_length_max"),
        "game_length_stdev": payload.get("game_length_stdev"),
        # Replay-buffer + per-head training-loss + calibration stats (nested object,
        # None for producers that don't emit it — e.g. dense_cnn runs — so the
        # frontend just omits the detail band). The dashboard bridge attaches this
        # to the published selfplay payload; without this passthrough the per-head
        # Losses group never reaches epochProgressDetail in app.js.
        "buffer": payload.get("buffer"),
    }


def _training_epoch_summary(payload: dict[str, object]) -> dict[str, object]:
    # Producer: dense_cnn/trainer.py DenseCNNTrainer.train_passes return dict.
    # Real keys: status, epoch, passes, generic_passes_requested, steps, samples,
    # batch_size, loss, validation, elapsed_seconds, samples_per_second,
    # train_state. loss_components/source_summary/policy_imitation are NOT in this
    # payload; the dense_cnn.policy_targets.epoch_*.json overlay populates them on
    # the row afterward (see _epoch_history), so they are None here.
    return {
        "status": payload.get("status"),
        "loss": payload.get("loss"),
        "loss_components": None,  # no producer key (overlaid from policy_targets file)
        "source_summary": None,  # no producer key (overlaid from policy_targets file)
        "policy_imitation": None,  # no producer key (overlaid from policy_targets file)
        "steps": payload.get("steps"),
        "samples": payload.get("samples"),
        "batch_size": payload.get("batch_size"),
        "samples_per_second": payload.get("samples_per_second"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
    }


def _evaluation_epoch_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "status": payload.get("status"),
        "games": payload.get("games"),
        "completed": payload.get("completed"),
        "wins": payload.get("wins"),
        "losses": payload.get("losses"),
        "mean_turns": payload.get("mean_turns"),
    }


def _coerce_epoch(value: object, path: str) -> int | None:
    try:
        if value is not None:
            return int(value)
    except (TypeError, ValueError):
        pass
    return _epoch_from_artifact_path(path)


def _source_fraction(source_counts: object, token: str) -> float | None:
    if not isinstance(source_counts, dict):
        return None
    total = 0
    matching = 0
    needle = token.lower()
    for key, value in source_counts.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        total += count
        if needle in str(key).lower():
            matching += count
    return (matching / total) if total > 0 else None


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _run_relative_or_value(value: object) -> object:
    if value is None:
        return None
    try:
        path = Path(str(value))
        if path.parts:
            parts = path.parts
            if "runs" in parts:
                index = parts.index("runs")
                return Path(*parts[index + 2 :]).as_posix()
    except Exception:
        pass
    return value


def _diagnostic_label(name: str) -> str:
    lowered = name.lower()
    if "evaluation" in lowered:
        return "evaluation"
    if "selfplay" in lowered:
        return "selfplay"
    if lowered.startswith("epoch_"):
        return "epoch"
    return Path(name).stem


def _history_source(path: str) -> str:
    parts = Path(path).parts
    if parts:
        return str(parts[0])
    return "history"


def _epoch_from_artifact_path(path: str) -> int | None:
    match = re.search(r"epoch[_-](\d+)", path)
    if not match:
        return None
    return int(match.group(1))


def _winner_label(winner: object | None) -> str:
    if winner == "player0":
        return "P0"
    if winner == "player1":
        return "P1"
    return "None"


def _players_by_role(players: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    by_role = {
        str(player.get("role")): player
        for player in players
        if player.get("role") in PLAYER_ROLES
    }
    for role in PLAYER_ROLES:
        by_role.setdefault(role, {"role": role, "kind": "unknown", "label": role, "player_id": role})
    return by_role


def _abort_payload(abort: object | None) -> object | None:
    if abort is None:
        return None
    return {
        "stage": getattr(abort, "stage", None),
        "exception_type": getattr(abort, "exception_type", None),
        "message": getattr(abort, "message", None),
    }


def _training_live_status(run_dir: Path) -> dict[str, object]:
    diagnostics = run_dir / "diagnostics"
    events = _stage_status_from_events(diagnostics / "events.jsonl")
    watchdog = _read_last_jsonl(diagnostics / "resource_watchdog.jsonl")
    calibration = _read_json_file(diagnostics / "dense_cnn.performance_calibration.json")
    selfplay_live = _read_json_file(diagnostics / "dense_cnn.selfplay.live.json")
    training_progress = _latest_training_progress(diagnostics)
    bootstrap_progress = _latest_bootstrap_training_progress(run_dir)
    trainer_command = ""
    if isinstance(watchdog, dict) and isinstance(watchdog.get("trainer"), dict):
        trainer_command = str(watchdog["trainer"].get("command_line") or "")
    status: dict[str, object] = {
        "stage": events.get("stage") or "unknown",
        "stage_status": events.get("status") or "unknown",
        "current_epoch": events.get("epoch"),
        "last_event": events.get("last_event"),
    }
    if "bootstrap_dense_cnn_classical.py" in trainer_command and isinstance(bootstrap_progress, dict):
        training_progress = bootstrap_progress
        status.update(
            {
                "stage": "classical_bootstrap_prefit",
                "stage_status": bootstrap_progress.get("status") or "running",
                "current_epoch": None,
                "bootstrap": {
                    "status": bootstrap_progress.get("status"),
                    "output_dir": bootstrap_progress.get("output_dir"),
                    "path": bootstrap_progress.get("path"),
                },
            }
        )
    if isinstance(watchdog, dict):
        status["watchdog"] = _watchdog_summary(watchdog)
    if isinstance(calibration, dict):
        status["calibration"] = _calibration_summary(calibration)
    if isinstance(selfplay_live, dict):
        status["selfplay_live"] = _selfplay_live_summary(selfplay_live)
    if isinstance(training_progress, dict):
        status["training_progress"] = _training_progress_summary(training_progress)
    return status


def _training_run_status(run_dir: Path, histories: list[dict[str, object]], live_status: dict[str, object]) -> dict[str, object]:
    lengths = [
        int(item.get("length") or item.get("actions") or 0)
        for item in histories
        if int(item.get("length") or item.get("actions") or 0) > 0
    ]
    latest_history = max(
        histories,
        key=lambda item: float(item.get("modified") or 0.0),
        default=None,
    )
    status = dict(live_status)
    p0_wins = sum(1 for item in histories if item.get("winner") == "player0")
    p1_wins = sum(1 for item in histories if item.get("winner") == "player1")
    status["history"] = {
        "games": len(histories),
        "complete": False,
        "scope": "recent",
        "completed": sum(1 for item in histories if item.get("status") == "completed"),
        "aborted": sum(1 for item in histories if item.get("status") != "completed"),
        "p0_wins": p0_wins,
        "p1_wins": p1_wins,
        "min_length": min(lengths) if lengths else None,
        "max_length": max(lengths) if lengths else None,
        "avg_length": (sum(lengths) / len(lengths)) if lengths else None,
        "latest_modified": latest_history.get("modified") if latest_history else None,
        "latest_path": latest_history.get("path") if latest_history else None,
    }
    latest_selfplay = max(
        (path for path in (run_dir / "selfplay").glob("*.hxr") if path.is_file()),
        key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)),
        default=None,
    )
    if latest_selfplay is not None:
        stat = _safe_stat(latest_selfplay)
        status["latest_selfplay_record"] = {
            "path": latest_selfplay.relative_to(run_dir).as_posix(),
            "bytes": stat.st_size if stat is not None else 0,
            "modified": stat.st_mtime if stat is not None else 0,
        }
    return status


def _live_history_diagnostic_summary(live_status: dict[str, object]) -> dict[str, object]:
    watchdog = live_status.get("watchdog") if isinstance(live_status.get("watchdog"), dict) else {}
    calibration = live_status.get("calibration") if isinstance(live_status.get("calibration"), dict) else {}
    summary: dict[str, object] = {
        "stage": live_status.get("stage") or "unknown",
        "epoch": live_status.get("current_epoch") or "--",
    }
    if watchdog:
        summary["watchdog"] = watchdog.get("status") or "unknown"
        summary["free_ram_gb"] = watchdog.get("free_ram_gb")
        summary["gpu_free_gb"] = watchdog.get("gpu_free_gb")
        summary["trainer_private_gb"] = watchdog.get("trainer_private_gb")
    if calibration:
        summary["selfplay_pos_s"] = calibration.get("selfplay_pos_s")
        summary["exact_128"] = calibration.get("exact_128")
    return summary


def _latest_training_progress(diagnostics_dir: Path) -> dict[str, object] | None:
    # NOTE: no current producer emits this file; see dense_cnn/selfplay.py (kept for forward-compat / manual drops).
    latest = max(
        diagnostics_dir.glob("dense_cnn.training_progress.epoch_*.json"),
        key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)),
        default=None,
    )
    if latest is None:
        return None
    payload = _read_json_file(latest)
    return payload if isinstance(payload, dict) else None


def _latest_bootstrap_training_progress(run_dir: Path) -> dict[str, object] | None:
    # NOTE: no current producer emits this file; see dense_cnn/selfplay.py (kept for forward-compat / manual drops).
    latest = max(
        (run_dir / "bootstrap").glob("*/diagnostics/dense_cnn.training_progress.epoch_*.json"),
        key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)),
        default=None,
    )
    if latest is None:
        return None
    payload = _read_json_file(latest)
    if not isinstance(payload, dict):
        return None
    payload = dict(payload)
    payload["path"] = latest.relative_to(run_dir).as_posix()
    payload["output_dir"] = latest.parents[1].relative_to(run_dir).as_posix()
    return payload


def _training_progress_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "epoch": payload.get("epoch"),
        "status": payload.get("status"),
        "progress": payload.get("progress"),
        "steps": payload.get("steps"),
        "total_steps": payload.get("total_steps"),
        "samples_seen": payload.get("samples_seen"),
        "samples": payload.get("samples"),
        "passes": payload.get("passes"),
        "loss": payload.get("loss"),
        "samples_per_second": payload.get("samples_per_second"),
        "path": payload.get("path"),
        "output_dir": payload.get("output_dir"),
    }


def _stage_status_from_events(path: Path) -> dict[str, object]:
    active_stage: str | None = None
    active_epoch: int | None = None
    last_event: dict[str, object] | None = None
    for event in _iter_jsonl(path):
        if not isinstance(event, dict):
            continue
        last_event = event
        name = str(event.get("event") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        stage = str(payload.get("stage") or "")
        if name == "stage_started" and stage:
            active_stage = stage
            active_epoch = _epoch_from_artifact_path(stage)
        elif name == "stage_finished" and stage == active_stage:
            active_stage = None
            active_epoch = None
    return {
        "stage": active_stage or _event_stage(last_event),
        "status": "running" if active_stage else _event_status(last_event),
        "epoch": active_epoch,
        "last_event": last_event,
    }


def _event_stage(event: dict[str, object] | None) -> str | None:
    payload = event.get("payload") if isinstance(event, dict) and isinstance(event.get("payload"), dict) else {}
    stage = payload.get("stage")
    return str(stage) if stage is not None else None


def _event_status(event: dict[str, object] | None) -> str | None:
    payload = event.get("payload") if isinstance(event, dict) and isinstance(event.get("payload"), dict) else {}
    status = payload.get("status")
    return str(status) if status is not None else None


def _watchdog_summary(payload: dict[str, object]) -> dict[str, object]:
    memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}
    trainer = payload.get("trainer") if isinstance(payload.get("trainer"), dict) else {}
    gpu = payload.get("gpu") if isinstance(payload.get("gpu"), dict) else {}
    return {
        "timestamp": payload.get("timestamp"),
        "status": payload.get("status"),
        "critical": payload.get("critical") or [],
        "free_ram_gb": memory.get("free_ram_gb"),
        "free_virtual_gb": memory.get("free_virtual_gb"),
        "trainer_private_gb": trainer.get("private_gb"),
        "trainer_working_set_gb": trainer.get("working_set_gb"),
        "gpu_free_gb": gpu.get("free_gb"),
        "gpu_used_gb": gpu.get("used_gb"),
        "gpu_utilization_percent": gpu.get("utilization_percent"),
    }


def _calibration_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "status": payload.get("status"),
        "device": payload.get("device"),
        "selfplay_pos_s": payload.get("measured_selfplay_positions_per_second"),
        "target_pos_s": payload.get("target_selfplay_positions_per_second"),
        "meets_target": payload.get("meets_target"),
        "exact_128": (
            payload.get("all_searches_exact") is True
            and int(payload.get("selected_mcts_visits") or 0) == 128
        ),
        "selected_inference_batch_size": payload.get("selected_inference_batch_size"),
        "selected_selfplay_batch_size": payload.get("selected_selfplay_batch_size"),
        "selected_training_batch_size": payload.get("selected_training_batch_size"),
        "selected_mcts_virtual_batch_size": payload.get("selected_mcts_virtual_batch_size"),
    }


def _selfplay_live_summary(payload: dict[str, object]) -> dict[str, object]:
    # Self-play writes this file every couple of seconds during an epoch and a
    # final "completed" snapshot at epoch end. "live" means the writer is still
    # running and the file is fresh, so the dashboard can trust the in-progress
    # search-pos/s; a stale "running" file (writer died) falls back to not-live.
    timestamp = payload.get("timestamp")
    age_seconds: float | None = None
    if isinstance(timestamp, (int, float)):
        age_seconds = max(0.0, wall_clock() - float(timestamp))
    status = str(payload.get("status") or "")
    is_live = status == "running" and age_seconds is not None and age_seconds <= 20.0
    return {
        "status": status or "unknown",
        "live": is_live,
        "age_seconds": age_seconds,
        "epoch": payload.get("epoch"),
        "search_pos_s": payload.get("search_positions_per_second"),
        "pos_s": payload.get("positions_per_second"),
        "searched_positions": payload.get("searched_positions"),
        "games_finished": payload.get("games_finished"),
        "requested_games": payload.get("requested_games"),
        "active_games": payload.get("active_games"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
    }


def _iter_jsonl(path: Path) -> list[object]:
    if not path.is_file():
        return []
    records: list[object] = []
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return records
    return records


def _read_last_jsonl(path: Path) -> object | None:
    records = _iter_jsonl(path)
    return records[-1] if records else None


def _safe_stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


def _read_json_file(path: Path, *, retries: int = 1) -> object | None:
    for attempt in range(retries + 1):
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if attempt < retries:
                continue
            return None
        except OSError:
            return None
    return None


def _artifact_summary(payload: object) -> object:
    if not isinstance(payload, dict):
        return None
    keys = (
        "status",
        "epoch",
        "positions_per_second",
        "search_positions_per_second",
        "end_to_end_positions_per_second",
        "mcts_search_elapsed_seconds",
        "samples_added",
        "samples_per_second",
        "measured_selfplay_positions_per_second",
        "selected_inference_batch_size",
        "selected_selfplay_batch_size",
        "selected_mcts_virtual_batch_size",
        "selected_mcts_visits",
        "searched_positions",
        "mcts_simulations",
        "mcts_sims_per_searched_position",
        "meets_target",
        "games",
        "completed",
        "wins",
        "losses",
        "mean_turns",
        "winner",
        "length",
    )
    return {key: payload[key] for key in keys if key in payload}


def _resolve_run_dir(name: str) -> Path | None:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    matches: list[Path] = []
    for root in _training_roots():
        resolved_root = root.resolve()
        path = (resolved_root / name).resolve()
        if resolved_root != path and resolved_root not in path.parents:
            continue
        if path.is_dir():
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)))


def _resolve_run_path(run_name: str, artifact_path: str) -> Path | None:
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None or not artifact_path or artifact_path.startswith(("/", "\\")):
        return None
    path = (run_dir / artifact_path).resolve()
    if run_dir.resolve() != path and run_dir.resolve() not in path.parents:
        return None
    return path


def make_handler(controller: ManualMatchController) -> type[HexoPlayHandler]:
    class BoundHexoPlayHandler(HexoPlayHandler):
        pass

    BoundHexoPlayHandler.controller = controller
    return BoundHexoPlayHandler


def run(host: str = "127.0.0.1", port: int = 8765, *, sealbot_path: str | Path | None = None) -> None:
    controller = ManualMatchController(sealbot_path=sealbot_path)
    server = ThreadingHTTPServer((host, port), make_handler(controller))
    print(f"Hexo frontend match: http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        controller.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the manual Hexo web match.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--sealbot-path", default=None, help="Path to an external SealBot checkout.")
    args = parser.parse_args(argv)
    run(host=args.host, port=args.port, sealbot_path=args.sealbot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
