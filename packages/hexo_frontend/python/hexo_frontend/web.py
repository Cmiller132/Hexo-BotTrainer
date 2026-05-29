"""Tiny stdlib web app for manually playing a Hexo match through the runner."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Condition, RLock, Thread
from time import monotonic, perf_counter
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
BotFactory = Callable[[str, float], object]
PLAYER_ROLES = ("player0", "player1")
MANUAL_KIND = "manual"
SEALBOT_PREFIX = "sealbot-"


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
                self._send_json(_training_runs())
            elif path == "/api/training/run":
                query = parse_qs(parsed.query)
                self._send_json(_training_run(str(query.get("name", [""])[0])))
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

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def _error_payload(self, message: str) -> dict[str, object]:
        try:
            return {"error": message, "state": self.controller.state()}
        except Exception:
            return {"error": message}

    def _send_static(self, name: str) -> None:
        if not name or "/" in name or name.startswith("."):
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        resource = STATIC_ROOT.joinpath(name)
        try:
            encoded = resource.read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        extension = name.rsplit(".", 1)[-1]
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", STATIC_TYPES.get(extension, "application/octet-stream"))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def _send_training_file(self, run_name: str, artifact_path: str) -> None:
        path = _resolve_run_path(run_name, artifact_path)
        if path is None or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        encoded = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ARTIFACT_TYPES.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return


def _query_int(value: str | None) -> int | None:
    if value in {"", None}:
        return None
    return int(value)


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
        for path in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True):
            if not path.is_dir():
                continue
            diagnostics = path / "diagnostics"
            selfplay = path / "selfplay"
            if not diagnostics.exists() and not selfplay.exists():
                continue
            current = {
                "name": path.name,
                "path": str(path),
                "diagnostics": str(diagnostics),
                "selfplay": str(selfplay),
                "modified": path.stat().st_mtime,
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
    artifacts = []
    diagnostics_by_epoch = _diagnostics_by_epoch(run_dir)
    live_status = _training_live_status(run_dir)
    histories_by_path = _training_histories(run_dir, diagnostics_by_epoch, live_status)
    epoch_history = _epoch_history(run_dir)
    evaluation_history = _evaluation_history(run_dir)
    for path in sorted(_iter_training_files(run_dir), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ARTIFACT_SUFFIXES:
            continue
        rel = path.relative_to(run_dir).as_posix()
        history_count = len(histories_by_path.get(rel, ()))
        artifact: dict[str, object] = {
            "path": rel,
            "name": path.name,
            "bytes": path.stat().st_size,
            "modified": path.stat().st_mtime,
            "kind": path.suffix.lower().lstrip(".") or "file",
            "loadable_history": history_count > 0,
            "history_count": history_count,
        }
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                artifact["summary"] = _artifact_summary(payload)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                artifact["summary"] = None
        artifacts.append(artifact)
    histories = [
        item
        for path_histories in histories_by_path.values()
        for item in path_histories
    ]
    histories.sort(
        key=lambda item: (
            float(item.get("modified") or 0.0),
            int(item.get("epoch") or 0),
            str(item.get("path") or ""),
            int(item.get("record_index") or 0),
        ),
        reverse=True,
    )
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "artifacts": artifacts,
        "histories": histories,
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
    if path.stat().st_size <= 0:
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
            "version": int(path.stat().st_mtime_ns % 9_000_000_000_000_000),
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
        if not path.is_file() or path.stat().st_size <= 0:
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
                    "modified": path.stat().st_mtime,
                    "bytes": path.stat().st_size,
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
                "modified": path.stat().st_mtime,
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
            row["checkpoint"] = {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "modified": path.stat().st_mtime,
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
    return {
        "status": payload.get("status"),
        "games": payload.get("games"),
        "completed_games": payload.get("completed_games"),
        "truncated_games": payload.get("truncated_games"),
        "winner_counts": payload.get("winner_counts") if isinstance(payload.get("winner_counts"), dict) else None,
        "lengths": payload.get("lengths") if isinstance(payload.get("lengths"), dict) else None,
        "samples_added": payload.get("samples_added"),
        "searched_positions": payload.get("searched_positions"),
        "mcts_simulations": payload.get("mcts_simulations"),
        "search_positions_per_second": payload.get("search_positions_per_second"),
        "mcts_sims_per_searched_position": payload.get("mcts_sims_per_searched_position"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
    }


def _training_epoch_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "status": payload.get("status"),
        "loss": payload.get("loss"),
        "loss_components": payload.get("loss_components") if isinstance(payload.get("loss_components"), dict) else None,
        "source_summary": payload.get("source_summary") if isinstance(payload.get("source_summary"), dict) else None,
        "policy_imitation": payload.get("policy_imitation") if isinstance(payload.get("policy_imitation"), dict) else None,
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
        key=lambda item: item.stat().st_mtime,
        default=None,
    )
    if latest_selfplay is not None:
        status["latest_selfplay_record"] = {
            "path": latest_selfplay.relative_to(run_dir).as_posix(),
            "bytes": latest_selfplay.stat().st_size,
            "modified": latest_selfplay.stat().st_mtime,
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
    latest = max(
        diagnostics_dir.glob("dense_cnn.training_progress.epoch_*.json"),
        key=lambda item: item.stat().st_mtime,
        default=None,
    )
    if latest is None:
        return None
    payload = _read_json_file(latest)
    return payload if isinstance(payload, dict) else None


def _latest_bootstrap_training_progress(run_dir: Path) -> dict[str, object] | None:
    latest = max(
        (run_dir / "bootstrap").glob("*/diagnostics/dense_cnn.training_progress.epoch_*.json"),
        key=lambda item: item.stat().st_mtime,
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


def _read_json_file(path: Path) -> object | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
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
    return max(matches, key=lambda item: item.stat().st_mtime)


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
