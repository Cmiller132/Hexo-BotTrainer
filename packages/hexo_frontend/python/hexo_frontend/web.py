"""Tiny stdlib web app for manually playing a Hexo match through the runner."""

from __future__ import annotations

import argparse
import json
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
from hexo_runner.records import GameResult
from hexo_runner.session import GameSpec

from .dashboard import dashboard_state


STATIC_ROOT = files("hexo_frontend").joinpath("static")
STATIC_TYPES = {
    "css": "text/css; charset=utf-8",
    "html": "text/html; charset=utf-8",
    "js": "text/javascript; charset=utf-8",
}
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
        self.wfile.write(encoded)

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
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _query_int(value: str | None) -> int | None:
    if value in {"", None}:
        return None
    return int(value)


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
