"""Tiny stdlib web app for manually playing a Hexo match through the runner."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from threading import Condition, RLock, Thread
from time import monotonic
from typing import Any, ClassVar
from urllib.parse import unquote, urlparse

import hexo_engine as engine
from hexo_runner.modes.match import run_match
from hexo_runner.player import DecisionRequest, DecisionResult, FinalSummary, PlayerIdentity, TransitionEvent
from hexo_runner.records import GameResult, PositionRecord
from hexo_runner.session import GameSpec, SessionContext

from .dashboard import dashboard_state


STATIC_ROOT = files("hexo_frontend").joinpath("static")
STATIC_TYPES = {
    "css": "text/css; charset=utf-8",
    "html": "text/html; charset=utf-8",
    "js": "text/javascript; charset=utf-8",
}


class ManualMatchController:
    """Frontend-owned bridge between HTTP clicks and generic runner players."""

    def __init__(self) -> None:
        self._condition = Condition(RLock())
        self._thread: Thread | None = None
        self._game_number = 0
        self._cancelled = False
        self._request: DecisionRequest | None = None
        self._pending_action: engine.Action | None = None
        self._raw_state: dict[str, object] | None = None
        self._version = 0
        self._entries: list[PositionRecord] = []
        self._result: GameResult | None = None
        self._error: BaseException | None = None
        self.reset()

    def reset(self) -> dict[str, object]:
        self.close()
        with self._condition:
            self._game_number += 1
            game_id = f"manual-{self._game_number}"
            self._cancelled = False
            self._request = None
            self._pending_action = None
            self._raw_state = None
            self._version = 0
            self._entries = []
            self._result = None
            self._error = None
            players = (_ManualPlayer(self, 0), _ManualPlayer(self, 1))
            spec = GameSpec(game_id=game_id)
            self._thread = Thread(target=self._run_match, args=(spec, players), daemon=True)
            self._thread.start()
            self._wait_for_state_locked()
            return dashboard_state(self._require_raw_state_locked())

    def state(self) -> dict[str, object]:
        with self._condition:
            self._wait_for_state_locked()
            return dashboard_state(self._require_raw_state_locked())

    def submit_move(self, q: int, r: int) -> dict[str, object]:
        with self._condition:
            self._wait_for_state_locked()
            request = self._request
            if request is None or self._result is not None:
                raise ValueError("No move is currently pending.")
            action = engine.PlacementAction(engine.AxialCoord(q=q, r=r))
            if action not in request.legal_actions:
                raise ValueError(f"{q},{r} is not legal.")

            start_version = self._version
            self._pending_action = action
            self._condition.notify_all()
            while self._version == start_version and self._error is None and self._result is None:
                self._condition.wait(timeout=0.25)
            if self._error is not None:
                raise RuntimeError(str(self._error)) from self._error
            return dashboard_state(self._require_raw_state_locked())

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

    def decide(self, player_index: int, request: DecisionRequest) -> DecisionResult:
        with self._condition:
            if self._cancelled:
                raise RuntimeError("manual match reset")
            self._request = request
            self._raw_state = _raw_from_view(request.state)
            self._version += 1
            self._condition.notify_all()

            while self._pending_action is None and not self._cancelled:
                self._condition.wait()
            if self._cancelled:
                raise RuntimeError("manual match reset")

            action = self._pending_action
            self._pending_action = None
            return DecisionResult(action=action, diagnostics={"manual_player": player_index})

    def write_entry(self, entry: PositionRecord) -> None:
        with self._condition:
            self._entries.append(entry)
            self._raw_state = _raw_from_snapshot(entry.after_snapshot)
            self._version += 1
            self._condition.notify_all()

    def close_game(self, game_id: str, terminal: object | None = None) -> object:
        with self._condition:
            self._condition.notify_all()
            return {"game_id": game_id, "entries": len(self._entries), "terminal": terminal is not None}

    def _run_match(self, spec: GameSpec, players: tuple["_ManualPlayer", "_ManualPlayer"]) -> None:
        try:
            result = run_match(spec, players, self)
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
            return
        with self._condition:
            self._result = result
            self._condition.notify_all()

    def _wait_for_state_locked(self, timeout: float = 5.0) -> None:
        deadline = monotonic() + timeout
        while self._raw_state is None and self._error is None:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise RuntimeError("Timed out waiting for match state.")
            self._condition.wait(timeout=remaining)
        if self._error is not None:
            raise RuntimeError(str(self._error)) from self._error

    def _require_raw_state_locked(self) -> dict[str, object]:
        if self._raw_state is None:
            raise RuntimeError("Match state is unavailable.")
        return self._raw_state


class _ManualPlayer:
    def __init__(self, controller: ManualMatchController, player_index: int) -> None:
        self._controller = controller
        self._player_index = player_index
        self.identity = PlayerIdentity(player_id=f"manual-player-{player_index}", label=f"Player {player_index}")

    def initialize(self, session_context: SessionContext) -> None:
        return

    def decide(self, request: DecisionRequest) -> DecisionResult:
        return self._controller.decide(self._player_index, request)

    def observe_transition(self, transition: TransitionEvent) -> None:
        return

    def close(self, final_summary: FinalSummary) -> None:
        return


class HexoPlayHandler(BaseHTTPRequestHandler):
    server_version = "hexo-frontend-play/0.1"
    controller: ClassVar[ManualMatchController]

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(self.controller.state())
        elif path == "/" or path == "/index.html":
            self._send_static("index.html")
        elif path.startswith("/static/"):
            self._send_static(unquote(path.removeprefix("/static/")))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/new":
                self._send_json(self.controller.reset())
            elif path == "/api/move":
                body = self._read_json()
                self._send_json(self.controller.submit_move(int(body["q"]), int(body["r"])))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json({"error": str(exc), "state": self.controller.state()}, HTTPStatus.BAD_REQUEST)

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


def make_handler(controller: ManualMatchController) -> type[HexoPlayHandler]:
    class BoundHexoPlayHandler(HexoPlayHandler):
        pass

    BoundHexoPlayHandler.controller = controller
    return BoundHexoPlayHandler


def _raw_from_view(view: Any) -> dict[str, object]:
    return {
        "engine_state": view.game_state,
        "legal_actions": [_action_payload(action) for action in view.legal_actions],
        "legal_count": len(view.legal_actions),
        "terminal": _terminal_payload(view.terminal),
        "tactics": view.tactics,
        "snapshot": _snapshot_payload(view.snapshot),
    }


def _raw_from_snapshot(snapshot: engine.EngineSnapshot) -> dict[str, object]:
    state_ref = engine.load_snapshot(snapshot)
    legal = tuple(engine.legal_actions(state_ref))
    return {
        "engine_state": engine.game_state(state_ref),
        "legal_actions": [_action_payload(action) for action in legal],
        "legal_count": len(legal),
        "terminal": _terminal_payload(engine.terminal(state_ref)),
        "tactics": engine.tactics(state_ref),
        "snapshot": _snapshot_payload(snapshot),
    }


def _action_payload(action: object) -> dict[str, int]:
    if isinstance(action, engine.PlacementAction):
        return {"q": action.coord.q, "r": action.coord.r}
    raise ValueError(f"Unsupported action type: {type(action).__name__}")


def _snapshot_payload(snapshot: engine.EngineSnapshot) -> dict[str, object]:
    return {"version": snapshot.version, "placements": snapshot.payload.get("placements", [])}


def _terminal_payload(outcome: object) -> dict[str, object] | None:
    if outcome is None:
        return None
    winner = getattr(outcome, "winner", None)
    return {
        "winner": _raw_player(winner),
        "reason": getattr(outcome, "reason", None),
        "metadata": dict(getattr(outcome, "metadata", {}) or {}),
    }


def _raw_player(player: object) -> str | None:
    value = getattr(player, "value", player)
    if value == "player0":
        return "Player0"
    if value == "player1":
        return "Player1"
    return None


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    controller = ManualMatchController()
    server = ThreadingHTTPServer((host, port), make_handler(controller))
    print(f"Hexo frontend manual match: http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        controller.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the manual Hexo web match.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args(argv)
    run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
