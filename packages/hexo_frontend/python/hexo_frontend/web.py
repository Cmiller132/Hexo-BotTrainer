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
from hexo_runner.player import DecisionResult, FinalSummary, PlayerIdentity, TransitionEvent, WorkerContext, GameContext
from hexo_runner.records import GameRecordV1, GameResult
from hexo_runner.session import GameSpec

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
        self._state: engine.HexoState | None = None
        self._python_state: engine.PythonHexoState | None = None
        self._pending_action: engine.Action | None = None
        self._version = 0
        self._records: list[GameRecordV1] = []
        self._result: GameResult | None = None
        self._error: BaseException | None = None
        self.reset()

    def reset(self) -> dict[str, object]:
        self.close()
        with self._condition:
            self._game_number += 1
            game_id = f"manual-{self._game_number}"
            self._cancelled = False
            self._state = None
            self._python_state = None
            self._pending_action = None
            self._version = 0
            self._records = []
            self._result = None
            self._error = None
            players = (_ManualPlayer(self, 0), _ManualPlayer(self, 1))
            spec = GameSpec(game_id=game_id)
            self._thread = Thread(target=self._run_match, args=(spec, players), daemon=True)
            self._thread.start()
            self._wait_for_state_locked()
            return dashboard_state(self._require_state_locked())

    def state(self) -> dict[str, object]:
        with self._condition:
            self._wait_for_state_locked()
            return dashboard_state(self._require_state_locked())

    def submit_move(self, q: int, r: int) -> dict[str, object]:
        with self._condition:
            self._wait_for_state_locked()
            state = self._state
            if state is None or self._result is not None:
                raise ValueError("No move is currently pending.")
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
            return dashboard_state(self._require_state_locked())

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
            self._state = state
            self._python_state = engine.to_python_state(state)
            self._version += 1
            self._condition.notify_all()

            while self._pending_action is None and not self._cancelled:
                self._condition.wait()
            if self._cancelled:
                raise RuntimeError("manual match reset")

            action = self._pending_action
            self._pending_action = None
            return DecisionResult(action=action, diagnostics={"manual_player": player_index})

    def write_game(self, record: GameRecordV1) -> object:
        with self._condition:
            self._records.append(record)
            self._condition.notify_all()
            return {"game_id": record.game_id, "actions": len(record.actions), "status": record.status}

    def observe_transition(self, transition: TransitionEvent) -> None:
        with self._condition:
            self._state = transition.state
            self._python_state = engine.to_python_state(transition.state)
            self._version += 1
            self._condition.notify_all()

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
        while self._python_state is None and self._error is None:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise RuntimeError("Timed out waiting for match state.")
            self._condition.wait(timeout=remaining)
        if self._error is not None:
            raise RuntimeError(str(self._error)) from self._error

    def _require_state_locked(self) -> engine.PythonHexoState:
        if self._python_state is None:
            raise RuntimeError("Match state is unavailable.")
        return self._python_state


class _ManualPlayer:
    def __init__(self, controller: ManualMatchController, player_index: int) -> None:
        self._controller = controller
        self._player_index = player_index
        self.identity = PlayerIdentity(player_id=f"manual-player-{player_index}", label=f"Player {player_index}")

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
