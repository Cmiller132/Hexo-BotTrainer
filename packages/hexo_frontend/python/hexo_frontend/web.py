"""Tiny stdlib web app for manually playing a Hexo match."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from threading import Lock
from typing import Any, ClassVar
from urllib.parse import unquote, urlparse

from hexo_engine import IllegalActionError
from hexo_runner.modes.match import create_match

from .dashboard import dashboard_state


STATIC_ROOT = files("hexo_frontend").joinpath("static")
STATIC_TYPES = {
    "css": "text/css; charset=utf-8",
    "html": "text/html; charset=utf-8",
    "js": "text/javascript; charset=utf-8",
}


class HexoPlayHandler(BaseHTTPRequestHandler):
    server_version = "hexo-frontend-play/0.1"
    match: ClassVar[Any]
    match_lock: ClassVar[Lock]

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            with self.match_lock:
                self._send_json(dashboard_state(self.match.view()))
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
                with self.match_lock:
                    self._send_json(dashboard_state(self.match.reset()))
            elif path == "/api/undo":
                with self.match_lock:
                    self._send_json(dashboard_state(self.match.undo()))
            elif path == "/api/move":
                body = self._read_json()
                with self.match_lock:
                    self._send_json(dashboard_state(self.match.play(int(body["q"]), int(body["r"]))))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (IllegalActionError, KeyError, TypeError, ValueError) as exc:
            with self.match_lock:
                state = dashboard_state(self.match.view())
            self._send_json({"error": str(exc), "state": state}, HTTPStatus.BAD_REQUEST)

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


def make_handler(match: Any) -> type[HexoPlayHandler]:
    class BoundHexoPlayHandler(HexoPlayHandler):
        pass

    BoundHexoPlayHandler.match = match
    BoundHexoPlayHandler.match_lock = Lock()
    return BoundHexoPlayHandler


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(create_match()))
    print(f"Hexo frontend manual match: http://{host}:{port}")
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the manual Hexo web match.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args(argv)
    run(host=args.host, port=args.port)
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
