"""Loopback-only, read-only local application with a bounded synthetic endpoint."""

from __future__ import annotations

import json
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import parse_qs, urlsplit

from courtgraph.app.observations import Observations
from courtgraph.app.sandbox import Sandbox

ASSETS = Path(__file__).with_name("static")


def make_server(
    port: int, observations: Observations | None, sandbox: Sandbox
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server: ThreadingHTTPServer

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _trusted(self) -> bool:
            hosts = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if (
                host not in hosts
                or (origin is not None and origin != f"http://{host}")
                or self.headers.get("Sec-Fetch-Site", "none")
                not in {"none", "same-origin"}
            ):
                self._json(
                    403, {"error": "This app only accepts requests from its local page"}
                )
                return False
            return True

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "img-src 'self' data:; connect-src 'self'; frame-ancestors "
                    "'none'; base-uri 'none'; form-action 'none'"
                ),
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: dict[str, Any]) -> None:
            self._send(
                status,
                json.dumps(data, allow_nan=False).encode(),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:
            if not self._trusted():
                return
            url = urlsplit(self.path)
            assets = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if url.path in assets:
                filename, mime = assets[url.path]
                self._send(
                    200, (ASSETS / filename).read_bytes(), mime + "; charset=utf-8"
                )
            elif url.path == "/api/state":
                self._json(
                    200,
                    {
                        "real": observations.overview()
                        if observations
                        else {"loaded": False},
                        "synthetic": sandbox.catalog(),
                    },
                )
            elif url.path == "/api/observations":
                if observations is None:
                    self._json(
                        200,
                        {
                            "lineups": [],
                            "stints": 0,
                            "possessions": 0,
                            "points": 0,
                            "games": 0,
                        },
                    )
                    return
                try:
                    query = parse_qs(url.query, max_num_fields=8)
                    if set(query) - {"game", "team", "player", "minimum"}:
                        raise ValueError("Unknown filter")
                    self._json(
                        200,
                        observations.query(
                            game=query.get("game", [""])[0],
                            team=query.get("team", [""])[0],
                            player=query.get("player", [""])[0],
                            minimum=int(query.get("minimum", ["1"])[0]),
                        ),
                    )
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
            elif url.path == "/api/player-pool":
                if observations is None:
                    self._json(200, {"players": [], "team": "", "source": ""})
                    return
                try:
                    query = parse_qs(url.query, max_num_fields=2)
                    if set(query) - {"team"}:
                        raise ValueError("Unknown filter")
                    self._json(
                        200, observations.player_pool(query.get("team", [""])[0])
                    )
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
            else:
                self._json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if not self._trusted():
                return
            if self.path not in (
                "/api/compare",
                "/api/predict-real",
                "/api/compare-real",
            ):
                self._json(404, {"error": "Not found"})
                return
            if (
                self.headers.get("Content-Type") != "application/json"
                or self.headers.get("X-CourtGraph-Request") != "local"
            ):
                self._json(415, {"error": "Use the local app's JSON request"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16_384:
                    raise ValueError("Request body must be 1-16384 bytes")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object")
                if self.path == "/api/compare":
                    self._json(200, sandbox.compare(payload))
                elif observations is None:
                    raise ValueError("No real ingest directory is loaded")
                elif self.path == "/api/predict-real":
                    self._json(200, observations.predict(payload))
                else:
                    self._json(200, observations.compare(payload))
            except (ValueError, UnicodeError) as exc:
                self._json(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(
    port: int, ingest_dir: Path | None, names_path: Path | None, stream: TextIO
) -> int:
    if not 0 <= port <= 65535:
        print("app: port must be between 0 and 65535", file=stream)
        return 2
    if names_path is not None and ingest_dir is None:
        print("app: --names requires --ingest-dir", file=stream)
        return 2
    try:
        observations = Observations(ingest_dir, names_path) if ingest_dir else None
        print(
            "Preparing a small synthetic model locally; no NBA model is trained...",
            file=stream,
            flush=True,
        )
        sandbox = Sandbox()
        server = make_server(port, observations, sandbox)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"app: could not start: {exc}", file=stream)
        return 2
    with server:
        print(
            f"CourtGraph: http://127.0.0.1:{server.server_port} "
            "(local only; Ctrl+C to stop)",
            file=stream,
            flush=True,
        )
        with suppress(KeyboardInterrupt):
            server.serve_forever()
    return 0
