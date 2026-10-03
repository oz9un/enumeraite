"""Small local-only HTTP boundary; not a public deployment server."""

from __future__ import annotations

import json
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ..cli.scan_tree import render_tree
from ..core.text_safety import safe_exception_text

MAX_BODY = 65_536
SCAN_ROUTE = re.compile(r"/api/scans/([a-f0-9]{32})(?:/(pause|resume|export))?")
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; font-src 'self'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'; object-src 'none'"
)


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON fields are not allowed")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError("JSON numbers must be finite")


class DashboardServer(ThreadingHTTPServer):
    """Only binds IPv4 loopback and accepts a bounded number of short requests."""

    daemon_threads = True
    allow_reuse_address = True
    block_on_close = False

    def __init__(self, manager: Any, port: int = 8765):
        self.manager = manager
        self._slots = threading.BoundedSemaphore(32)
        super().__init__(("127.0.0.1", port), DashboardHandler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/"

    def process_request(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: Any
    ) -> None:
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: Any
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def handle_error(self, request: Any, client_address: Any) -> None:
        # Never dump request headers, provider keys, or tracebacks to logs.
        pass


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer
    server_version = "EnumeraiteLocal"
    sys_version = ""
    protocol_version = "HTTP/1.0"

    def setup(self) -> None:
        self.request.settimeout(5)
        super().setup()

    def log_message(self, format: str, *args: Any) -> None:
        # URLs/headers/bodies can contain credentials; no request logging.
        pass

    def _send(
        self,
        code: int,
        body: bytes,
        content_type: str = "application/json; charset=utf-8",
        filename: str | None = None,
    ) -> None:
        self.close_connection = True
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Connection", "close")
        if filename is not None:
            # Filename comes only from an opaque validated scan id and fixed suffix.
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return

    def _json(self, code: int, value: dict) -> None:
        self._send(code, json.dumps(value, ensure_ascii=True, allow_nan=False).encode())

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        # The base parser may include request-controlled data in its error message.
        self._json(code, {"error": "Request not accepted"})

    def _local_request(self) -> bool:
        hosts = self.headers.get_all("Host", [])
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if len(hosts) != 1 or hosts[0] not in allowed:
            self._json(403, {"error": "Only the local dashboard address is allowed"})
            return False
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1 or (origins and origins[0] != f"http://{hosts[0]}"):
            self._json(403, {"error": "Cross-origin requests are not allowed"})
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._json(403, {"error": "Cross-site requests are not allowed"})
            return False
        return True

    def _body(self) -> dict:
        if self.headers.get("Transfer-Encoding") is not None:
            raise ValueError("Streaming request bodies are not supported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,6}", lengths[0]):
            raise ValueError("A bounded Content-Length is required")
        length = int(lengths[0])
        if length > MAX_BODY:
            raise ValueError("Request exceeds the 64 KB limit")
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Use application/json requests")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError("Incomplete request body")
        try:
            payload = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_nonfinite
            )
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("Invalid JSON request") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request must be a JSON object")
        return payload

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_OPTIONS(self) -> None:
        self._json(405, {"error": "Cross-origin access is not supported"})

    def _dispatch(self, method: str) -> None:
        if not self._local_request():
            return
        try:
            location = urlsplit(self.path)
            if location.scheme or location.netloc or location.fragment:
                raise ValueError("Invalid request URL")
            path = location.path
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
            }
            if method == "GET" and path in assets and not location.query:
                name, mime = assets[path]
                content = files("enumeraite.web").joinpath("assets").joinpath(name).read_bytes()
                self._send(200, content, mime)
                return
            if not path.startswith("/api/"):
                self._json(404, {"error": "Not found"})
                return
            manager = self.server.manager
            match = SCAN_ROUTE.fullmatch(path)
            if location.query and not (method == "GET" and match and match[2] == "export"):
                raise ValueError("Unexpected query parameters")
            if method == "GET" and path == "/api/state":
                self._json(200, manager.list_state())
            elif method == "POST" and path == "/api/resolve":
                self._json(200, manager.resolve(self._body()))
            elif method == "POST" and path == "/api/credentials":
                payload = self._body()
                if set(payload) == {"minimax_key"}:
                    configured = manager.set_minimax_key(payload["minimax_key"])
                elif set(payload) == {"provider", "api_key"}:
                    configured = manager.set_api_key(payload["provider"], payload["api_key"])
                else:
                    raise ValueError("Supply provider and api_key")
                self._json(200, {"configured": configured})
            elif method == "POST" and path == "/api/scans":
                self._json(201, {"id": manager.start(self._body())})
            elif match:
                scan_id, action = match.groups()
                if method == "GET" and action is None:
                    self._json(200, manager.snapshot(scan_id))
                elif method == "GET" and action == "export":
                    self._export(scan_id, location.query)
                elif method == "POST" and action in {"pause", "resume"}:
                    payload = self._body()
                    if action == "pause":
                        if payload:
                            raise ValueError("Pause takes no options")
                        manager.pause(scan_id)
                    else:
                        manager.resume(scan_id, payload)
                    self._json(200, {"id": scan_id})
                else:
                    self._json(405, {"error": "Method not supported"})
            else:
                self._json(404, {"error": "Not found"})
        except KeyError:
            self._json(404, {"error": "Scan not found"})
        except (ValueError, TypeError) as exc:
            self._json(400, {"error": safe_exception_text(exc)})
        except RuntimeError as exc:
            self._json(409, {"error": safe_exception_text(exc)})
        except (TimeoutError, OSError):
            self._json(503, {"error": "Local request could not complete; try again"})
        except Exception:
            self._json(500, {"error": "Dashboard request failed; scan checkpoints are retained"})

    def _export(self, scan_id: str, query: str) -> None:
        params = parse_qs(query, keep_blank_values=True)
        if set(params) != {"format"} or len(params["format"]) != 1:
            raise ValueError("Select one export format: txt, json, or html")
        kind = params["format"][0]
        if kind not in {"txt", "json", "html"}:
            raise ValueError("Select one export format: txt, json, or html")
        report = self.server.manager.report(scan_id)
        if kind == "json":
            content = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
            mime = "application/json; charset=utf-8"
        elif kind == "txt":
            content = "".join(value + "\n" for value in report["findings"])
            mime = "text/plain; charset=utf-8"
        else:
            content = render_tree(report)
            mime = "text/html; charset=utf-8"
        self._send(200, content.encode("utf-8"), mime, f"enumeraite-{scan_id}.{kind}")
