"""Local dashboard routes need no token but retain strict browser-origin guards."""

import http.client
import json
import threading

import pytest

from enumeraite.web.server import MAX_BODY, DashboardServer

SCAN_ID = "a" * 32
API_ROUTES = [
    ("/api/state", "GET", None),
    ("/api/credentials", "POST", {"minimax_key": "example"}),
    ("/api/scans", "POST", {"target": "example.test"}),
    ("/api/resolve", "POST", {"target": "ny-01-iphone.example.com"}),
    (f"/api/scans/{SCAN_ID}", "GET", None),
    (f"/api/scans/{SCAN_ID}/pause", "POST", {}),
    (f"/api/scans/{SCAN_ID}/resume", "POST", {}),
    *[
        (f"/api/scans/{SCAN_ID}/export?format={kind}", "GET", None)
        for kind in ("json", "txt", "html")
    ],
]


class FakeManager:
    def __init__(self):
        self.calls = []

    def list_state(self):
        self.calls.append("list")
        return {"credentials": {"minimax": False}, "scans": []}

    def set_minimax_key(self, key):
        if not isinstance(key, str):
            raise ValueError("Key must be text")
        self.calls.append("key")
        return bool(key)

    def set_api_key(self, provider, key):
        if provider not in {"minimax", "openai", "anthropic", "jev"}:
            raise ValueError("Unsupported API provider")
        return self.set_minimax_key(key)

    def start(self, payload):
        self.calls.append(("start", payload))
        return SCAN_ID

    def resolve(self, payload):
        from enumeraite.web.manager import ScanManager

        self.calls.append(("resolve", payload))
        return ScanManager.resolve(payload)

    def pause(self, scan_id):
        self.calls.append(("pause", scan_id))

    def resume(self, scan_id, payload):
        self.calls.append(("resume", scan_id, payload))

    def snapshot(self, scan_id):
        self.calls.append(("snapshot", scan_id))
        if scan_id != SCAN_ID:
            raise KeyError(scan_id)
        return {"id": scan_id, "status": "paused"}

    def report(self, scan_id):
        self.calls.append(("report", scan_id))
        return {
            "target": "example.test",
            "findings": ["www.example.test"],
            "verified": 1,
            "known_verified": 1,
            "attempted": 2,
            "round": 0,
            "stop_reason": "paused; scan saved",
            "nodes": [],
            "patterns": [],
        }


@pytest.fixture
def dashboard():
    server = DashboardServer(FakeManager(), port=0)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(server, path="/api/state", method="GET", payload=None, headers=None, raw=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    fields = {}
    if payload is not None:
        raw = json.dumps(payload)
        fields["Content-Type"] = "application/json"
    fields.update(headers or {})
    connection.request(method, path, body=raw, headers=fields)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_server_binds_only_loopback_and_has_plain_local_url(dashboard):
    assert dashboard.server_address[0] == "127.0.0.1"
    assert dashboard.url == f"http://127.0.0.1:{dashboard.server_port}/"
    assert not hasattr(dashboard, "token")


@pytest.mark.parametrize(
    "path,method,payload",
    API_ROUTES,
)
def test_every_api_route_works_without_auth_from_local_client(dashboard, path, method, payload):
    status, headers, body = request(dashboard, path, method, payload)
    assert status == (201 if path == "/api/scans" else 200), body
    assert dashboard.manager.calls
    assert headers["Cache-Control"] == "no-store"
    assert "Access-Control-Allow-Origin" not in headers


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "attacker.test"},
        {"Host": "127.0.0.1.evil.test"},
        {"Origin": "https://attacker.test"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
@pytest.mark.parametrize(
    "path,method,payload",
    API_ROUTES + [(path, "GET", None) for path in ("/", "/app.js", "/style.css")],
)
def test_every_route_still_rejects_cross_origin_and_dns_rebinding(
    dashboard, headers, path, method, payload
):
    assert request(dashboard, path, method, payload, headers=headers)[0] == 403
    assert dashboard.manager.calls == []


def test_resolve_previews_seed_scope_without_starting(dashboard):
    status, _, body = request(
        dashboard, "/api/resolve", "POST", {"target": "ny-01-iphone.example.com"}
    )
    assert status == 200
    assert json.loads(body) == {
        "input": "ny-01-iphone.example.com",
        "target": "example.com",
        "seeds": ["ny-01-iphone.example.com"],
        "mode": "seeded",
        "passive_default": False,
    }
    assert len(dashboard.manager.calls) == 1
    assert dashboard.manager.calls[0][0] == "resolve"


@pytest.mark.parametrize(
    "payload",
    [
        {"target": "example.com", "seeds": ["foreign.com"]},
        {"target": "ny-01.example.com", "scope": "com"},
        {"target": "example.com", "options": {}},
    ],
)
def test_resolve_rejects_unsafe_or_unsupported_input(dashboard, payload):
    assert request(dashboard, "/api/resolve", "POST", payload)[0] == 400


def test_no_token_protocol_and_unexpected_query_parameters_still_rejected(dashboard):
    assert request(dashboard, "/api/state?token=obsolete")[0] == 400
    assert request(dashboard, headers={"Authorization": "Bearer obsolete"})[0] == 200


def test_static_assets_are_packaged_and_security_headers_present(dashboard):
    for path in ("/", "/app.js", "/style.css"):
        status, headers, body = request(dashboard, path)
        assert status == 200 and body
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert "unsafe-inline" not in headers["Content-Security-Policy"]
        assert "Access-Control-Allow-Origin" not in headers


@pytest.mark.parametrize(
    "path", ["/../pyproject.toml", "/%2e%2e/.env", "/assets/../../.env", "/favicon.ico"]
)
def test_no_arbitrary_file_server(dashboard, path):
    assert request(dashboard, path)[0] == 404


def test_key_write_returns_only_boolean_and_never_logs(dashboard, capsys):
    credential = "not-a-real-provider-credential"
    status, _, body = request(dashboard, "/api/credentials", "POST", {"minimax_key": credential})
    assert status == 200 and json.loads(body) == {"configured": True}
    assert credential.encode() not in body
    assert credential not in capsys.readouterr().err
    assert request(dashboard, "/api/credentials", "POST", {"minimax_key": ""})[0] == 200


@pytest.mark.parametrize("provider", ["minimax", "openai", "anthropic", "jev"])
def test_hosted_provider_credential_route(dashboard, provider):
    status, _, body = request(
        dashboard,
        "/api/credentials",
        "POST",
        {"provider": provider, "api_key": "fictional-credential"},
    )
    assert status == 200 and json.loads(body) == {"configured": True}
    assert b"fictional-credential" not in body


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "null",
        "{",
        '{"budget_usd":NaN}',
        '{"target":"a","target":"b"}',
        "[" * 2000 + "]" * 2000,
    ],
)
def test_malformed_payloads_fail_without_scan(dashboard, raw):
    assert (
        request(
            dashboard, "/api/scans", "POST", raw=raw, headers={"Content-Type": "application/json"}
        )[0]
        == 400
    )
    assert dashboard.manager.calls == []


@pytest.mark.parametrize(
    "headers,raw",
    [
        ({"Content-Type": "text/plain"}, "{}"),
        ({"Content-Length": str(MAX_BODY + 1), "Content-Type": "application/json"}, "{}"),
        ({"Transfer-Encoding": "chunked", "Content-Type": "application/json"}, "{}"),
        ({"Content-Type": "application/json"}, b"\xff"),
    ],
)
def test_bounded_json_body_only(dashboard, headers, raw):
    assert request(dashboard, "/api/scans", "POST", headers=headers, raw=raw)[0] == 400
    assert dashboard.manager.calls == []


def test_routes_dispatch_pause_resume_and_snapshots(dashboard):
    status, _, body = request(dashboard, "/api/scans", "POST", {"target": "example.test"})
    assert status == 201 and json.loads(body)["id"] == SCAN_ID
    assert request(dashboard, f"/api/scans/{SCAN_ID}/pause", "POST", {})[0] == 200
    assert (
        request(dashboard, f"/api/scans/{SCAN_ID}/resume", "POST", {"options": {"find": 2}})[0]
        == 200
    )
    assert request(dashboard, f"/api/scans/{SCAN_ID}")[0] == 200
    assert request(dashboard, "/api/scans/" + "b" * 32)[0] == 404
    assert request(dashboard, f"/api/scans/{SCAN_ID}/pause", "POST", {"path": "/tmp/no"})[0] == 400


@pytest.mark.parametrize("kind", ["txt", "json", "html"])
def test_downloads_are_local_attachments_without_token(dashboard, kind):
    status, headers, body = request(dashboard, f"/api/scans/{SCAN_ID}/export?format={kind}")
    assert status == 200
    assert headers["Content-Disposition"] == f'attachment; filename="enumeraite-{SCAN_ID}.{kind}"'
    assert b"example.test" in body


@pytest.mark.parametrize(
    "query", ["", "format=bad", "format=json&format=txt", "format=json&path=x"]
)
def test_export_parameters_are_restricted(dashboard, query):
    assert request(dashboard, f"/api/scans/{SCAN_ID}/export?{query}")[0] == 400


def test_preflight_does_not_enable_remote_access(dashboard):
    status, headers, _ = request(dashboard, method="OPTIONS")
    assert status == 405
    assert "Access-Control-Allow-Origin" not in headers


def test_exception_responses_do_not_expose_tracebacks(dashboard):
    def fail():
        raise Exception("private configuration value")

    dashboard.manager.list_state = fail
    status, _, body = request(dashboard)
    assert status == 500
    assert b"private configuration" not in body
