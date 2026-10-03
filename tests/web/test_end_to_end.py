"""Real HTTP → manager → discovery-session integration, with offline providers."""

import http.client
import json
import threading
import time
from contextlib import contextmanager

import pytest

from enumeraite.core.discovery_session import validate_checkpoint
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.passive import PassiveResult
from enumeraite.core.scan_dns import ScanDNSResult
from enumeraite.providers.llm_provider import ChatResult, LLMProvider
from enumeraite.web.manager import ENDPOINTS, ScanManager
from enumeraite.web.server import DashboardServer


class BrowserAPI:
    def __init__(self, server):
        self.server = server
        self.responses = []

    def request(self, path, *, method="GET", payload=None):
        client = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        headers = {}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload)
        try:
            client.request(method, path, body=body, headers=headers)
            response = client.getresponse()
            result = response.status, dict(response.getheaders()), response.read()
            self.responses.append(result[2])
            return result
        finally:
            client.close()

    def json(self, path, *, method="GET", payload=None, expected=200):
        status, _, body = self.request(path, method=method, payload=payload)
        assert status == expected, body
        return json.loads(body)

    def wait_finished(self, scan_id):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            snapshot = self.json(f"/api/scans/{scan_id}")
            if snapshot["status"] not in {"running", "pausing"}:
                return snapshot
            time.sleep(0.005)
        pytest.fail("Offline dashboard scan did not finish")


@contextmanager
def running_dashboard(storage_dir, **injected):
    # DiscoverySession and every HTTP route remain production implementations.
    manager = ScanManager(storage_dir, **injected)
    server = DashboardServer(manager, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield BrowserAPI(server)
    finally:
        manager.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        for row in list(manager._records.values()):
            if row.thread is not None:
                row.thread.join(timeout=2)


def passive_fixture(target, **kwargs):
    names = [f"ny-01-iphone.{target}"]
    return PassiveResult(names, {"fixture-certificate-log": names}, [])


def test_http_scan_collects_known_names_recurses_and_exports_real_tree(tmp_path):
    calls = []
    known = "ny-01-iphone.example.test"
    inferred = "ny-02-iphone.example.test"

    class OfflineDNS:
        def __init__(self, target, **kwargs):
            pass

        def validate(self, values, **kwargs):
            calls.extend(values)
            return [
                DNSResult(
                    value,
                    value in {known, inferred},
                    ["192.0.2.1"] if value in {known, inferred} else [],
                    0,
                )
                for value in values
            ]

    with running_dashboard(tmp_path, dns_factory=OfflineDNS, passive_fetch=passive_fixture) as api:
        scan_id = api.json(
            "/api/scans",
            method="POST",
            expected=201,
            payload={
                "target": "example.test",
                "options": {"max_candidates": 40, "max_per_host": 1, "max_number": 2, "find": 2},
            },
        )["id"]
        snapshot = api.wait_finished(scan_id)
        assert snapshot["status"] == "completed", snapshot["error"]
        assert snapshot["summary"]["findings"] == [known, inferred]
        assert snapshot["summary"]["known_verified"] == 1
        assert snapshot["summary"]["stop_reason"] == "verified target reached"
        assert calls[0] == known
        assert len(calls) == len(set(calls))
        nodes = {node["value"]: node for node in snapshot["nodes"]}
        assert nodes[known]["source"] == "passive"
        assert nodes[inferred]["parent"] == known
        assert nodes[inferred]["source"] == "pattern"
        assert any("number" in pattern["pattern"] for pattern in snapshot["patterns"])
        assert not snapshot["active"]
        history = api.json("/api/state")
        assert history["scans"][0]["id"] == scan_id
        assert history["scans"][0]["summary"]["verified"] == 2
        report = api.json(f"/api/scans/{scan_id}/export?format=json")
        assert report["nodes"] == snapshot["nodes"]
        assert report["findings"] == [known, inferred]
        status, headers, text = api.request(f"/api/scans/{scan_id}/export?format=txt")
        assert status == 200
        assert text.decode().splitlines() == [known, inferred]
        assert headers["Content-Disposition"].endswith('.txt"')
        status, _, html = api.request(f"/api/scans/{scan_id}/export?format=html")
        assert status == 200
        assert b"DISCOVERY TREE" in html
        assert known.encode() in html and inferred.encode() in html
        saved = json.loads((tmp_path / f"{scan_id}.json").read_text())
        validate_checkpoint(saved, "example.test")
        assert saved["_ui"]["status"] == "completed"


def test_http_pause_shows_live_worker_then_resumes_only_unfinished_work(tmp_path):
    entered = threading.Event()
    blocked_once = False
    calls, passive_calls = [], []
    known, pending = "api.example.test", "cdn.example.test"

    def passive(target, **kwargs):
        passive_calls.append(target)
        return PassiveResult([known, pending], {"fixture": [known, pending]}, [])

    class HoldingDNS:
        def __init__(self, target, *, cancel_event, event, **kwargs):
            self.cancel = cancel_event
            self.event = event

        def validate(self, values, **kwargs):
            nonlocal blocked_once
            calls.extend(values)
            if values == [pending] and not blocked_once:
                blocked_once = True
                self.event(
                    "dns_start",
                    {
                        "query_id": 42,
                        "worker_id": "fixture-dns-1",
                        "hostname": pending,
                        "candidate": pending,
                        "query_kind": "A",
                        "probe": False,
                    },
                )
                entered.set()
                assert self.cancel.wait(3), "HTTP pause did not reach the scanner"
                self.event("dns_end", {"query_id": 42})
                return [ScanDNSResult(pending, False, [], 0, cancelled=True)]
            return [DNSResult(value, True, ["192.0.2.2"], 0) for value in values]

    with running_dashboard(tmp_path, dns_factory=HoldingDNS, passive_fetch=passive) as api:
        scan_id = api.json(
            "/api/scans",
            method="POST",
            expected=201,
            payload={
                "target": "example.test",
                "options": {"batch_size": 1, "max_candidates": 4, "find": 2},
            },
        )["id"]
        assert entered.wait(2)
        live = api.json(f"/api/scans/{scan_id}")
        assert live["status"] == "running"
        assert live["summary"]["findings"] == [known]
        assert live["active"][0]["candidate"] == pending
        assert live["active"][0]["worker_id"] == "fixture-dns-1"
        assert live["active"][0]["query_type"] == "A"
        api.json(f"/api/scans/{scan_id}/pause", method="POST", payload={})
        paused = api.wait_finished(scan_id)
        assert paused["status"] == "paused"
        assert paused["summary"]["attempted"] == 1
        assert not paused["active"]
        pending_node = next(node for node in paused["nodes"] if node["value"] == pending)
        assert pending_node["status"] == "pending" and pending_node["attempts"] == 0
        api.json(f"/api/scans/{scan_id}/resume", method="POST", payload={})
        finished = api.wait_finished(scan_id)
        assert finished["status"] == "completed", finished["error"]
        assert finished["summary"]["findings"] == [known, pending]
        assert finished["summary"]["attempted"] == 2
        assert calls == [known, pending, pending]
        assert passive_calls == ["example.test"]
        assert any(event["kind"] == "finding" for event in finished["events"])


@pytest.mark.parametrize("provider_name", ["minimax", "openai", "anthropic"])
def test_http_hosted_key_stays_private_and_model_evidence_reaches_snapshot(
    tmp_path, capsys, provider_name
):
    credential = f"fixture-secret-{provider_name}-never-real"
    received, prompts = [], []
    known = "ny-01-iphone.example.test"
    proposed = "edge-new.example.test"
    rationale = "Regional service sibling based on verified evidence"

    class OfflineModel(LLMProvider):
        def __init__(self, config):
            received.append(dict(config))
            super().__init__(config)

        def get_provider_name(self):
            return "anthropic" if provider_name == "anthropic" else f"openai-compat:{provider_name}"

        def chat(self, system, user, schema):
            prompts.append(user)
            return ChatResult(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "value": proposed,
                                "confidence": 0.8,
                                # Even an accidental echo by a provider must not reach clients
                                # or checkpoints; the meaningful rationale remains visible.
                                "rationale": rationale + "; " + credential,
                            }
                        ]
                    }
                ),
                input_tokens=100,
                output_tokens=40,
                model=self.model,
            )

    class OfflineDNS:
        def __init__(self, target, **kwargs):
            pass

        def validate(self, values, **kwargs):
            return [
                DNSResult(
                    value,
                    value in {known, proposed},
                    ["192.0.2.3"] if value in {known, proposed} else [],
                    0,
                )
                for value in values
            ]

    with running_dashboard(
        tmp_path,
        dns_factory=OfflineDNS,
        passive_fetch=passive_fixture,
        provider_factory=OfflineModel,
    ) as api:
        assert api.json(
            "/api/credentials",
            method="POST",
            payload={
                "provider": provider_name,
                "api_key": credential,
            },
        ) == {"configured": True}
        scan_id = api.json(
            "/api/scans",
            method="POST",
            expected=201,
            payload={
                "target": "example.test",
                "provider": provider_name,
                "budget_usd": 1,
                "options": {"max_candidates": 40, "max_per_host": 1, "max_number": 2, "find": 2},
            },
        )["id"]
        snapshot = api.wait_finished(scan_id)
        assert snapshot["status"] == "completed", snapshot["error"]
        assert snapshot["summary"]["findings"] == [known, proposed]
        assert received[0]["api_key"] == credential
        assert received[0]["preset"] == provider_name
        assert received[0]["base_url"] == ENDPOINTS[provider_name]
        assert len(prompts) == 3 and all(known in prompt for prompt in prompts)
        node = next(node for node in snapshot["nodes"] if node["value"] == proposed)
        assert node["source"] == "model" and node["status"] == "verified"
        assert rationale in node["reason"]
        assert credential not in node["reason"]
        assert float(snapshot["summary"]["ledger"]["spent_usd"]) > 0
        assert snapshot["summary"]["ledger"]["calls"] == 3
        state = api.json("/api/state")
        assert state["credentials"][provider_name] is True
        for kind in ("json", "txt", "html"):
            assert api.request(f"/api/scans/{scan_id}/export?format={kind}")[0] == 200
        assert all(credential.encode() not in response for response in api.responses)
        for path in tmp_path.iterdir():
            if path.is_file():
                assert credential.encode() not in path.read_bytes()
        saved = json.loads((tmp_path / f"{scan_id}.json").read_text())
        validate_checkpoint(saved, "example.test")
        assert saved["provider"]["model"] == received[0]["model"]
        assert "api_key" not in json.dumps(saved)
    captured = capsys.readouterr()
    assert credential not in captured.out + captured.err
    reloaded = ScanManager(tmp_path)
    try:
        assert not any(reloaded.list_state()["credentials"].values())
        assert reloaded.snapshot(scan_id)["summary"]["findings"] == [known, proposed]
    finally:
        reloaded.close()
