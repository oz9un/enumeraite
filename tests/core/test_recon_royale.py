import json

import pytest

import enumeraite.core.recon_royale as recon_royale_module
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.recon_royale import ReconRoyaleClient, filter_in_scope, prepare_submission


class FakeDNS:
    def validate_subdomains(self, values, detect_wildcard=False, apex=None):
        return [
            DNSResult(
                value,
                not value.startswith("bad"),
                ["1.2.3.4"] if not value.startswith("bad") else [],
                0.01,
            )
            for value in values
        ]


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.text = ""
        self.status_code = status_code
        self.headers = {}
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=16_384):
        yield json.dumps(self.payload).encode("utf-8")

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self):
        self.posted = None
        self.get_kwargs = None
        self.post_kwargs = None

    def get(self, *args, **kwargs):
        self.get_kwargs = kwargs
        return FakeResponse({"target": "example.com"})

    def post(self, *args, **kwargs):
        self.post_kwargs = kwargs
        self.posted = kwargs["json"]
        return FakeResponse({"ok": True})


class InvalidTargetSession(FakeSession):
    def get(self, *args, **kwargs):
        return FakeResponse({"target": "https://example.com/path"})


class RedirectSession(FakeSession):
    def get(self, *args, **kwargs):
        self.get_kwargs = kwargs
        return FakeResponse({}, status_code=302)


def test_authenticated_response_body_has_a_wall_clock_deadline(monkeypatch):
    session = FakeSession()
    response = FakeResponse({"target": "example.com"})
    session.get = lambda *args, **kwargs: response
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(recon_royale_module.time, "perf_counter", lambda: next(clock))

    with pytest.raises(TimeoutError, match="deadline"):
        ReconRoyaleClient("secret", timeout=1.0, session=session).current_target()
    assert response.closed


def test_client_contract_and_submission_payload():
    session = FakeSession()
    client = ReconRoyaleClient("secret", session=session)
    assert client.current_target() == "example.com"
    assert client.submit(["api.example.com"]) == {"ok": True}
    assert session.posted == {"subdomains": ["api.example.com"]}
    assert session.get_kwargs["allow_redirects"] is False
    assert session.post_kwargs["allow_redirects"] is False
    assert session.get_kwargs["stream"] is True
    assert session.post_kwargs["stream"] is True
    assert session.get_kwargs["headers"]["Accept-Encoding"] == "identity"
    assert session.post_kwargs["headers"]["Accept-Encoding"] == "identity"


def test_client_rejects_invalid_target_and_submission_values():
    client = ReconRoyaleClient("secret", session=InvalidTargetSession())
    try:
        client.current_target()
    except ValueError as exc:
        assert "invalid target" in str(exc)
    else:
        raise AssertionError("expected target validation failure")

    session = FakeSession()
    client = ReconRoyaleClient("secret", session=session)
    client.submit(["API.EXAMPLE.COM.", "not a domain", "https://outside.test"])
    assert session.posted == {"subdomains": ["api.example.com"]}


def test_client_refuses_authenticated_redirects_and_unsafe_base_urls():
    client = ReconRoyaleClient("secret", session=RedirectSession())
    try:
        client.current_target()
    except ValueError as exc:
        assert "redirect refused" in str(exc)
    else:
        raise AssertionError("expected redirect refusal")

    for base_url in (
        "ftp://example.com",
        "http://example.com",
        "https://user:secret@example.com",
        "https://x/?q=1",
    ):
        try:
            ReconRoyaleClient("secret", base_url=base_url)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected unsafe URL rejection: {base_url}")

    assert ReconRoyaleClient("secret", base_url="http://127.0.0.1:8080").base_url == (
        "http://127.0.0.1:8080"
    )


def test_client_rejects_unsafe_credentials_and_oversized_responses():
    with pytest.raises(ValueError, match="API key"):
        ReconRoyaleClient("secret\nInjected: true")
    with pytest.raises(ValueError, match="API key"):
        ReconRoyaleClient("secret with spaces")
    with pytest.raises(ValueError, match="timeout"):
        ReconRoyaleClient("secret", timeout=True)

    class OversizedSession(FakeSession):
        def get(self, *args, **kwargs):
            response = FakeResponse({"target": "example.com"})
            response.headers["Content-Length"] = "1000001"
            return response

    with pytest.raises(ValueError, match="1 MB"):
        ReconRoyaleClient("secret", session=OversizedSession()).current_target()

    class OversizedChunkSession(FakeSession):
        def get(self, *args, **kwargs):
            response = FakeResponse({"target": "example.com"})
            response.iter_content = lambda chunk_size=16_384: iter((b"x" * 1_000_001,))
            return response

    with pytest.raises(ValueError, match="1 MB"):
        ReconRoyaleClient("secret", session=OversizedChunkSession()).current_target()


def test_prepare_submission_scopes_resolves_and_measures_margin():
    report = prepare_submission(
        "example.com",
        ["api.example.com", "bad-passive.example.com"],
        ["api.example.com", "new.example.com", "outside.test"],
        validator=FakeDNS(),
        min_valid_rate=0.5,
    )
    assert report.eligible
    assert report.marginal_enumeraite == ["new.example.com"]
    assert report.enumeraite_valid_rate == 1.0
    assert "outside.test" not in report.valid_subdomains


def test_passive_results_cannot_mask_invalid_generation():
    report = prepare_submission(
        "example.com",
        [f"passive-{index}.example.com" for index in range(10)],
        ["bad-generated.example.com"],
        validator=FakeDNS(),
        min_valid_rate=0.1,
    )
    assert report.valid_rate > 0.9
    assert report.enumeraite_valid_rate == 0.0
    assert not report.eligible
    assert "Enumeraite valid rate" in report.reason


def test_submission_requires_marginal_enumeraite_finding():
    report = prepare_submission(
        "example.com",
        ["same.example.com"],
        ["same.example.com"],
        validator=FakeDNS(),
        min_valid_rate=0.1,
    )
    assert report.enumeraite_valid_rate == 1.0
    assert not report.eligible
    assert "no marginal" in report.reason


def test_scope_filter_requires_a_proper_subdomain():
    assert filter_in_scope(
        [
            "example.com",
            "api.example.com",
            "fakeexample.com",
            "API.EXAMPLE.COM.",
            "bad..example.com",
        ],
        "example.com",
    ) == ["api.example.com"]
    with pytest.raises(ValueError, match="not a string"):
        filter_in_scope("api.example.com", "example.com")
    with pytest.raises(ValueError, match="sequence"):
        ReconRoyaleClient("secret", session=FakeSession()).submit("api.example.com")
    with pytest.raises(ValueError, match="only DNS-name strings"):
        filter_in_scope(["api.example.com", 7], "example.com")


def test_submission_workflows_reject_oversized_candidate_sets():
    values = [f"host-{index}.example.com" for index in range(10_001)]
    client = ReconRoyaleClient("secret", session=FakeSession())

    with pytest.raises(ValueError, match="10000-item limit"):
        client.submit(values)
    with pytest.raises(ValueError, match="10000-candidate validation limit"):
        prepare_submission("example.com", [], values, validator=FakeDNS())


def test_prepare_submission_rejects_invalid_target_and_validator_injection():
    try:
        prepare_submission("https://example.com", [], [], validator=FakeDNS())
    except ValueError as exc:
        assert "valid DNS apex" in str(exc)
    else:
        raise AssertionError("expected target rejection")

    class InjectingDNS:
        def validate_subdomains(self, values, detect_wildcard=False, apex=None):
            return [DNSResult("injected.example.com", True, ["1.2.3.4"], 0.01)]

    report = prepare_submission(
        "example.com", [], ["candidate.example.com"], validator=InjectingDNS(), min_valid_rate=0.1
    )
    assert report.valid_subdomains == []
    assert not report.eligible
