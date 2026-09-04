import dns.exception
import dns.resolver
import pytest

from enumeraite.core.dns_validator import DNSResult, DNSValidator
from enumeraite.core.http_validator import (
    HTTPValidator,
    PathHTTPValidator,
    _bounded_response_text,
    _normalize_base_url,
)


class Response:
    def __init__(self, status_code, text, location=""):
        self.status_code = status_code
        self.text = text
        self.location = location


def test_streamed_http_body_has_a_wall_clock_deadline(monkeypatch):
    class Raw:
        called = False

        def read1(self, amount, decode_content=True):
            assert decode_content is False
            self.called = True
            return b"late"

    class StreamResponse:
        encoding = "utf-8"
        closed = False
        raw = Raw()

        def iter_content(self, chunk_size=4_096):
            raise AssertionError("real urllib3 responses must use deadline-checkable read1")

        def close(self):
            self.closed = True

    response = StreamResponse()
    clock = iter((0.0, 2.0))
    monkeypatch.setattr("enumeraite.core.http_validator.time.perf_counter", lambda: next(clock))
    with pytest.raises(TimeoutError, match="deadline"):
        _bounded_response_text(response, 1_024, deadline=1.0)
    assert response.raw.called
    assert response.closed


def test_path_validator_soft_404_and_auth_findings(monkeypatch):
    validator = PathHTTPValidator(max_concurrent=1)

    def fetch(url):
        if "enumeraite-not-found" in url:
            return Response(200, "custom missing page"), 0.01, None
        if url.endswith("/ghost"):
            return Response(200, "custom missing page"), 0.01, None
        if url.endswith("/secret"):
            return Response(403, "custom missing page"), 0.01, None
        return Response(200, "real application content with records"), 0.01, None

    monkeypatch.setattr(validator, "_fetch", fetch)
    results = {
        result.path: result
        for result in validator.validate_paths("example.com", ["/ghost", "/secret", "/real"])
    }
    assert not results["/ghost"].exists
    assert results["/secret"].exists
    assert results["/real"].exists


def test_path_validator_calibrates_virtual_router_prefixes(monkeypatch):
    validator = PathHTTPValidator(
        max_concurrent=2,
        baseline_samples=2,
        baseline_depth=1,
    )
    requested = []

    def fetch(url):
        requested.append(url)
        if "enumeraite-not-found" in url and "/api/" in url:
            return Response(200, "api router miss 12345"), 0.01, None
        if "enumeraite-not-found" in url:
            return Response(404, "root miss"), 0.01, None
        if url.endswith("/api/ghost"):
            return Response(200, "api router miss 67890"), 0.01, None
        return Response(200, "real route payload"), 0.01, None

    monkeypatch.setattr(validator, "_fetch", fetch)
    results = {
        result.path: result
        for result in validator.validate_paths("example.test", ["/api/ghost", "/api/real"])
    }
    assert not results["/api/ghost"].exists
    assert results["/api/real"].exists
    assert sum("/api/enumeraite-not-found" in url for url in requested) == 2


def test_path_validator_baseline_group_cap_includes_root_fallback(monkeypatch):
    validator = PathHTTPValidator(
        max_concurrent=1,
        baseline_samples=1,
        baseline_depth=1,
        max_baseline_groups=2,
    )
    requested = []

    def fetch(url):
        requested.append(url)
        return Response(404, "missing"), 0.01, None

    monkeypatch.setattr(validator, "_fetch", fetch)
    validator.validate_paths("example.test", ["/one/a", "/two/a", "/three/a"])
    baseline_requests = [url for url in requested if "enumeraite-not-found" in url]
    assert len(baseline_requests) == 2
    assert any("example.test/enumeraite-not-found" in url for url in baseline_requests)


def test_path_validator_calibrates_non_get_route_statuses(monkeypatch):
    validator = PathHTTPValidator(max_concurrent=1, baseline_samples=1)

    def fetch(url):
        if "enumeraite-not-found" in url:
            return Response(404, "missing"), 0.01, None
        return Response(405, "use POST"), 0.01, None

    monkeypatch.setattr(validator, "_fetch", fetch)
    result = validator.validate_paths("example.test", ["/submit"])[0]
    assert result.exists
    assert result.status_code == 405


def test_path_validator_suppresses_global_auth_and_reports_failed_calibration(
    monkeypatch,
):
    validator = PathHTTPValidator(max_concurrent=1, baseline_samples=1)

    monkeypatch.setattr(
        validator,
        "_fetch",
        lambda url: (Response(403, "global access denied"), 0.01, None),
    )
    assert not validator.validate_paths("example.test", ["/secret"])[0].exists

    def unavailable(url):
        if "enumeraite-not-found" in url:
            return None, 0.01, "timeout"
        return Response(200, "candidate"), 0.01, None

    monkeypatch.setattr(validator, "_fetch", unavailable)
    result = validator.validate_paths("example.test", ["/candidate"])[0]
    assert not result.exists
    assert result.error == "soft-404 calibration unavailable"


def test_dns_wildcard_candidates_are_suppressed(monkeypatch):
    validator = DNSValidator()

    def resolve(hostname):
        return DNSResult(hostname, True, ["203.0.113.10"], 0.01)

    monkeypatch.setattr(validator, "_validate_single_subdomain", resolve)
    result = validator.validate_subdomains(
        ["candidate.example.com"], detect_wildcard=True, apex="example.com"
    )[0]
    assert result.wildcard
    assert not result.exists


def test_dns_wildcard_detection_handles_rotating_addresses(monkeypatch):
    validator = DNSValidator()
    addresses = iter(["203.0.113.1", "203.0.113.2", "203.0.113.3"])

    def resolve(hostname):
        return DNSResult(hostname, True, [next(addresses)], 0.01)

    monkeypatch.setattr(validator, "_validate_single_subdomain", resolve)
    wildcard = validator.detect_wildcard("example.com")
    assert wildcard.detected
    assert wildcard.ip_addresses == [
        "203.0.113.1",
        "203.0.113.2",
        "203.0.113.3",
    ]


def test_dns_wildcard_calibration_failure_makes_positive_results_inconclusive(
    monkeypatch,
):
    validator = DNSValidator()

    def resolve(hostname):
        if hostname.startswith("candidate"):
            return DNSResult(hostname, True, ["203.0.113.9"], 0.01)
        return DNSResult(hostname, False, [], 0.01, error="timeout")

    monkeypatch.setattr(validator, "_validate_single_subdomain", resolve)
    result = validator.validate_subdomains(
        ["candidate.example.com"], detect_wildcard=True, apex="example.com"
    )[0]
    assert not result.exists
    assert result.error == "wildcard DNS calibration inconclusive"


def test_dns_wildcard_detection_requires_explicit_apex(monkeypatch):
    validator = DNSValidator()
    monkeypatch.setattr(
        validator,
        "_validate_single_subdomain",
        lambda hostname: DNSResult(hostname, False, [], 0.01),
    )
    with pytest.raises(ValueError, match="apex is required"):
        validator.validate_subdomains(["api.example.co.uk"], detect_wildcard=True)
    with pytest.raises(ValueError, match="between 2 and 10"):
        validator.detect_wildcard("example.com", probes=11)
    with pytest.raises(ValueError, match="valid DNS name"):
        validator.validate_subdomains(
            ["api.example.com"], detect_wildcard=True, apex="https://example.com"
        )


def test_dns_validator_uses_per_query_resolver_timeout(monkeypatch):
    validator = DNSValidator(timeout=1.25)
    resolver = type("Resolver", (), {})()
    resolver.timeout = None
    resolver.lifetime = None
    calls = []

    def resolve(hostname, record_type, **kwargs):
        calls.append((hostname, record_type, kwargs["lifetime"]))
        return ["203.0.113.9"] if record_type == "A" else []

    resolver.resolve = resolve
    monkeypatch.setattr(
        "enumeraite.core.dns_validator.dns.resolver.Resolver",
        lambda configure=True: resolver,
    )
    result = validator._validate_single_subdomain("api.example.com")
    assert result.exists
    assert result.ip_addresses == ["203.0.113.9"]
    assert calls == [
        ("api.example.com", "A", 1.25),
        ("api.example.com", "AAAA", 1.25),
    ]
    assert resolver.timeout == 1.25
    assert resolver.lifetime == 1.25


@pytest.mark.parametrize(
    "failure",
    [dns.exception.Timeout, dns.resolver.NoNameservers],
)
def test_dns_transport_failures_are_inconclusive(monkeypatch, failure):
    validator = DNSValidator(timeout=0.1)

    class Resolver:
        timeout = None
        lifetime = None

        def resolve(self, *args, **kwargs):
            raise failure()

    monkeypatch.setattr(
        "enumeraite.core.dns_validator.dns.resolver.Resolver",
        lambda configure=True: Resolver(),
    )
    result = validator._validate_single_subdomain("api.example.com")
    assert not result.exists
    assert result.error


def test_dns_positive_answer_wins_over_partial_timeout(monkeypatch):
    validator = DNSValidator(timeout=0.1)

    class Resolver:
        timeout = None
        lifetime = None

        def resolve(self, hostname, record_type, **kwargs):
            if record_type == "A":
                return ["203.0.113.9"]
            raise dns.exception.Timeout()

    monkeypatch.setattr(
        "enumeraite.core.dns_validator.dns.resolver.Resolver",
        lambda configure=True: Resolver(),
    )
    result = validator._validate_single_subdomain("api.example.com")
    assert result.exists
    assert result.ip_addresses == ["203.0.113.9"]
    assert result.error is None


def test_dns_authoritative_nxdomain_wins_over_partial_timeout(monkeypatch):
    validator = DNSValidator(timeout=0.1)

    class Resolver:
        timeout = None
        lifetime = None
        calls = 0

        def resolve(self, hostname, record_type, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise dns.exception.Timeout()
            raise dns.resolver.NXDOMAIN()

    monkeypatch.setattr(
        "enumeraite.core.dns_validator.dns.resolver.Resolver",
        lambda configure=True: Resolver(),
    )
    result = validator._validate_single_subdomain("missing.example.com")
    assert not result.exists
    assert result.error is None


def test_dns_validator_normalizes_and_rejects_before_resolving(monkeypatch):
    validator = DNSValidator(max_concurrent=1)
    calls = []

    def resolve(hostname):
        calls.append(hostname)
        return DNSResult(hostname, True, ["203.0.113.9"], 0.01)

    monkeypatch.setattr(validator, "_validate_single_subdomain", resolve)
    results = validator.validate_subdomains(
        ["API.Example.COM.", "api.example.com", "bad..example.com", "https://example.com"]
    )
    assert calls == ["api.example.com"]
    assert [result.subdomain for result in results] == [
        "api.example.com",
        "bad..example.com",
        "https://example.com",
    ]
    assert results[1].error == "invalid DNS name"
    assert results[2].error == "invalid DNS name"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout": float("nan")}, "timeout"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": "5"}, "timeout"),
        ({"max_concurrent": 0}, "max_concurrent"),
    ],
)
def test_dns_validator_rejects_invalid_resource_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DNSValidator(**kwargs)


def test_validators_reject_string_inputs_instead_of_enumerating_characters():
    with pytest.raises(ValueError, match="subdomains must be a list"):
        DNSValidator().validate_subdomains("api.example.test")
    with pytest.raises(ValueError, match="only DNS-name strings"):
        DNSValidator().validate_subdomains(["api.example.test", 7])
    with pytest.raises(ValueError, match="10000-item"):
        DNSValidator().validate_subdomains(["api.example.test"] * 10_001)
    with pytest.raises(ValueError, match="DNS-name string"):
        DNSValidator().detect_wildcard(7)
    with pytest.raises(ValueError, match="paths must be a list"):
        PathHTTPValidator().validate_paths("example.test", "/api")
    with pytest.raises(ValueError, match="only path strings"):
        PathHTTPValidator().validate_paths("example.test", ["/api", 7])
    with pytest.raises(ValueError, match="10000-item"):
        PathHTTPValidator().validate_paths("example.test", ["/api"] * 10_001)
    with pytest.raises(ValueError, match="subdomains must be a list"):
        HTTPValidator().validate_subdomains("api.example.test")
    with pytest.raises(ValueError, match="only DNS-name strings"):
        HTTPValidator().validate_subdomains(["api.example.test", 7])
    with pytest.raises(ValueError, match="boolean"):
        HTTPValidator().validate_subdomains(["api.example.test"], check_both_protocols="yes")


def test_path_validator_normalizes_dynamic_soft_404_tokens(monkeypatch):
    validator = PathHTTPValidator(max_concurrent=1, baseline_samples=2)
    counter = {"baseline": 0}

    def fetch(url):
        if "enumeraite-not-found" in url:
            counter["baseline"] += 1
            return (
                Response(
                    200,
                    f"missing request {counter['baseline']} at 2026-09-02T10:00:0{counter['baseline']}Z",
                ),
                0.01,
                None,
            )
        return Response(200, "missing request 987654 at 2026-09-02T10:00:59Z"), 0.01, None

    monkeypatch.setattr(validator, "_fetch", fetch)
    result = validator.validate_paths("example.com", ["/ghost"])[0]
    assert not result.exists
    assert result.soft_404_similarity == 1.0


def test_path_validator_compares_redirect_locations_for_soft_404(monkeypatch):
    validator = PathHTTPValidator(max_concurrent=1, baseline_samples=2)

    def fetch(url):
        if "enumeraite-not-found" in url or url.endswith("/ghost"):
            return Response(302, "", "/missing/12345"), 0.01, None
        return Response(302, "", "/auth/login"), 0.01, None

    monkeypatch.setattr(validator, "_fetch", fetch)
    results = {
        result.path: result
        for result in validator.validate_paths("example.test", ["/ghost", "/real"])
    }
    assert not results["/ghost"].exists
    assert results["/ghost"].soft_404_similarity == 1.0
    assert results["/real"].exists
    assert results["/real"].soft_404_similarity == 0.0


def test_path_validator_normalizes_target_url_without_changing_base_path():
    assert (
        _normalize_base_url("api.example.test/v1/?debug=1#fragment")
        == "https://api.example.test/v1"
    )
    assert _normalize_base_url("http://api.example.test/") == "http://api.example.test"
    with pytest.raises(ValueError, match="HTTP"):
        _normalize_base_url("ftp://api.example.test/files")
    with pytest.raises(ValueError, match="credentials"):
        _normalize_base_url("https://user:secret@api.example.test/v1")
    with pytest.raises(ValueError, match="unsafe path"):
        _normalize_base_url("https://api.example.test/%2e%2e/admin")
    with pytest.raises(ValueError, match="invalid hostname"):
        _normalize_base_url("https://api.example%0a.test/v1")
    with pytest.raises(ValueError, match="invalid port"):
        _normalize_base_url("https://api.example.test:notaport/v1")
    with pytest.raises(ValueError, match="HTTP"):
        _normalize_base_url(7)


def test_path_validator_does_not_request_unsafe_candidates(monkeypatch):
    validator = PathHTTPValidator(max_concurrent=1, baseline_samples=1)
    requested = []

    def fetch(url):
        requested.append(url)
        return Response(404, "missing"), 0.01, None

    monkeypatch.setattr(validator, "_fetch", fetch)
    results = validator.validate_paths(
        "example.test/api", ["/%2e%2e/admin", "/safe?debug=true", "/safe"]
    )
    assert [result.exists for result in results] == [False, False, False]
    assert results[0].error == "invalid or unsafe path candidate"
    assert results[1].error == "invalid or unsafe path candidate"
    assert all("%2e%2e" not in value and "debug=true" not in value for value in requested)
    assert requested[-1] == "https://example.test/api/safe"


def test_legacy_subdomain_validator_does_not_follow_redirects(monkeypatch):
    captured = {}

    class LegacyResponse:
        status_code = 302
        headers = {"Location": "https://outside.example/"}
        text = "redirect"

    def get(url, **kwargs):
        captured.update(kwargs)
        return LegacyResponse()

    monkeypatch.setattr("enumeraite.core.http_validator.requests.get", get)
    result = HTTPValidator()._validate_single_url(
        "https://api.example.test", "api.example.test", "https"
    )
    assert result.accessible
    assert result.status_code == 302
    assert captured["allow_redirects"] is False
    assert captured["verify"] is True
    assert captured["headers"]["Accept-Encoding"] == "identity"


def test_legacy_http_validator_is_stable_and_rejects_invalid_hosts(monkeypatch):
    validator = HTTPValidator(max_concurrent=2)
    requested = []

    def validate(url, subdomain, protocol):
        requested.append(url)
        return type(
            "Result",
            (),
            {
                "subdomain": subdomain,
                "url": url,
                "accessible": protocol == "http",
                "status_code": 200 if protocol == "http" else None,
                "response_time": 0.01,
                "protocol": protocol,
                "title": None,
                "server": None,
                "error": None if protocol == "http" else "failed",
            },
        )()

    monkeypatch.setattr(validator, "_validate_single_url", validate)
    results = validator.validate_subdomains(["B.Example.COM.", "bad..example.com", "a.example.com"])
    assert [result.subdomain for result in results] == [
        "b.example.com",
        "bad..example.com",
        "a.example.com",
    ]
    assert results[0].protocol == "http"
    assert results[1].error == "invalid DNS name"
    assert all("bad..example.com" not in url for url in requested)


@pytest.mark.parametrize(
    "validator",
    [
        lambda: PathHTTPValidator(timeout=0),
        lambda: PathHTTPValidator(timeout="10"),
        lambda: PathHTTPValidator(max_concurrent=0),
        lambda: PathHTTPValidator(soft_404_threshold=True),
        lambda: PathHTTPValidator(soft_404_threshold=float("nan")),
        lambda: PathHTTPValidator(baseline_depth=11),
        lambda: PathHTTPValidator(max_baseline_groups=0),
        lambda: PathHTTPValidator(baseline_samples=0),
        lambda: PathHTTPValidator(max_body_bytes=100),
        lambda: HTTPValidator(timeout=float("inf")),
        lambda: HTTPValidator(timeout="10"),
        lambda: HTTPValidator(max_body_bytes=0),
        lambda: HTTPValidator(max_body_bytes=True),
        lambda: HTTPValidator(max_body_bytes=10_000_001),
    ],
)
def test_http_validators_reject_invalid_resource_limits(validator):
    with pytest.raises(ValueError):
        validator()
