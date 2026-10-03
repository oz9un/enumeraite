import json
import queue
import threading
import time

import pytest
import requests

from enumeraite.core import passive


@pytest.fixture(autouse=True)
def no_optional_executable(monkeypatch):
    monkeypatch.setattr(passive.shutil, "which", lambda name: None)


def install_sources(monkeypatch, crt_names=(), host_names=()):
    def fetch(url, params, *, deadline):
        if url == "https://crt.sh/":
            assert params["output"] == "json"
            return json.dumps([{"name_value": value} for value in crt_names]).encode()
        assert url == "https://api.hackertarget.com/hostsearch/"
        return (
            "\n".join(f"{name},192.0.2.1" for name in host_names) or "No records found"
        ).encode()

    monkeypatch.setattr(passive, "_fetch_bytes", fetch)


def test_passive_scopes_normalizes_and_retains_provenance(monkeypatch):
    install_sources(
        monkeypatch,
        crt_names=[
            "API.Example.com\n*.DEV.example.com\nexample.com",
            "api.example.com.",
            "example.com.evil.test\nnotexample.com\n*.*.example.com\nhttps://a.example.com",
            "bad\x00.example.com\nbad_name.example.com\nshop.bücher.example.com",
        ],
        host_names=["API.example.com", "mail.example.com", "evil.test"],
    )
    result = passive.collect_passive("EXAMPLE.com.")
    assert result.names == [
        "api.example.com",
        "dev.example.com",
        "mail.example.com",
        "shop.xn--bcher-kva.example.com",
    ]
    assert result.sources["hackertarget"] == ["api.example.com", "mail.example.com"]
    assert "dev.example.com" in result.sources["crt.sh"]
    assert not result.errors
    assert not result.truncated


def test_unicode_target_and_source_are_normalized_consistently(monkeypatch):
    install_sources(monkeypatch, crt_names=["www.bücher.example", "www.bucher.example"])
    result = passive.collect_passive("bücher.example")
    assert result.names == ["www.xn--bcher-kva.example"]


@pytest.mark.parametrize(
    "target",
    [
        "https://example.com",
        "*.example.com",
        "127.0.0.1",
        "localhost",
        "a..com",
        "a\x00.com",
        "xn--.example",
    ],
)
def test_invalid_target_never_makes_requests(monkeypatch, target):
    monkeypatch.setattr(passive.requests, "Session", lambda: pytest.fail("unexpected network"))
    with pytest.raises(ValueError, match="target"):
        passive.collect_passive(target)


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": 0},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"timeout": 60.01},
        {"timeout": True},
        {"timeout": "30"},
        {"timeout": None},
        {"max_results": 0},
        {"deadline": True},
        {"deadline": "30"},
        {"deadline": float("inf")},
    ],
)
def test_invalid_limits(options):
    with pytest.raises(ValueError):
        passive.collect_passive("example.com", **options)


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, 110.0),
        ({"timeout": 30}, 130.0),
        ({"timeout": 60}, 160.0),
        ({"timeout": 30, "deadline": 105.0}, 105.0),
        ({"timeout": 30, "deadline": 140.0}, 130.0),
    ],
    ids=["default-10s", "explicit-30s", "maximum-60s", "earlier-deadline", "earlier-timeout"],
)
def test_requested_time_budget_reaches_each_source_without_silent_clamping(
    monkeypatch, options, expected
):
    observed = {}

    def worker(source, target, max_results, deadline, replies):
        observed[source] = deadline
        replies.put_nowait(passive._SourceResult(source, []))

    monkeypatch.setattr(passive.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(passive, "_source_worker", worker)
    result = passive.collect_passive("example.com", **options)
    assert observed == {"crt.sh": expected, "hackertarget": expected}
    assert not result.errors


def test_sources_run_concurrently_and_one_failure_does_not_hide_other(monkeypatch):
    both_started = threading.Barrier(2)

    def fetch(url, params, *, deadline):
        both_started.wait(timeout=1)
        if url == "https://crt.sh/":
            raise requests.ConnectionError("https://secret-user:secret-token@proxy.test")
        return b"www.example.com,192.0.2.1"

    monkeypatch.setattr(passive, "_fetch_bytes", fetch)
    result = passive.collect_passive("example.com")
    assert result.names == ["www.example.com"]
    assert result.sources["crt.sh"] == []
    assert result.errors == ["crt.sh: service connection failed"]
    assert "secret" not in str(result)


def test_result_cap_is_deterministic_and_limits_provenance(monkeypatch):
    install_sources(
        monkeypatch,
        crt_names=["z.example.com", "a.example.com", "b.example.com"],
        host_names=["c.example.com", "a.example.com"],
    )
    result = passive.collect_passive("example.com", max_results=2)
    assert result.names == ["a.example.com", "b.example.com"]
    assert result.sources["hackertarget"] == ["a.example.com"]
    assert result.truncated


def test_source_sampling_keeps_semantic_evidence_among_many_numeric_names(monkeypatch):
    numeric = [f"209-64-116-{index:03d}.example.com" for index in range(100)]
    semantic = [
        "api.example.com",
        "payments.example.com",
        "sandbox.example.com",
        "auth.example.com",
    ]
    install_sources(monkeypatch, crt_names=numeric + semantic)

    result = passive.collect_passive("example.com", max_results=20)

    assert len(result.names) == 20
    assert set(semantic) <= set(result.names)
    assert set(result.names) <= set(numeric + semantic)
    assert result.names == sorted(result.names)
    assert result.sources["crt.sh"] == result.names
    assert result.truncated


def test_union_sampling_remains_fair_after_individual_source_caps(monkeypatch):
    numeric = [f"worker{index:03d}.example.com" for index in range(20)]
    semantic = [
        "api.example.com",
        "auth.example.com",
        "payments.example.com",
        "sandbox.example.com",
    ]
    install_sources(monkeypatch, crt_names=numeric, host_names=semantic)

    result = passive.collect_passive("example.com", max_results=5)

    assert result.names == sorted([numeric[0], *semantic])
    assert result.sources["crt.sh"] == [numeric[0]]
    assert result.sources["hackertarget"] == sorted(semantic)
    assert result.truncated


def test_numeric_padding_variants_share_sampling_family(monkeypatch):
    install_sources(
        monkeypatch,
        crt_names=[
            "node9.example.com",
            "node10.example.com",
            "node011.example.com",
            "z.example.com",
        ],
    )
    result = passive.collect_passive("example.com", max_results=2)
    assert result.names == ["node011.example.com", "z.example.com"]


def test_fair_sample_is_independent_of_source_completion_order(monkeypatch):
    replies = {
        "crt.sh": passive._SourceResult(
            "crt.sh", ["node1.example.com", "node2.example.com", "z.example.com"]
        ),
        "hackertarget": passive._SourceResult(
            "hackertarget", ["api.example.com", "node1.example.com"]
        ),
    }
    outcomes = []
    for first in ("crt.sh", "hackertarget"):
        first_replied = threading.Event()

        def worker(source, target, max_results, deadline, output):
            if source != first:
                assert first_replied.wait(1)
            output.put_nowait(replies[source])
            if source == first:
                first_replied.set()

        monkeypatch.setattr(passive, "_source_worker", worker)
        outcomes.append(passive.collect_passive("example.com", max_results=3))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0].names == ["api.example.com", "node1.example.com", "z.example.com"]
    assert "node1.example.com" in outcomes[0].sources["crt.sh"]
    assert "node1.example.com" in outcomes[0].sources["hackertarget"]


def test_retained_name_keeps_membership_even_if_another_source_selected_it(monkeypatch):
    def worker(source, target, max_results, deadline, replies):
        if source == "crt.sh":
            result = passive._SourceResult(
                source,
                ["a.example.com"],
                truncated=True,
                observed_names={"a.example.com", "b.example.com"},
            )
        else:
            result = passive._SourceResult(source, ["b.example.com"])
        replies.put_nowait(result)

    monkeypatch.setattr(passive, "_source_worker", worker)
    result = passive.collect_passive("example.com", max_results=2)
    assert result.names == ["a.example.com", "b.example.com"]
    assert result.sources["crt.sh"] == result.names
    assert result.sources["hackertarget"] == ["b.example.com"]
    assert result.truncated


def test_sampling_scales_without_generating_or_losing_family_representatives():
    numeric = {f"node{index:05d}.example.com" for index in range(50_000)}
    semantic = {"api.example.com", "auth.example.com", "z.example.com"}
    result = passive._sample_names(numeric | semantic, "example.com", 20)
    assert len(result) == 20
    assert semantic <= set(result)
    assert set(result) <= numeric | semantic
    assert result == sorted(result)


def test_deadline_during_normalization_retains_only_valid_completed_names(monkeypatch):
    install_sources(monkeypatch, crt_names=["API.example.com", "outside.test", "late.example.com"])
    now = [100.0]
    original = passive._scoped_name

    def normalize(value, target):
        name = original(value, target)
        now[0] += 1.0
        return name

    monkeypatch.setattr(passive, "_scoped_name", normalize)
    monkeypatch.setattr(passive.time, "monotonic", lambda: now[0])
    replies = queue.Queue()
    passive._source_worker("crt.sh", "example.com", 5, 102.0, replies)
    result = replies.get_nowait()
    assert result.names == ["api.example.com"]
    assert result.observed_names == {"api.example.com"}
    assert result.truncated
    assert "timed out" in result.error


def test_free_service_limit_marks_result_as_partial(monkeypatch):
    install_sources(monkeypatch, host_names=[f"host{i}.example.com" for i in range(50)])
    result = passive.collect_passive("example.com")
    assert len(result.names) == 50
    assert result.truncated


def test_source_errors_are_not_reported_as_a_successful_empty_search(monkeypatch):
    def fetch(url, params, *, deadline):
        return b"API count exceeded" if "hackertarget" in url else b"<html>Unavailable</html>"

    monkeypatch.setattr(passive, "_fetch_bytes", fetch)
    result = passive.collect_passive("example.com")
    assert not result.names
    assert len(result.errors) == 2


def test_source_size_limit_is_visible(monkeypatch):
    def fetch(url, params, *, deadline):
        raise passive._ResponseTooLarge()

    monkeypatch.setattr(passive, "_fetch_bytes", fetch)
    result = passive.collect_passive("example.com")
    assert result.truncated
    assert len(result.errors) == 2
    assert all("8 MiB" in error for error in result.errors)


def test_expired_deadline_skips_network(monkeypatch):
    monkeypatch.setattr(passive.requests, "Session", lambda: pytest.fail("unexpected network"))
    result = passive.collect_passive("example.com", deadline=time.monotonic() - 1)
    assert not result.names
    assert len(result.errors) == 2
    assert result.truncated


def test_uninterruptible_lookup_cannot_hold_up_the_caller(monkeypatch):
    release = threading.Event()

    def fetch(url, params, *, deadline):
        release.wait(timeout=2)
        return b"[]" if "crt.sh" in url else b"No records found"

    monkeypatch.setattr(passive, "_fetch_bytes", fetch)
    start = time.monotonic()
    try:
        result = passive.collect_passive("example.com", timeout=0.04)
        elapsed = time.monotonic() - start
    finally:
        release.set()
    assert elapsed < 0.5
    assert result.truncated
    assert len(result.errors) == 2


class FakeResponse:
    def __init__(self, body=b"[]", status=200, headers=None):
        self.body = body
        self.status_code = status
        self.headers = headers or {}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def iter_content(self, chunk_size):
        yield self.body


def install_session(monkeypatch, response):
    calls = []

    class Session:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get(self, url, **kwargs):
            calls.append(kwargs)
            assert url == "https://crt.sh/"
            assert not self.trust_env
            assert kwargs["allow_redirects"] is False
            assert kwargs["stream"] is True
            assert kwargs["headers"]["Accept-Encoding"] == "identity"
            assert kwargs["timeout"][0] <= 3
            assert kwargs["timeout"][1] <= 10
            return response

    monkeypatch.setattr(passive.requests, "Session", Session)
    return calls


@pytest.mark.parametrize("remaining", [0.1, 5.0, 30.0])
def test_socket_timeouts_follow_remaining_budget_without_one_second_read_cap(
    monkeypatch, remaining
):
    response = FakeResponse()
    calls = install_session(monkeypatch, response)
    monkeypatch.setattr(passive.time, "monotonic", lambda: 100.0)
    assert passive._fetch_bytes("https://crt.sh/", {}, deadline=100.0 + remaining) == b"[]"
    assert calls[0]["timeout"] == pytest.approx((min(3.0, remaining), min(10.0, remaining)))
    assert response.closed


def test_response_taking_over_one_second_is_accepted_within_explicit_deadline(monkeypatch):
    now = [100.0]
    response = FakeResponse()
    calls = install_session(monkeypatch, response)
    original_get = passive.requests.Session.get

    def delayed_get(session, url, **kwargs):
        assert kwargs["timeout"][1] >= 2.0
        now[0] += 2.0
        return original_get(session, url, **kwargs)

    monkeypatch.setattr(passive.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(passive.requests.Session, "get", delayed_get)
    assert passive._fetch_bytes("https://crt.sh/", {}, deadline=130.0) == b"[]"
    assert calls[0]["timeout"] == (3.0, 10.0)
    assert response.closed


@pytest.mark.parametrize(
    "response, expected",
    [
        (FakeResponse(status=302), "redirect refused"),
        (FakeResponse(status=429), "rate limit"),
        (FakeResponse(status=503), "HTTP 503"),
        (FakeResponse(headers={"Content-Encoding": "gzip"}), "uncompressed"),
        (FakeResponse(headers={"Content-Length": str(9 * 1024 * 1024)}), "^$"),
    ],
)
def test_request_rejects_unsafe_or_failed_responses_and_closes(monkeypatch, response, expected):
    install_session(monkeypatch, response)
    with pytest.raises(passive._SourceError, match=expected):
        passive._fetch_bytes("https://crt.sh/", {}, deadline=time.monotonic() + 5)
    assert response.closed


def test_stream_body_is_bounded_even_without_content_length(monkeypatch):
    monkeypatch.setattr(passive, "_MAX_RESPONSE_BYTES", 8)
    response = FakeResponse(body=b"123456789")
    install_session(monkeypatch, response)
    with pytest.raises(passive._ResponseTooLarge):
        passive._fetch_bytes("https://crt.sh/", {}, deadline=time.monotonic() + 5)
    assert response.closed


def test_drip_feed_uses_raw_single_reads_and_checks_deadline(monkeypatch):
    class Raw:
        def read1(self, size, *, decode_content):
            assert decode_content is False
            return b"."

    response = FakeResponse()
    response.raw = Raw()
    ticks = iter([0.0, 0.1, 0.2, 1.1])
    monkeypatch.setattr(passive.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError):
        passive._read_body(response, deadline=1.0)


def test_valid_response_is_read_and_closed(monkeypatch):
    response = FakeResponse(b"[]")
    install_session(monkeypatch, response)
    assert passive._fetch_bytes("https://crt.sh/", {}, deadline=time.monotonic() + 5) == b"[]"
    assert response.closed


@pytest.mark.parametrize("body", [b"{}", b"[null]", b"[1]", b"[{}]", b'[{"name_value": null}]'])
def test_invalid_certificate_schema_is_a_failure(body):
    with pytest.raises(passive._SourceError):
        passive._parse_crtsh(body)


@pytest.mark.parametrize(
    "body", [b"<html>nope</html>", b"a.example.com,bogus", b"error invalid query"]
)
def test_invalid_hostsearch_schema_is_a_failure(body):
    with pytest.raises(passive._SourceError):
        passive._parse_hostsearch(body)
