import json
import queue
import threading
import time
from datetime import datetime, timezone

import pytest
import requests

from enumeraite.core import passive


@pytest.fixture(autouse=True)
def no_optional_executable(monkeypatch, tmp_path):
    monkeypatch.delenv("MERKLEMAP_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(passive.shutil, "which", lambda name: None)


def install_sources(monkeypatch, crt_names=()):
    def fetch(url, params, *, deadline):
        assert url == "https://crt.sh/"
        assert params["output"] == "json"
        return json.dumps([{"name_value": value} for value in crt_names]).encode()

    monkeypatch.setattr(passive, "_fetch_bytes", fetch)


def test_passive_scopes_normalizes_and_retains_provenance(monkeypatch):
    install_sources(
        monkeypatch,
        crt_names=[
            "API.Example.com\n*.DEV.example.com\nexample.com",
            "api.example.com.",
            "mail.example.com",
            "example.com.evil.test\nnotexample.com\n*.*.example.com\nhttps://a.example.com",
            "bad\x00.example.com\nbad_name.example.com\nshop.bücher.example.com",
        ],
    )
    result = passive.collect_passive("EXAMPLE.com.")
    assert result.names == [
        "api.example.com",
        "dev.example.com",
        "mail.example.com",
        "shop.xn--bcher-kva.example.com",
    ]
    assert result.sources["crt.sh"] == result.names
    assert not result.errors
    assert not result.truncated
    assert set(result.evidence) == set(result.names)
    assert {
        (record["source"], record["matched_name"]) for record in result.evidence["api.example.com"]
    } == {
        ("crt.sh", "API.Example.com"),
        ("crt.sh", "api.example.com."),
    }
    assert result.evidence["dev.example.com"][0]["matched_name"] == "*.DEV.example.com"


def test_unicode_target_and_source_are_normalized_consistently(monkeypatch):
    install_sources(monkeypatch, crt_names=["www.bücher.example", "www.bucher.example"])
    result = passive.collect_passive("bücher.example")
    assert result.names == ["www.xn--bcher-kva.example"]


def test_certificate_evidence_preserves_individual_matches_and_historical_fields(monkeypatch):
    certificate = {
        "name_value": "*.API.Example.com\nother.example.com\noutside.test",
        "common_name": "cn.example.com",
        "id": 123456789,
        "issuer_name": "C=US, O=Example CA",
        "not_before": "2020-01-01T00:00:00",
        "not_after": "2021-01-01T00:00:00",
        "entry_timestamp": "2020-01-02T03:04:05.123",
        "href": "https://untrusted.test/",
    }

    def fetch(url, params, *, deadline):
        assert url == "https://crt.sh/"
        assert params == {"q": "%.example.com", "output": "json"}
        return json.dumps([certificate]).encode()

    monkeypatch.setattr(passive, "_fetch_bytes", fetch)
    before = datetime.now(timezone.utc)
    result = passive.collect_passive("EXAMPLE.com.")
    after = datetime.now(timezone.utc)
    assert result.names == ["api.example.com", "cn.example.com", "other.example.com"]
    records = result.evidence["api.example.com"]
    assert [
        {key: value for key, value in record.items() if key != "collected_at"} for record in records
    ] == [
        {
            "source": "crt.sh",
            "query": "%.example.com",
            "matched_name": "*.API.Example.com",
            "certificate_id": "123456789",
            "issuer": "C=US, O=Example CA",
            "not_before": "2020-01-01T00:00:00",
            "not_after": "2021-01-01T00:00:00",
            "entry_timestamp": "2020-01-02T03:04:05.123",
        },
    ]
    for record in records:
        assert before <= datetime.fromisoformat(record["collected_at"]) <= after
    assert result.evidence["other.example.com"][0]["matched_name"] == "other.example.com"
    assert result.evidence["cn.example.com"] == [{**records[0], "matched_name": "cn.example.com"}]
    assert not result.errors and not result.truncated
    assert set(result.evidence) == set(result.names)


@pytest.mark.parametrize(
    "malformed_row",
    [None, 1, {}, {"name_value": None}, {"name_value": ["not-a-name.example.com"]}],
)
def test_malformed_certificate_rows_keep_valid_observations_and_warn(monkeypatch, malformed_row):
    certificates = [
        {"name_value": "first.example.com", "id": 1},
        malformed_row,
        {"name_value": None, "common_name": "cn.example.com", "id": 2},
        {"name_value": "san.example.com", "common_name": 99, "id": 3},
        {"common_name": "last.example.com", "id": 4},
    ]
    monkeypatch.setattr(
        passive, "_fetch_bytes", lambda *args, **kwargs: json.dumps(certificates).encode()
    )

    result = passive.collect_passive("example.com")

    assert result.names == [
        "cn.example.com",
        "first.example.com",
        "last.example.com",
        "san.example.com",
    ]
    assert result.sources["crt.sh"] == result.names
    assert {
        name: [record["certificate_id"] for record in records]
        for name, records in result.evidence.items()
    } == {
        "first.example.com": ["1"],
        "cn.example.com": ["2"],
        "san.example.com": ["3"],
        "last.example.com": ["4"],
    }
    assert result.truncated
    assert len(result.errors) == 1
    assert "malformed" in result.errors[0]


def test_evidence_caps_distinct_records_per_collector_without_retaining_dropped_names(monkeypatch):
    certificates = [
        {
            "name_value": "api.example.com\nz.example.com",
            "common_name": "api.example.com",
            "id": certificate_id,
        }
        for certificate_id in (1, 1, 2, 3, 4, 5)
    ]

    def fetch(url, params, *, deadline):
        assert url == "https://crt.sh/"
        return json.dumps(certificates).encode()

    monkeypatch.setattr(passive, "_fetch_bytes", fetch)
    result = passive.collect_passive("example.com", max_results=1)
    assert result.names == ["api.example.com"]
    assert set(result.evidence) == {"api.example.com"}
    records = result.evidence["api.example.com"]
    assert [record["certificate_id"] for record in records if record["source"] == "crt.sh"] == [
        "1",
        "2",
        "3",
    ]
    assert result.sources == {"crt.sh": ["api.example.com"]}
    assert result.truncated


@pytest.mark.parametrize("certificate_id", [True, 1.2, "-1", "1" * 33, "１２", {"id": 1}])
def test_invalid_optional_certificate_metadata_is_not_invented_or_stringified(
    monkeypatch, certificate_id
):
    certificate = {
        "common_name": "api.example.com",
        "id": certificate_id,
        "issuer_name": {"name": "not a string"},
        "not_before": 123456789,
        "not_after": "not a date",
        "entry_timestamp": ["2020-01-01"],
    }
    monkeypatch.setattr(
        passive,
        "_fetch_bytes",
        lambda url, *args, **kwargs: (
            json.dumps([certificate]).encode() if "crt.sh" in url else b"No records found"
        ),
    )
    result = passive.collect_passive("example.com")
    record = result.evidence["api.example.com"][0]
    assert set(record) == {"source", "query", "matched_name", "collected_at"}
    assert record["matched_name"] == "api.example.com"
    assert not result.errors


def test_issuer_evidence_is_bounded_and_controls_are_not_persisted(monkeypatch):
    monkeypatch.setattr(
        passive,
        "_fetch_bytes",
        lambda url, *args, **kwargs: (
            json.dumps(
                [{"name_value": "api.example.com", "issuer_name": "CA\x00" + "x" * 1000}]
            ).encode()
            if "crt.sh" in url
            else b"No records found"
        ),
    )
    result = passive.collect_passive("example.com")
    assert result.evidence["api.example.com"][0]["issuer"] == "CA " + "x" * 497


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
    assert observed == {"crt.sh": expected}
    assert not result.errors


def test_optional_subfinder_is_collected_when_installed(monkeypatch):
    observed = {}
    monkeypatch.setattr(passive.shutil, "which", lambda name: "fixture-subfinder")

    def worker(source, target, max_results, deadline, replies):
        assert source == "crt.sh"
        replies.put_nowait(passive._SourceResult(source, ["ct.example.com"]))

    def run_subfinder(executable, target, max_results, deadline):
        observed.update(executable=executable, target=target)
        return passive._SourceResult(
            "subfinder",
            ["cli.example.com"],
            evidence={
                "cli.example.com": [
                    {
                        "source": "subfinder",
                        "query": target,
                        "matched_name": "cli.example.com",
                        "collected_at": "2026-01-01T00:00:00+00:00",
                    }
                ]
            },
        )

    monkeypatch.setattr(passive, "_source_worker", worker)
    monkeypatch.setattr(passive, "_run_subfinder", run_subfinder)
    result = passive.collect_passive("example.com")
    assert observed["target"] == "example.com"
    assert observed["executable"].endswith("fixture-subfinder")
    assert result.names == ["cli.example.com", "ct.example.com"]
    assert result.sources == {
        "crt.sh": ["ct.example.com"],
        "subfinder": ["cli.example.com"],
    }
    assert result.evidence["cli.example.com"][0]["source"] == "subfinder"


def test_sources_run_concurrently_and_one_failure_does_not_hide_other(monkeypatch):
    monkeypatch.setenv("MERKLEMAP_API_KEY", "fixture-key")
    both_started = threading.Barrier(2)
    crt_attempts = 0

    def fetch(url, params, *, deadline, **kwargs):
        nonlocal crt_attempts
        if url == "https://crt.sh/":
            crt_attempts += 1
            if crt_attempts == 1:
                both_started.wait(timeout=1)
            raise requests.ConnectionError("https://secret-user:secret-token@proxy.test")
        assert url == "https://api.merklemap.com/v1/search"
        both_started.wait(timeout=1)
        return b'{"count":1,"results":[{"hostname":"www.example.com"}]}'

    monkeypatch.setattr(passive, "_fetch_bytes", fetch)
    result = passive.collect_passive("example.com")
    assert result.names == ["www.example.com"]
    assert result.sources["crt.sh"] == []
    assert result.sources["merklemap"] == ["www.example.com"]
    assert result.errors == ["crt.sh: service connection failed"]
    assert crt_attempts == 3
    assert "secret" not in str(result)


def test_result_cap_is_deterministic_and_limits_provenance(monkeypatch):
    install_sources(
        monkeypatch,
        crt_names=["z.example.com", "a.example.com", "b.example.com"],
    )
    result = passive.collect_passive("example.com", max_results=2)
    assert result.names == ["a.example.com", "b.example.com"]
    assert result.sources["crt.sh"] == result.names
    assert result.truncated
    assert set(result.evidence) == {"a.example.com", "b.example.com"}


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
    monkeypatch.setenv("MERKLEMAP_API_KEY", "fixture-key")
    numeric = [f"worker{index:03d}.example.com" for index in range(20)]
    semantic = [
        "api.example.com",
        "auth.example.com",
        "payments.example.com",
        "sandbox.example.com",
    ]

    def worker(source, target, max_results, deadline, replies, **kwargs):
        values = numeric if source == "crt.sh" else semantic
        replies.put_nowait(passive._SourceResult(source, sorted(values)))

    monkeypatch.setattr(passive, "_source_worker", worker)
    result = passive.collect_passive("example.com", max_results=5)

    assert result.names == sorted([numeric[0], *semantic])
    assert result.sources["crt.sh"] == [numeric[0]]
    assert result.sources["merklemap"] == sorted(semantic)
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
    monkeypatch.setenv("MERKLEMAP_API_KEY", "fixture-key")
    replies = {
        "crt.sh": passive._SourceResult(
            "crt.sh", ["node1.example.com", "node2.example.com", "z.example.com"]
        ),
        "merklemap": passive._SourceResult("merklemap", ["api.example.com", "node1.example.com"]),
    }
    outcomes = []
    for first in ("crt.sh", "merklemap"):
        first_replied = threading.Event()

        def worker(source, target, max_results, deadline, output, **kwargs):
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
    assert "node1.example.com" in outcomes[0].sources["merklemap"]


def test_retained_name_keeps_membership_even_if_another_source_selected_it(monkeypatch):
    monkeypatch.setenv("MERKLEMAP_API_KEY", "fixture-key")

    def worker(source, target, max_results, deadline, replies, **kwargs):
        if source == "crt.sh":
            result = passive._SourceResult(
                source,
                ["a.example.com"],
                truncated=True,
                observed_names={"a.example.com", "b.example.com"},
                evidence={
                    "b.example.com": [
                        {
                            "source": "crt.sh",
                            "query": "%.example.com",
                            "matched_name": "*.B.example.com",
                            "collected_at": "2026-01-01T00:00:00+00:00",
                            "certificate_id": "42",
                        }
                    ],
                },
            )
        else:
            result = passive._SourceResult(source, ["b.example.com"])
        replies.put_nowait(result)

    monkeypatch.setattr(passive, "_source_worker", worker)
    result = passive.collect_passive("example.com", max_results=2)
    assert result.names == ["a.example.com", "b.example.com"]
    assert result.sources["crt.sh"] == result.names
    assert result.sources["merklemap"] == ["b.example.com"]
    assert result.truncated
    assert result.evidence["b.example.com"][0]["certificate_id"] == "42"
    assert result.evidence["b.example.com"][0]["matched_name"] == "*.B.example.com"


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
    assert set(result.evidence) == {"api.example.com"}
    assert result.truncated
    assert "timed out" in result.error


def test_source_errors_are_not_reported_as_a_successful_empty_search(monkeypatch):
    def fetch(url, params, *, deadline):
        assert url == "https://crt.sh/"
        return b"<html>Unavailable</html>"

    monkeypatch.setattr(passive, "_fetch_bytes", fetch)
    result = passive.collect_passive("example.com")
    assert not result.names
    assert len(result.errors) == 1


def test_source_size_limit_is_visible(monkeypatch):
    def fetch(url, params, *, deadline):
        raise passive._ResponseTooLarge()

    monkeypatch.setattr(passive, "_fetch_bytes", fetch)
    result = passive.collect_passive("example.com")
    assert result.truncated
    assert len(result.errors) == 1
    assert all("8 MiB" in error for error in result.errors)


def test_expired_deadline_skips_network(monkeypatch):
    monkeypatch.setattr(passive.requests, "Session", lambda: pytest.fail("unexpected network"))
    result = passive.collect_passive("example.com", deadline=time.monotonic() - 1)
    assert not result.names
    assert len(result.errors) == 1
    assert result.truncated


def test_uninterruptible_lookup_cannot_hold_up_the_caller(monkeypatch):
    release = threading.Event()

    def fetch(url, params, *, deadline):
        release.wait(timeout=2)
        return b"[]"

    monkeypatch.setattr(passive, "_fetch_bytes", fetch)
    start = time.monotonic()
    try:
        result = passive.collect_passive("example.com", timeout=0.04)
        elapsed = time.monotonic() - start
    finally:
        release.set()
    assert elapsed < 0.5
    assert result.truncated
    assert len(result.errors) == 1


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
            result = response() if callable(response) else response
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr(passive.requests, "Session", Session)
    return calls


@pytest.mark.parametrize(
    "failure",
    [FakeResponse(status=404), FakeResponse(status=502), requests.Timeout("private fixture")],
)
def test_crtsh_retry_recovers_names_and_certificate_evidence(monkeypatch, failure):
    now = [100.0]
    success = FakeResponse(b'[{"name_value":"api.example.com","id":123}]')
    responses = iter([failure, success])
    calls = install_session(monkeypatch, lambda: next(responses))
    monkeypatch.setattr(passive.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(passive.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    replies = queue.Queue()

    passive._source_worker("crt.sh", "example.com", 100, 105.0, replies)

    result = replies.get_nowait()
    assert result.names == ["api.example.com"]
    assert result.evidence["api.example.com"][0]["certificate_id"] == "123"
    assert result.error is None
    assert len(calls) == 2
    assert success.closed
    if isinstance(failure, FakeResponse):
        assert failure.closed


def test_crtsh_repeated_transient_failure_is_bounded_and_visible(monkeypatch):
    calls = install_session(monkeypatch, FakeResponse(status=404))
    monkeypatch.setattr(passive.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(passive.time, "sleep", lambda delay: None)
    replies = queue.Queue()

    passive._source_worker("crt.sh", "example.com", 100, 160.0, replies)

    result = replies.get_nowait()
    assert not result.names
    assert "HTTP 404" in result.error
    assert len(calls) == 3


@pytest.mark.parametrize("remaining", [0.1, 1.0])
def test_crtsh_retry_does_not_request_after_shared_deadline(monkeypatch, remaining):
    now = [100.0]
    calls = install_session(monkeypatch, FakeResponse(status=502))
    monkeypatch.setattr(passive.time, "monotonic", lambda: now[0])
    # A scheduler can resume a bounded backoff later than requested.
    monkeypatch.setattr(passive.time, "sleep", lambda delay: now.__setitem__(0, 102.0))
    replies = queue.Queue()

    passive._source_worker("crt.sh", "example.com", 100, 100.0 + remaining, replies)

    result = replies.get_nowait()
    assert "HTTP 502" in result.error
    assert len(calls) == 1


@pytest.mark.parametrize("status", [302, 401, 429])
def test_crtsh_does_not_retry_redirects_authentication_or_rate_limits(monkeypatch, status):
    calls = install_session(monkeypatch, FakeResponse(status=status))
    replies = queue.Queue()

    passive._source_worker("crt.sh", "example.com", 100, time.monotonic() + 5, replies)

    assert replies.get_nowait().error
    assert len(calls) == 1


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


@pytest.mark.parametrize("body", [b"{}", b"null", b"1", b'"example.com"', b"["])
def test_invalid_certificate_response_is_a_failure(monkeypatch, body):
    monkeypatch.setattr(passive, "_fetch_bytes", lambda *args, **kwargs: body)
    result = passive.collect_passive("example.com")
    assert result.names == []
    assert result.sources == {"crt.sh": []}
    assert result.evidence == {}
    assert len(result.errors) == 1


def install_merklemap(monkeypatch, pages):
    monkeypatch.setenv("MERKLEMAP_API_KEY", "fixture-merklemap-secret")
    responses = iter(pages)
    calls = []

    class Session:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            if url == "https://crt.sh/":
                return FakeResponse()
            assert url == "https://api.merklemap.com/v1/search"
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            if callable(response):
                return response()
            if isinstance(response, dict):
                return FakeResponse(json.dumps(response).encode())
            return response

    monkeypatch.setattr(passive.requests, "Session", Session)
    return calls


def test_merklemap_pages_scope_evidence_and_authentication_boundary(monkeypatch):
    calls = install_merklemap(
        monkeypatch,
        [
            {
                "count": 7,
                "results": [
                    {
                        "hostname": "*.API.EXAMPLE.com.",
                        "subject_common_name": "*.example.com",
                        "first_seen": "2025-01-02T05:00:00+02:00",
                    },
                    {"hostname": "example.com"},
                    {"hostname": "otherexample.com"},
                    {"hostname": "www.example.com.attacker.test"},
                ],
            },
            {
                "count": 7,
                "results": [
                    {"hostname": "api.example.com", "first_seen": "not a date"},
                    {"hostname": "dev.example.com", "first_seen": "2025-01-02T03:00:00"},
                    {"hostname": "bücher.example.com"},
                ],
            },
        ],
    )
    result = passive.collect_passive("EXAMPLE.com.")
    assert result.names == ["api.example.com", "dev.example.com", "xn--bcher-kva.example.com"]
    assert result.sources["merklemap"] == result.names
    assert not result.errors and not result.truncated
    assert result.evidence["api.example.com"][0]["first_seen"] == "2025-01-02T03:00:00+00:00"
    assert result.evidence["api.example.com"][0]["subject_common_name"] == "*.example.com"
    assert "first_seen" not in result.evidence["api.example.com"][1]
    assert "first_seen" not in result.evidence["dev.example.com"][0]
    from enumeraite.core.origin_evidence import validate_origin_evidence

    for name, records in result.evidence.items():
        validate_origin_evidence({"passive": records}, name, "example.com")
    api_calls = [kwargs for url, kwargs in calls if "merklemap.com" in url]
    assert [call["params"] for call in api_calls] == [
        {"query": "*.example.com", "type": "wildcard", "page": "0"},
        {"query": "*.example.com", "type": "wildcard", "page": "1"},
    ]
    assert all(
        call["headers"]["Authorization"] == "Bearer fixture-merklemap-secret"
        and not call["allow_redirects"]
        for call in api_calls
    )
    assert all(
        "Authorization" not in call["headers"] for url, call in calls if "merklemap.com" not in url
    )
    assert "fixture-merklemap-secret" not in repr(result)


def test_merklemap_retained_limit_finishes_page_without_spending_on_another(monkeypatch):
    calls = install_merklemap(
        monkeypatch,
        [
            {
                "count": 1000,
                "results": [
                    {"hostname": "node1.example.com"},
                    {"hostname": "node2.example.com"},
                    {"hostname": "node3.example.com"},
                    {"hostname": "z.example.com"},
                ],
            }
        ],
    )
    result = passive.collect_passive("example.com", max_results=2)
    assert result.names == ["node1.example.com", "z.example.com"]
    assert set(result.evidence) == set(result.names)
    assert result.truncated and not result.errors
    assert len([url for url, _ in calls if "merklemap.com" in url]) == 1


@pytest.mark.parametrize(
    "failure",
    [
        FakeResponse(
            b'{"error":"fixture-merklemap-secret"}',
            status=302,
            headers={"Location": "https://attacker.test/"},
        ),
        FakeResponse(b'{"error":"fixture-merklemap-secret"}', status=401),
        FakeResponse(b'{"error":"fixture-merklemap-secret"}', status=429),
        FakeResponse(status=404),
        FakeResponse(status=502),
        requests.ConnectionError("Authorization: Bearer fixture-merklemap-secret"),
        requests.Timeout("Authorization: Bearer fixture-merklemap-secret"),
        FakeResponse(b"not-json fixture-merklemap-secret"),
        {"count": True, "results": []},
        {"count": 3, "results": [{"hostname": None}]},
        {"count": 3, "results": []},
    ],
)
def test_merklemap_later_page_failure_preserves_observations_without_secrets(monkeypatch, failure):
    calls = install_merklemap(
        monkeypatch,
        [
            {"count": 3, "results": [{"hostname": "known.example.com"}]},
            failure,
            {"count": 3, "results": [{"hostname": "should-not-retry.example.com"}]},
        ],
    )
    result = passive.collect_passive("example.com")
    assert result.names == ["known.example.com"]
    assert result.evidence["known.example.com"][0]["source"] == "merklemap"
    assert result.truncated and result.errors
    assert "fixture-merklemap-secret" not in repr(result)
    assert len([url for url, _ in calls if "merklemap.com" in url]) == 2


def test_merklemap_completed_page_survives_a_stalled_followup(monkeypatch):
    release = threading.Event()
    entered = threading.Event()

    def stalled_page():
        entered.set()
        release.wait(2)
        return FakeResponse(b'{"count":2,"results":[{"hostname":"late.example.com"}]}')

    install_merklemap(
        monkeypatch,
        [{"count": 2, "results": [{"hostname": "known.example.com"}]}, stalled_page],
    )
    started = time.monotonic()
    try:
        result = passive.collect_passive("example.com", timeout=0.15)
        assert entered.is_set()
        assert time.monotonic() - started < 1
        assert result.names == ["known.example.com"]
        assert result.truncated and result.errors
        assert result.evidence["known.example.com"][0]["source"] == "merklemap"
    finally:
        release.set()


def test_merklemap_repeated_pages_stop_paid_requests(monkeypatch):
    page = {"count": 1000, "results": [{"hostname": "known.example.com"}]}
    calls = install_merklemap(monkeypatch, [page, page])
    result = passive.collect_passive("example.com")
    assert result.names == ["known.example.com"]
    assert result.truncated and result.errors
    assert len([url for url, _ in calls if "merklemap.com" in url]) == 2


def test_merklemap_response_limit_is_shared_across_pages(monkeypatch):
    pages = [
        {"count": 3, "results": [{"hostname": "first.example.com"}]},
        {"count": 3, "results": [{"hostname": "second.example.com"}]},
    ]
    monkeypatch.setattr(passive, "_MAX_RESPONSE_BYTES", len(json.dumps(pages[0]).encode()) + 1)
    install_merklemap(monkeypatch, pages)
    result = passive.collect_passive("example.com")
    assert result.names == ["first.example.com"]
    assert result.truncated and result.errors


def test_merklemap_page_limit_bounds_nonproductive_searches(monkeypatch):
    monkeypatch.setattr(passive, "_MAX_MERKLEMAP_PAGES", 2)
    calls = install_merklemap(
        monkeypatch,
        [
            {"count": 1000, "results": [{"hostname": "first.outside.test"}]},
            {"count": 1000, "results": [{"hostname": "second.outside.test"}]},
        ],
    )
    result = passive.collect_passive("example.com")
    assert not result.names
    assert result.truncated and result.errors
    assert len([url for url, _ in calls if "merklemap.com" in url]) == 2


def test_blank_merklemap_key_does_not_opt_into_paid_requests(monkeypatch):
    install_sources(monkeypatch, crt_names=["public.example.com"])
    monkeypatch.setenv("MERKLEMAP_API_KEY", " \t")
    result = passive.collect_passive("example.com")
    assert result.names == ["public.example.com"]
    assert "merklemap" not in result.sources


def test_merklemap_file_key_enables_collection_without_ai_config_or_env_mutation(
    monkeypatch, tmp_path
):
    calls = install_merklemap(
        monkeypatch, [{"count": 1, "results": [{"hostname": "api.example.com"}]}]
    )
    monkeypatch.delenv("MERKLEMAP_API_KEY")
    monkeypatch.setenv("ENUMERAITE_PROVIDER_TIMEOUT", "invalid-unrelated-AI-setting")
    path = tmp_path / ".config" / "enumeraite" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"MERKLEMAP_API_KEY": "fixture-file-key"}), encoding="utf-8")

    result = passive.collect_passive("example.com")
    assert result.names == ["api.example.com"]
    assert result.sources["merklemap"] == result.names
    assert not result.errors
    assert all(
        call["headers"]["Authorization"] == "Bearer fixture-file-key"
        for url, call in calls
        if "merklemap.com" in url
    )
    assert "MERKLEMAP_API_KEY" not in passive.os.environ
    assert "fixture-file-key" not in repr(result)
