"""Jev protocol tests use a fake HTTP transport and never contact a service."""

import copy
import json
import time
from types import SimpleNamespace

import pytest
import requests

from enumeraite.core.scan_budget import BudgetExhausted, ModelPrice, ScanPaused, SpendLedger
from enumeraite.providers import jev
from enumeraite.providers.jev import JEV_MODEL, JEV_PRICE, JevPredictionError, JevPredictor

KEY = "fixture-jev-key-not-a-real-secret"
PLANS = [
    {
        "id": "numeric.family-1",
        "description": "Check two observed-slot numeric gaps",
        "evidence": ["ny-01.example.test", "ny-03.example.test"],
        "feedback": {"hits": 2, "misses": 0, "inconclusive": 1},
    },
    {
        "id": "child.family-1",
        "description": "Try a small sample of child names",
        "feedback": {"hits": 0, "misses": 4},
    },
]
STATE = {"target": "example.test", "remaining_candidates": 20}


def response_document(**overrides):
    return {
        "model": JEV_MODEL,
        "answers": {
            PLANS[0]["id"]: {"type": "noul", "noul": 0.8},
            PLANS[1]["id"]: {"type": "noul", "noul": 0.1},
        },
        "usage": {"input_tokens": 300, "output_tokens": 20},
        **overrides,
    }


class Response:
    def __init__(self, document=None, *, status=200, body=None, headers=None):
        self.status_code = status
        self.body = (
            json.dumps(response_document() if document is None else document).encode()
            if body is None
            else body
        )
        self.headers = headers or {}
        self.closed = False
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def iter_content(self, *, chunk_size):
        for offset in range(0, len(self.body), chunk_size):
            self.read_count += 1
            yield self.body[offset : offset + chunk_size]


@pytest.fixture
def transport(monkeypatch):
    state = SimpleNamespace(calls=[], sessions=[], response=Response(), error=None, on_post=None)

    class Session:
        def __init__(self):
            self.trust_env = True
            self.headers = {}
            self.adapters = {}
            self.closed = False
            state.sessions.append(self)

        def mount(self, prefix, adapter):
            self.adapters[prefix] = adapter

        def post(self, url, **kwargs):
            state.calls.append((url, kwargs, dict(self.headers)))
            if state.on_post:
                state.on_post()
            if state.error is not None:
                raise state.error
            return state.response

        def close(self):
            self.closed = True

    monkeypatch.setattr(jev.requests, "Session", Session)
    return state


def score(predictor, ledger, snapshots=None, **kwargs):
    return predictor.score_plans(
        kwargs.pop("state", STATE),
        kwargs.pop("plans", copy.deepcopy(PLANS)),
        ledger=ledger,
        deadline=kwargs.pop("deadline", time.monotonic() + 20),
        checkpoint=kwargs.pop(
            "checkpoint",
            lambda: (
                snapshots.append(copy.deepcopy(ledger.to_dict())) if snapshots is not None else None
            ),
        ),
        stop_requested=kwargs.pop("stop_requested", lambda: False),
        **kwargs,
    )


def test_protocol_uses_pinned_model_typed_questions_and_official_input_only_pricing(transport):
    predictor = JevPredictor(KEY)
    ledger = SpendLedger(1, JEV_PRICE)
    snapshots = []
    transport.on_post = lambda: snapshots.append({"wire": ledger.reserved_usd})
    scores = score(predictor, ledger, snapshots)
    assert scores == {PLANS[0]["id"]: 0.8, PLANS[1]["id"]: 0.1}
    assert predictor.model == JEV_MODEL == "jev-1.13.0"
    assert JEV_PRICE.input_usd_per_million == 0.042
    assert JEV_PRICE.output_usd_per_million == 0
    assert JEV_PRICE.checked_at == "2026-10-03"
    assert JEV_PRICE.source == "https://docs.typesafe.ai/models"
    assert len(transport.calls) == 1
    url, options, headers = transport.calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert headers["Authorization"] == "Bearer " + KEY
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept-Encoding"] == "identity"
    assert options["allow_redirects"] is False and options["stream"] is True
    assert all(0 < timeout <= 20 for timeout in options["timeout"])
    payload = json.loads(options["data"])
    assert payload["model"] == JEV_MODEL and payload["state"] == STATE
    assert set(payload["questions"]) == {plan["id"] for plan in PLANS}
    for plan in PLANS:
        question = payload["questions"][plan["id"]]
        assert question["type"] == "noul"
        assert question["instructions"]["candidate_plan"] == plan
        assert set(question["criteria"]) == {"true", "false"}
    assert KEY.encode() not in options["data"]
    session = transport.sessions[0]
    assert session.trust_env is False
    assert session.adapters["https://"].max_retries.total == 0
    reservation = 65_536 * 0.042 / 1_000_000
    assert float(snapshots[0]["reserved_usd"]) == pytest.approx(reservation)
    assert snapshots[1]["wire"] == pytest.approx(reservation)
    assert ledger.spent_usd == pytest.approx(300 * 0.042 / 1_000_000)
    assert ledger.reserved_usd == 0 and ledger.calls == 1 and ledger.uncertain_calls == 0
    assert transport.response.closed
    assert KEY not in json.dumps(snapshots)
    predictor.close()
    assert session.closed and "Authorization" not in session.headers


@pytest.mark.parametrize("key", [None, "", "short", "a" * 2049, "a" * 20 + "\n", "é" * 20])
def test_invalid_key_is_rejected_without_transport_creation_or_echo(transport, key):
    with pytest.raises(ValueError, match="valid printable API key"):
        JevPredictor(key)
    assert not transport.sessions


def test_only_supported_pinned_model_allowed_and_constructor_never_sends(transport):
    with pytest.raises(ValueError, match="pinned model"):
        JevPredictor(KEY, model="jev-latest")
    predictor = JevPredictor(KEY)
    assert not transport.calls
    predictor.close()
    with pytest.raises(JevPredictionError, match="closed"):
        score(predictor, SpendLedger(1, JEV_PRICE))
    assert not transport.calls


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.update(model="jev-latest"),
        lambda document: document.update(answers={}),
        lambda document: document["answers"].pop(PLANS[0]["id"]),
        lambda document: document["answers"].update(extra={"type": "noul", "noul": 0.5}),
        lambda document: document["answers"][PLANS[0]["id"]].update(type="score"),
        lambda document: document["answers"][PLANS[0]["id"]].update(noul=True),
        lambda document: document["answers"][PLANS[0]["id"]].update(noul="0.5"),
        lambda document: document["answers"][PLANS[0]["id"]].update(noul=-0.1),
        lambda document: document["answers"][PLANS[0]["id"]].update(noul=1.1),
        lambda document: document["answers"][PLANS[0]["id"]].update(noul=float("nan")),
        lambda document: document["answers"].update({PLANS[0]["id"]: []}),
        lambda document: document.update(usage=None),
        lambda document: document["usage"].update(input_tokens=0),
        lambda document: document["usage"].update(input_tokens=True),
        lambda document: document["usage"].update(output_tokens=-1),
        lambda document: document["usage"].pop("output_tokens"),
    ],
)
def test_malformed_model_scores_or_usage_fail_closed_and_charge_full_reservation(
    transport, mutation
):
    document = response_document()
    mutation(document)
    document["private"] = KEY
    transport.response = Response(document)
    ledger = SpendLedger(1, JEV_PRICE)
    snapshots = []
    with pytest.raises(JevPredictionError) as caught:
        score(JevPredictor(KEY), ledger, snapshots)
    assert KEY not in str(caught.value)
    assert ledger.spent_usd == pytest.approx(65_536 * 0.042 / 1_000_000)
    assert ledger.reserved_usd == 0 and ledger.uncertain_calls == 1
    assert float(snapshots[-1]["spent_usd"]) == ledger.spent_usd
    assert len(transport.calls) == 1


@pytest.mark.parametrize("body", [b"not json", b"[]", b"\xff", b'{"model":"first","model":"last"}'])
def test_invalid_json_and_duplicate_response_fields_are_generic(transport, body):
    transport.response = Response(body=body)
    ledger = SpendLedger(1, JEV_PRICE)
    with pytest.raises(JevPredictionError, match="no response details retained"):
        score(JevPredictor(KEY), ledger)
    assert ledger.uncertain_calls == 1


@pytest.mark.parametrize("status", [301, 302, 307, 401, 422, 429, 500, 529])
def test_http_errors_or_redirects_are_never_retried_or_leaked(transport, status, capsys):
    transport.response = Response(
        status=status, body=KEY.encode(), headers={"Location": "https://elsewhere.test/" + KEY}
    )
    ledger = SpendLedger(1, JEV_PRICE)
    with pytest.raises(JevPredictionError) as caught:
        score(JevPredictor(KEY), ledger)
    assert KEY not in str(caught.value)
    assert len(transport.calls) == 1
    assert transport.response.read_count == 0
    assert ledger.calls == ledger.uncertain_calls == 1
    captured = capsys.readouterr()
    assert KEY not in captured.out + captured.err


@pytest.mark.parametrize(
    "error",
    [requests.Timeout(KEY), requests.ConnectionError(KEY), RuntimeError(KEY), KeyboardInterrupt()],
)
def test_uncertain_transport_outcome_is_charged_and_persisted(transport, error):
    transport.error = error
    ledger = SpendLedger(1, JEV_PRICE)
    snapshots = []
    expected = KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else JevPredictionError
    with pytest.raises(expected) as caught:
        score(JevPredictor(KEY), ledger, snapshots)
    assert KEY not in str(caught.value)
    assert ledger.calls == ledger.uncertain_calls == 1
    assert float(snapshots[-1]["reserved_usd"]) == 0
    assert float(snapshots[-1]["spent_usd"]) > 0


def test_zero_output_tokens_are_valid_when_output_is_free(transport):
    transport.response = Response(
        response_document(usage={"input_tokens": 100, "output_tokens": 0})
    )
    ledger = SpendLedger(1, JEV_PRICE)
    assert score(JevPredictor(KEY), ledger)
    assert ledger.uncertain_calls == 0
    assert ledger.spent_usd == pytest.approx(100 * 0.042 / 1_000_000)


def test_reported_usage_beyond_reservation_records_actual_charge_and_blocks_future_calls(transport):
    transport.response = Response(
        response_document(usage={"input_tokens": 70_000, "output_tokens": 10})
    )
    ledger = SpendLedger(1, JEV_PRICE)
    snapshots = []
    predictor = JevPredictor(KEY)
    with pytest.raises(BudgetExhausted, match="exceeded"):
        score(predictor, ledger, snapshots)
    assert ledger.blocked and ledger.spent_usd == pytest.approx(70_000 * 0.042 / 1_000_000)
    assert snapshots[-1]["blocked"] is True
    with pytest.raises(BudgetExhausted):
        score(predictor, ledger)
    assert len(transport.calls) == 1


def test_too_small_budget_prevents_wire_and_durable_reservation(transport):
    ledger = SpendLedger(0.001, JEV_PRICE)
    snapshots = []
    with pytest.raises(BudgetExhausted):
        score(JevPredictor(KEY), ledger, snapshots)
    assert not snapshots and not transport.calls
    assert ledger.calls == ledger.reserved_usd == ledger.spent_usd == 0


def test_checkpoint_failure_cancels_untransmitted_reservation_without_leaking_error(transport):
    ledger = SpendLedger(1, JEV_PRICE)

    def checkpoint():
        raise OSError(KEY)

    with pytest.raises(JevPredictionError, match="Could not save") as caught:
        score(JevPredictor(KEY), ledger, checkpoint=checkpoint)
    assert KEY not in str(caught.value)
    assert not transport.calls
    assert ledger.calls == ledger.reserved_usd == ledger.spent_usd == 0


def test_pause_before_wire_never_reserves_or_sends(transport):
    ledger = SpendLedger(1, JEV_PRICE)
    with pytest.raises(ScanPaused):
        score(JevPredictor(KEY), ledger, stop_requested=lambda: True)
    assert ledger.calls == ledger.reserved_usd == 0
    assert not transport.calls


def test_pause_after_reservation_checkpoint_releases_it_durably(transport):
    ledger = SpendLedger(1, JEV_PRICE)
    snapshots = []
    with pytest.raises(ScanPaused):
        score(JevPredictor(KEY), ledger, snapshots, stop_requested=lambda: bool(snapshots))
    assert not transport.calls
    assert ledger.calls == ledger.reserved_usd == 0
    assert float(snapshots[0]["reserved_usd"]) > 0
    assert float(snapshots[-1]["reserved_usd"]) == 0
    assert SpendLedger.from_dict(snapshots[-1]).spent_usd == 0


def test_open_ended_scan_preserves_request_timeouts_and_pause_budget_guards(transport):
    ledger = SpendLedger(1, JEV_PRICE)
    predictor = JevPredictor(KEY)
    scores = score(predictor, ledger, deadline=None, state={**STATE, "remaining_candidates": None})
    assert scores == {PLANS[0]["id"]: 0.8, PLANS[1]["id"]: 0.1}
    assert transport.calls[0][1]["timeout"] == (3.0, 5.0)
    assert ledger.spent_usd == pytest.approx(300 * 0.042 / 1_000_000)
    assert ledger.reserved_usd == 0
    with pytest.raises(ScanPaused):
        score(predictor, ledger, deadline=None, stop_requested=lambda: True)
    with pytest.raises(BudgetExhausted):
        score(predictor, SpendLedger(0, JEV_PRICE), deadline=None)
    assert len(transport.calls) == 1


def test_deadline_before_wire_or_during_checkpoint_prevents_paid_request(transport, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(jev.time, "monotonic", lambda: now[0])
    ledger = SpendLedger(1, JEV_PRICE)
    predictor = JevPredictor(KEY)
    with pytest.raises(TimeoutError):
        score(predictor, ledger, deadline=99)
    snapshots = []

    def checkpoint():
        snapshots.append(ledger.to_dict())
        now[0] = 102

    with pytest.raises(TimeoutError):
        score(predictor, ledger, deadline=101, checkpoint=checkpoint)
    assert not transport.calls
    assert ledger.calls == ledger.reserved_usd == ledger.spent_usd == 0
    assert float(snapshots[-1]["reserved_usd"]) == 0


def test_deadline_bounds_socket_timeouts_and_late_body_is_charged_uncertain(transport, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(jev.time, "monotonic", lambda: now[0])
    transport.on_post = lambda: now.__setitem__(0, 101.0)
    ledger = SpendLedger(1, JEV_PRICE)
    with pytest.raises(TimeoutError):
        score(JevPredictor(KEY), ledger, deadline=100.5)
    assert transport.calls[0][1]["timeout"] == (0.5, 0.5)
    assert ledger.calls == ledger.uncertain_calls == 1
    assert transport.response.read_count == 0


def test_real_transport_read1_shape_uses_bounded_chunks_without_decompression(transport):
    body = transport.response.body
    reads = []

    def read1(amount, *, decode_content):
        reads.append((amount, decode_content))
        return body if len(reads) == 1 else b""

    transport.response.raw = SimpleNamespace(read1=read1)
    ledger = SpendLedger(1, JEV_PRICE)
    assert score(JevPredictor(KEY), ledger)
    assert reads == [(4096, False), (4096, False)]
    assert transport.response.read_count == 0
    assert ledger.uncertain_calls == 0


def test_pause_during_request_charges_uncertain_outcome_and_saves_before_return(transport):
    paused = [False]
    transport.on_post = lambda: paused.__setitem__(0, True)
    ledger = SpendLedger(1, JEV_PRICE)
    snapshots = []
    with pytest.raises(ScanPaused):
        score(JevPredictor(KEY), ledger, snapshots, stop_requested=lambda: paused[0])
    assert len(transport.calls) == 1 and transport.response.closed
    assert ledger.calls == ledger.uncertain_calls == 1
    assert ledger.reserved_usd == 0
    assert float(snapshots[-1]["spent_usd"]) == ledger.spent_usd


def test_failed_final_checkpoint_keeps_settled_charge_without_exposing_error(transport):
    ledger = SpendLedger(1, JEV_PRICE)
    saves = []

    def checkpoint():
        saves.append(ledger.to_dict())
        if len(saves) > 1:
            raise OSError(KEY)

    with pytest.raises(JevPredictionError, match="Could not save") as caught:
        score(JevPredictor(KEY), ledger, checkpoint=checkpoint)
    assert KEY not in str(caught.value)
    assert len(transport.calls) == 1
    assert ledger.spent_usd == pytest.approx(300 * 0.042 / 1_000_000)
    assert ledger.reserved_usd == 0
    # A failed final save leaves the earlier on-disk reservation conservative.
    recovered = SpendLedger.from_dict(saves[0])
    assert recovered.spent_usd >= ledger.spent_usd


@pytest.mark.parametrize(
    "headers,body",
    [
        ({"Content-Length": "70000"}, b""),
        ({"Content-Length": "invalid"}, b""),
        ({"Content-Encoding": "gzip"}, b"compressed"),
        ({}, b" " * 70_000),
    ],
)
def test_response_size_and_encoding_caps_fail_closed(transport, headers, body):
    transport.response = Response(headers=headers, body=body)
    ledger = SpendLedger(1, JEV_PRICE)
    with pytest.raises(JevPredictionError):
        score(JevPredictor(KEY), ledger)
    assert ledger.uncertain_calls == 1
    assert transport.response.closed


@pytest.mark.parametrize(
    "state,plans",
    [
        ([], PLANS),
        ({"long": "x" * 20_000}, PLANS),
        ({"nan": float("nan")}, PLANS),
        ({"custom": object()}, PLANS),
        (STATE, {}),
        (STATE, PLANS * 13),
        (STATE, [PLANS[0], PLANS[0]]),
        (STATE, [{"id": "valid", "description": ""}]),
        (STATE, [{"id": "bad id", "description": "test"}]),
        (STATE, [{"id": "valid", "description": "x" * 3000}]),
    ],
)
def test_invalid_or_oversized_input_is_rejected_without_reserving(transport, state, plans):
    ledger = SpendLedger(1, JEV_PRICE)
    with pytest.raises(ValueError):
        score(JevPredictor(KEY), ledger, state=state, plans=plans)
    assert not transport.calls
    assert ledger.calls == ledger.reserved_usd == 0


def test_whole_request_cap_and_circular_input_are_bounded(transport):
    ledger = SpendLedger(1, JEV_PRICE)
    plans = [{"id": str(index), "description": "x" * 1500} for index in range(24)]
    predictor = JevPredictor(KEY)
    with pytest.raises(ValueError, match="size limit"):
        score(predictor, ledger, plans=plans, state={"data": "x" * 12_000})
    circular = {}
    circular["self"] = circular
    with pytest.raises(ValueError, match="structure limit"):
        score(predictor, ledger, state=circular)
    assert not transport.calls and ledger.calls == 0


def test_empty_plan_list_is_noop_without_wire_or_reservation(transport):
    ledger = SpendLedger(1, JEV_PRICE)
    assert score(JevPredictor(KEY), ledger, plans=[]) == {}
    assert not transport.calls and ledger.calls == 0


def test_wrong_rate_cannot_underprice_jev_calls(transport):
    with pytest.raises(ValueError, match="input-only pricing"):
        score(JevPredictor(KEY), SpendLedger(1, ModelPrice(0, 0)))
    assert not transport.calls
