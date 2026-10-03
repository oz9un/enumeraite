import copy

import pytest

from enumeraite.core.discovery_session import DiscoverySession, ScanOptions, validate_checkpoint
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.passive import PassiveResult


class FakeDNS:
    def __init__(self, hits=(), errors=(), wildcard=()):
        self.hits = set(hits)
        self.errors = set(errors)
        self.wildcard = set(wildcard)
        self.calls = []

    def validate(self, values, *, deadline):
        self.calls.extend(values)
        return [
            DNSResult(
                value,
                value in self.hits,
                ["192.0.2.1"] if value in self.hits else [],
                0.0,
                error="timeout" if value in self.errors else None,
                wildcard=value in self.wildcard,
            )
            for value in values
        ]


def passive(names=(), errors=(), truncated=False):
    return lambda *args, **kwargs: PassiveResult(
        list(names), {"fixture": list(names)}, list(errors), truncated
    )


def make_session(*, hits=(), names=(), options=None, **kwargs):
    return DiscoverySession(
        "example.test",
        options or ScanOptions(max_candidates=200),
        FakeDNS(hits),
        passive_fetch=passive(names),
        **kwargs,
    )


def test_known_first_then_recursive_numeric_middle_and_children():
    known = "ny-01-iphone.example.test"
    second = "ny-02-iphone.example.test"
    third = "ny-03-iphone.example.test"
    child = "api.ny-02-iphone.example.test"
    session = make_session(
        hits=[known, second, third, child],
        names=[known],
        options=ScanOptions(max_candidates=300, max_per_host=8, max_number=3),
    )
    report = session.run()
    assert set(report["findings"]) == {known, second, third, child}
    assert session.validator.calls[0] == known
    assert session.nodes[second]["parent"] == known
    assert session.nodes[child]["parent"] == second
    assert session.nodes[known]["source"] == "passive"
    assert len(session.validator.calls) == len(set(session.validator.calls))
    assert report["known_verified"] == 1
    assert any("number" in entry["pattern"] for entry in session.report()["patterns"])


def test_find_counts_verified_not_guesses_and_cannot_overshoot():
    names = ["a.example.test", "b.example.test", "c.example.test", "d.example.test"]
    session = make_session(names=names, hits=names[1:], options=ScanOptions(find=2, batch_size=40))
    report = session.run()
    assert report["findings"] == names[1:3]
    assert report["attempted"] == 3
    assert report["stop_reason"] == "verified target reached"


def test_seedless_free_bootstrap_works_when_all_passive_sources_fail():
    session = DiscoverySession(
        "example.test",
        ScanOptions(max_candidates=30),
        FakeDNS(["www.example.test"]),
        passive_fetch=passive(errors=["CT unavailable", "hostsearch unavailable"]),
    )
    result = session.run()
    assert "www.example.test" in result["findings"]
    assert len(result["warnings"]) == 2
    assert result["ledger"] is None


def test_scope_and_wildcards_do_not_become_evidence():
    session = DiscoverySession(
        "example.test",
        ScanOptions(max_candidates=20),
        FakeDNS(["wild.example.test"], wildcard=["wild.example.test"]),
        passive_fetch=passive(["wild.example.test", "example.test.evil.test"]),
    )
    result = session.run(["outside.test", "example.test", "../bad"])
    assert not result["findings"]
    assert all(name.endswith(".example.test") for name in session.validator.calls)
    assert session.nodes["wild.example.test"]["status"] == "wildcard"


def test_candidate_and_round_limits_are_global():
    session = make_session(
        names=["api.example.test"], hits=["api.example.test"], options=ScanOptions(max_candidates=5)
    )
    result = session.run()
    assert result["attempted"] == len(session.validator.calls) == 5
    assert result["stop_reason"] == "candidate limit reached"
    session = make_session(
        names=["api.example.test"],
        hits=["api.example.test"],
        options=ScanOptions(max_rounds=1, max_candidates=100),
    )
    assert session.run()["round"] == 1


def test_explicit_seeds_have_priority_with_small_candidate_cap():
    session = make_session(
        names=["passive.example.test"],
        hits=["manual.example.test"],
        options=ScanOptions(max_candidates=1),
    )
    assert session.run(["manual.example.test"])["findings"] == ["manual.example.test"]


def test_seeded_mode_learns_before_bootstrap_can_exhaust_small_candidate_cap():
    first, second = "ny-01-iphone.example.test", "ny-02-iphone.example.test"
    session = make_session(
        hits=[first, second],
        options=ScanOptions(passive=False, max_candidates=2, max_number=2, find=2),
    )
    result = session.run([first])
    assert result["findings"] == [first, second]
    assert session.validator.calls == [first, second]
    assert session.nodes[second]["parent"] == first
    assert all(node["source"] != "bootstrap" for node in session.nodes.values())


def test_failed_seed_does_not_silently_switch_to_unrelated_bootstrap_enumeration():
    session = make_session(options=ScanOptions(passive=False, max_candidates=50))
    result = session.run(["missing.example.test"])
    assert result["findings"] == []
    assert session.validator.calls == ["missing.example.test"]
    assert result["stop_reason"] == "frontier exhausted"


def test_failed_and_missing_dns_results_stay_inconclusive():
    class MissingDNS:
        def validate(self, values, *, deadline):
            return []

    session = DiscoverySession(
        "example.test",
        ScanOptions(max_candidates=1),
        MissingDNS(),
        passive_fetch=passive(["api.example.test"]),
    )
    assert not session.run()["findings"]
    assert session.nodes["api.example.test"]["status"] == "inconclusive"


def test_resume_does_not_repeat_completed_dns_or_passive_calls():
    session = make_session(
        names=["api.example.test", "cdn.example.test"],
        hits=["api.example.test", "cdn.example.test"],
        options=ScanOptions(find=1),
    )
    session.run()
    state = copy.deepcopy(session.state)
    second = DiscoverySession(
        "example.test",
        ScanOptions(find=2),
        FakeDNS(["cdn.example.test"]),
        state=state,
        passive_fetch=lambda *a, **k: pytest.fail("passive repeated"),
    )
    result = second.run()
    assert result["findings"] == ["api.example.test", "cdn.example.test"]
    assert "api.example.test" not in second.validator.calls
    assert result["attempted"] == 2


def test_resume_retries_inconclusive_once_without_recounting_unique_attempt():
    session = DiscoverySession(
        "example.test",
        ScanOptions(max_candidates=1),
        FakeDNS(errors=["api.example.test"]),
        passive_fetch=passive(["api.example.test"]),
    )
    session.run()
    second = DiscoverySession(
        "example.test",
        ScanOptions(max_candidates=1, find=1),
        FakeDNS(["api.example.test"]),
        state=copy.deepcopy(session.state),
    )
    result = second.run()
    assert result["findings"] == ["api.example.test"]
    assert result["attempted"] == 1
    assert second.nodes["api.example.test"]["attempts"] == 2


def test_empty_frontier_page_does_not_hide_later_productive_parent(monkeypatch):
    from enumeraite.core import discovery_session
    from enumeraite.core.subdomain_patterns import PatternProposal

    session = make_session(options=ScanOptions(max_candidates=202, max_depth=1))
    for index in range(201):
        value = f"host{index:03}.example.test"
        session._enqueue(value, source="seed")
        session.nodes[value].update(status="verified", attempts=1)
    session.state["attempted"] = 201

    def expand(host, *args, **kwargs):
        return (
            [PatternProposal("new.example.test", host, "test", "test", 0.8)]
            if host.startswith("host200")
            else []
        )

    monkeypatch.setattr(discovery_session, "expand_host", expand)
    assert session._expand()
    assert session.nodes["new.example.test"]["parent"] == "host200.example.test"


def test_interrupted_expansion_stays_eligible_on_resume(monkeypatch):
    from enumeraite.core import discovery_session
    from enumeraite.core.subdomain_patterns import PatternProposal

    now = [0.0]
    session = make_session(options=ScanOptions(time_limit=10), clock=lambda: now[0])
    session._enqueue("api.example.test", source="seed")
    session.nodes["api.example.test"].update(status="verified", attempts=1)
    session.state["attempted"] = 1

    def expand(host, *args, **kwargs):
        now[0] = 11.0
        return [PatternProposal("api2.example.test", host, "test", "test", 0.8)]

    monkeypatch.setattr(discovery_session, "expand_host", expand)
    assert not session._expand()
    assert session.nodes["api.example.test"]["expanded_revision"] == 0


def test_partial_expansion_at_capacity_can_continue_after_cap_increase():
    session = make_session(options=ScanOptions(max_candidates=2, max_per_host=8))
    session._enqueue("ny-01-iphone.example.test", source="seed")
    session.nodes["ny-01-iphone.example.test"].update(status="verified", attempts=1)
    session.state["attempted"] = 1
    assert session._expand()
    assert len(session.nodes) == 2
    assert session.nodes["ny-01-iphone.example.test"]["expanded_revision"] == 0
    session.options.max_candidates = 10
    assert session._expand()
    assert len(session.nodes) > 2


@pytest.mark.parametrize("stale_rounds", [1, 2])
def test_resume_with_larger_number_range_revisits_evidence_without_repeating_dns(stale_rounds):
    known = "ny-01-iphone.example.test"
    new = "ny-02-iphone.example.test"
    first = make_session(
        hits=[known],
        options=ScanOptions(
            passive=False, max_number=1, max_depth=1, max_candidates=100, stale_rounds=stale_rounds
        ),
    )
    first.run([known])
    assert new not in first.nodes
    # A full candidate page remains eligible now; simulate an exhausted parent
    # to check that a larger numeric range invalidates its saved revision too.
    first.nodes[known]["expanded_revision"] = 1
    previous_round = first.state["round"]
    previous_attempts = first.state["attempted"]
    prior_names = set(first.validator.calls)
    second = DiscoverySession(
        "example.test",
        ScanOptions(
            passive=False,
            max_number=99,
            max_depth=1,
            max_candidates=100,
            find=2,
            stale_rounds=stale_rounds,
        ),
        FakeDNS([new]),
        state=copy.deepcopy(first.state),
    )
    assert second.state["round"] == previous_round
    assert second.state["attempted"] == previous_attempts
    assert second.state["stale"] == 0
    assert second.nodes[known]["expanded_revision"] == 0

    result = second.run()
    assert result["findings"] == [known, new]
    assert not prior_names.intersection(second.validator.calls)
    assert result["round"] > previous_round
    validate_checkpoint(second.state, "example.test")


def test_reduced_dns_label_depth_preserves_pending_name_for_later_resume():
    known = "api.example.test"
    deep = "deep.api.example.test"
    first = make_session(
        hits=[known],
        options=ScanOptions(passive=False, max_candidates=2, max_depth=4, find=1),
    )
    first.run([known, deep])

    class NoDNS:
        def validate(self, values, *, deadline):
            pytest.fail("excluded names must not cause even an empty DNS batch")

    reduced = DiscoverySession(
        "example.test",
        ScanOptions(passive=False, max_candidates=2, max_depth=1, find=2),
        NoDNS(),
        state=copy.deepcopy(first.state),
    )
    assert reduced._validate_batch([deep]) == 0
    result = reduced.run()
    assert result["findings"] == [known]
    assert result["stop_reason"] == "remaining queued work is outside current limits"
    assert reduced.nodes[deep]["status"] == "pending"
    assert reduced.nodes[deep]["attempts"] == 0
    validate_checkpoint(reduced.state, "example.test")

    expanded = DiscoverySession(
        "example.test",
        ScanOptions(passive=False, max_candidates=2, max_depth=2, find=2),
        FakeDNS([deep]),
        state=copy.deepcopy(reduced.state),
    )
    assert expanded.run()["findings"] == [known, deep]
    assert expanded.validator.calls == [deep]


def test_reduced_graph_depth_preserves_shallow_dns_name_with_deep_provenance():
    first = make_session(options=ScanOptions(max_candidates=3, max_depth=3, max_rounds=1))
    names = ["api.example.test", "api2.example.test", "api3.example.test"]
    for depth, value in enumerate(names):
        first._enqueue(
            value,
            source="pattern" if depth else "seed",
            parent=names[depth - 1] if depth else None,
            depth=depth,
        )
        if depth < 2:
            first.nodes[value].update(status="verified", attempts=1)
    first.state.update(initialized=True, attempted=2, round=1)
    reduced = DiscoverySession(
        "example.test",
        ScanOptions(max_candidates=3, max_depth=1, max_rounds=1),
        FakeDNS([names[2]]),
        state=copy.deepcopy(first.state),
    )
    reduced.run()
    assert reduced.validator.calls == []
    assert reduced.nodes[names[2]]["status"] == "pending"
    expanded = DiscoverySession(
        "example.test",
        ScanOptions(max_candidates=3, max_depth=3, max_rounds=1),
        FakeDNS([names[2]]),
        state=copy.deepcopy(reduced.state),
    )
    assert expanded.run()["findings"] == names
    assert expanded.validator.calls == [names[2]]


def test_excluded_retry_at_candidate_cap_does_not_loop_and_can_resume_later():
    deep = "deep.api.example.test"
    first = DiscoverySession(
        "example.test",
        ScanOptions(passive=False, max_candidates=1, max_depth=2),
        FakeDNS(errors=[deep]),
    )
    first.run([deep])
    assert first.nodes[deep]["status"] == "inconclusive"
    reduced = DiscoverySession(
        "example.test",
        ScanOptions(passive=False, max_candidates=1, max_depth=1),
        FakeDNS([deep]),
        state=copy.deepcopy(first.state),
    )
    result = reduced.run()
    assert result["stop_reason"] == "candidate limit reached"
    assert reduced.validator.calls == []
    assert reduced.nodes[deep]["status"] == "pending"
    assert reduced.nodes[deep]["attempts"] == 1
    validate_checkpoint(reduced.state, "example.test")

    expanded = DiscoverySession(
        "example.test",
        ScanOptions(passive=False, max_candidates=1, max_depth=2, find=1),
        FakeDNS([deep]),
        state=copy.deepcopy(reduced.state),
    )
    assert expanded.run()["findings"] == [deep]
    assert expanded.state["attempted"] == 1
    assert expanded.nodes[deep]["attempts"] == 2
    assert expanded.nodes[deep]["error"] == ""
    assert expanded.validator.calls == [deep]


def test_no_work_resume_does_not_count_another_stale_wave():
    first = make_session(
        hits=["ny-01-iphone.example.test"],
        options=ScanOptions(passive=False, max_number=1, max_depth=1, max_candidates=100),
    )
    first.run(["ny-01-iphone.example.test"])
    previous_stale = first.state["stale"]
    second = DiscoverySession(
        "example.test", first.options, FakeDNS(), state=copy.deepcopy(first.state)
    )
    second.run()
    assert second.validator.calls == []
    assert second.state["stale"] == previous_stale


@pytest.mark.parametrize("option", ["max_number", "max_depth", "max_per_host"])
def test_changed_expansion_options_preserve_caps_spending_and_completed_records(option):
    from enumeraite.core.scan_budget import ModelPrice, SpendLedger

    first = make_session(names=["api.example.test"], hits=["api.example.test"])
    first.run()
    state = copy.deepcopy(first.state)
    price = ModelPrice(1, 2)
    ledger = SpendLedger(5, price, spent_usd=0.5)
    state["ledger"] = ledger.to_dict()
    original_options = dict(state["options"])
    changed_options = {**original_options, option: original_options[option] + 1}
    second = DiscoverySession(
        "example.test", ScanOptions(**changed_options), FakeDNS(), state=state, ledger=ledger
    )
    assert second.state["attempted"] == first.state["attempted"]
    assert second.state["round"] == first.state["round"]
    assert second.state["ledger"] == ledger.to_dict()
    assert second.state["options"]["max_candidates"] == original_options["max_candidates"]
    assert [node["status"] for node in second.nodes.values()] == [
        node["status"] for node in first.nodes.values()
    ]
    assert second.nodes["api.example.test"]["expanded_revision"] == 0


def test_parentless_model_proposals_remain_usable_after_many_shallow_rounds():
    from enumeraite.core.scan_budget import ModelPrice, SpendLedger
    from enumeraite.providers.llm_provider import ChatResult, LLMProvider

    class FakeModel(LLMProvider):
        def chat(self, system, user, schema):
            return ChatResult(
                '{"candidates":[{"value":"new.example.test","confidence":0.8}]}',
                input_tokens=10,
                output_tokens=10,
            )

        def get_provider_name(self):
            return "fixture-model"

    ledger = SpendLedger(1, ModelPrice(1, 2))
    session = make_session(
        options=ScanOptions(max_depth=1),
        provider=FakeModel({"model": "fixture", "max_batch_calls": 1}),
        ledger=ledger,
    )
    session._enqueue("api.example.test", source="seed")
    session.nodes["api.example.test"].update(status="verified", attempts=1)
    session.state["attempted"] = 1
    session.state["round"] = 7

    assert session._model_candidates(session.findings) == 1
    node = session.nodes["new.example.test"]
    assert node["source"] == "model"
    assert node["parent"] is None
    assert node["depth"] == 0
    assert node["round"] == 7
    assert ledger.calls == 3
    assert ledger.spent_usd > 0
    validate_checkpoint(session.state, "example.test")


def test_deadline_stops_before_dns_and_checkpoint_survives():
    now = [0.0]

    def fetch(*args, **kwargs):
        now[0] = 11.0
        return PassiveResult(["api.example.test"], {}, [])

    checkpoints = []
    session = DiscoverySession(
        "example.test",
        ScanOptions(time_limit=10),
        FakeDNS(),
        passive_fetch=fetch,
        clock=lambda: now[0],
        checkpoint=lambda state: checkpoints.append(copy.deepcopy(state)),
    )
    result = session.run()
    assert result["stop_reason"] == "time limit reached"
    assert session.validator.calls == []
    assert checkpoints[-1]["stop_reason"] == "time limit reached"


def test_keyboard_interrupt_saves_pending_candidates_for_resume():
    class InterruptedDNS:
        def validate(self, values, **kwargs):
            raise KeyboardInterrupt()

    saved = []
    session = DiscoverySession(
        "example.test",
        ScanOptions(),
        InterruptedDNS(),
        passive_fetch=passive(["api.example.test"]),
        checkpoint=lambda state: saved.append(copy.deepcopy(state)),
    )
    assert session.run()["stop_reason"].startswith("interrupted")
    assert saved[-1]["nodes"][0]["status"] == "pending"
    validate_checkpoint(saved[-1], "example.test")


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_candidates", True),
        ("workers", 0),
        ("rate", float("nan")),
        ("time_limit", 0),
        ("find", 1.5),
        ("max_depth", 0),
        ("passive", "yes"),
    ],
)
def test_invalid_limits_rejected(field, value):
    with pytest.raises(ValueError):
        ScanOptions(**{field: value})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state.update(provider=[]),
        lambda state: state.update(passive_sources=None),
        lambda state: state.update(passive_truncated="yes"),
        lambda state: state.update(stop_reason=None),
        lambda state: state.update(ledger={}),
        lambda state: state.update(attempted=100),
        lambda state: state["nodes"][0].update(value="outside.test"),
        lambda state: state["nodes"][0].update(parent=state["nodes"][0]["value"]),
    ],
)
def test_corrupt_checkpoints_rejected(mutation):
    session = make_session(options=ScanOptions(max_candidates=1))
    session.run()
    state = copy.deepcopy(session.state)
    mutation(state)
    with pytest.raises(ValueError):
        validate_checkpoint(state, "example.test")
