"""Reusable pattern snapshots preserve evidence boundaries and proposal order."""

import random
from dataclasses import FrozenInstanceError

import pytest

from enumeraite.core import subdomain_patterns as patterns
from enumeraite.core.subdomain_patterns import build_pattern_context, expand_host

TARGET = "example.test"


def test_snapshot_canonicalizes_scope_and_does_not_borrow_mutable_inputs():
    evidence = ["NY-01-IPHONE.EXAMPLE.TEST.", "ca-02-ipad.example.test", "other.invalid"]
    excluded = ["ny-02-iphone.example.test", "api.ny-01-iphone.example.test"]
    context = build_pattern_context(evidence, "EXAMPLE.TEST.", exclude=excluded)
    before = expand_host(evidence[0], None, TARGET, count=80, context=context)
    evidence.append("tx-04-watch.example.test")
    excluded.append(before[0].value)
    assert expand_host(evidence[0], None, TARGET, count=80, context=context) == before
    assert context.target == TARGET
    assert context.known == frozenset({"ny-01-iphone.example.test", "ca-02-ipad.example.test"})
    assert not set(excluded[:2]) & {proposal.value for proposal in before}


def test_snapshot_is_deeply_immutable():
    context = build_pattern_context(["app01.example.test", "app02.example.test"], TARGET)
    group = next(iter(context.groups.values()))
    slots = next(iter(group.anchors.values()))
    children = next(iter(context.children.values()))
    with pytest.raises(FrozenInstanceError):
        context.target = "other.test"
    with pytest.raises(AttributeError):
        context.known.add("app03.example.test")
    for mapping in (context.groups, group.anchors, slots.varying, context.children, children):
        with pytest.raises(TypeError):
            mapping["changed"] = ()
    assert isinstance(slots.base, tuple)
    assert all(isinstance(values, tuple) for values in slots.varying.values())
    assert all(isinstance(values, tuple) for values in children.values())


@pytest.mark.parametrize("evidence", [[], ["app01.example.test"], "ignored"])
def test_context_requires_explicit_snapshot_evidence(evidence):
    context = build_pattern_context(["app01.example.test"], TARGET)
    with pytest.raises(ValueError, match="authoritative"):
        expand_host("app01.example.test", evidence, TARGET, context=context)


@pytest.mark.parametrize("exclude", [[], {}, {"app02.example.test"}, "", None])
def test_context_rejects_supplied_exclusions_instead_of_ignoring_them(exclude):
    context = build_pattern_context(["app01.example.test"], TARGET)
    with pytest.raises(ValueError, match="authoritative"):
        expand_host("app01.example.test", None, TARGET, context=context, exclude=exclude)


def test_context_requires_same_normalized_target_and_valid_type():
    context = build_pattern_context(["app01.example.test"], TARGET)
    assert expand_host("app01.example.test", None, "EXAMPLE.TEST.", context=context)
    for wrong in (context, object()):
        with pytest.raises(ValueError, match="same normalized target"):
            expand_host("app01.other.test", None, "other.test", context=wrong)
    with pytest.raises(ValueError, match="evidence"):
        expand_host("app01.example.test", None, TARGET)


def test_excluded_misses_never_become_slot_or_child_vocabulary():
    host = "app-red.example.test"
    context = build_pattern_context(
        [host, "app-blue.example.test"],
        TARGET,
        exclude=["app-green.example.test", "secret.app-blue.example.test"],
    )
    proposals = expand_host(host, None, TARGET, count=100, context=context)
    assert "green" not in str(context.groups)
    assert "secret" not in str(context.children)
    assert not any("green" in item.value or "secret" in item.value for item in proposals)


def test_compatibility_is_not_a_transitive_family_merge():
    evidence = ["alpha-red.example.test", "alpha-blue.example.test", "gamma-blue.example.test"]
    context = build_pattern_context(evidence, TARGET)
    current = patterns._parse(evidence[0], TARGET)
    assert patterns._indexed_slots(current, context) == {0: ("alpha",), 2: ("blue", "red")}


def test_children_are_indexed_by_parent_pattern_not_parent_verification():
    evidence = ["app01.example.test", "status.app02.example.test"]
    context = build_pattern_context(evidence, TARGET)
    proposals = expand_host(evidence[0], None, TARGET, count=30, max_number=2, context=context)
    observed = next(item for item in proposals if item.value == "status.app01.example.test")
    assert observed.reason == "child label observed below a host with the same naming pattern"
    assert observed.confidence == 0.70


def test_shared_child_parent_is_parsed_once_and_reuses_immutable_labels(monkeypatch):
    parent = "long-literal-parent.example.test"
    evidence = [f"child{index:04d}.{parent}" for index in range(1_000)]
    original = patterns._parse
    parsed_parents = []

    def parse(value, target):
        if value == parent:
            parsed_parents.append(value)
        return original(value, target)

    monkeypatch.setattr(patterns, "_parse", parse)
    context = build_pattern_context(evidence, TARGET)
    assert parsed_parents == [parent]
    children = next(iter(context.children.values()))
    assert len(children) == 3
    assert len({id(labels) for labels in children.values()}) == 1
    assert len(next(iter(children.values()))) == len(evidence)


def test_apex_children_and_unknown_host_still_match_standalone():
    evidence = [TARGET, "app01.example.test", "admin.app01.example.test"]
    context = build_pattern_context(evidence, TARGET)
    for host in (TARGET, "app02.example.test", "ny-01-iphone.example.test", "other.test"):
        assert expand_host(host, None, TARGET, context=context) == expand_host(
            host, evidence, TARGET
        )


def test_randomized_slots_match_reference_peer_filter_and_outputs_match():
    rng = random.Random(4109)
    names = [
        f"{left}-{number:02d}-{right}.{TARGET}"
        for left in ("alpha", "bravo", "ca", "ny", "qa", "prod")
        for number in range(4)
        for right in ("red", "blue", "iphone", "ipad")
    ]
    names += [f"{child}.{host}" for child in ("secret", "status") for host in names[:20]]
    names += ["209-64-116-001.example.test", "209-64-116-002.example.test", TARGET]
    names += ["bad..example.test", "ny-01-iphone.other.test", "https://example.test"]
    for _ in range(60):
        evidence = rng.sample(names, rng.randrange(1, 90))
        exclusions = rng.sample(names, rng.randrange(0, 40))
        host = rng.choice(names[:-3])
        context = build_pattern_context(evidence, TARGET, exclude=exclusions)
        current = patterns._parse(host, TARGET)
        peers = [
            patterns._parse(value, TARGET) for value in sorted(context.known) if value != TARGET
        ]
        peers = [peer for peer in peers if patterns._compatible(current, peer)]
        assert patterns._indexed_slots(current, context) == patterns._observed_slots(current, peers)
        options = dict(count=rng.randrange(1, 35), max_number=rng.randrange(1, 8), max_depth=3)
        assert expand_host(host, None, TARGET, context=context, **options) == expand_host(
            host, evidence, TARGET, exclude=exclusions, **options
        )


def test_repeated_expansion_does_not_renormalize_evidence_or_history(monkeypatch):
    evidence = [f"worker{index:05d}.{TARGET}" for index in range(1_000)]
    history = [*evidence, *(f"miss{index:05d}.{TARGET}" for index in range(1_000))]
    context = build_pattern_context(evidence, TARGET, exclude=history)
    original = patterns._normalize
    calls = []

    def track(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(patterns, "_normalize", track)
    for host in evidence[:20]:
        expand_host(host, None, TARGET, max_number=2, context=context)
    assert len(calls) < 200
    assert not any(value.startswith("miss") for value in calls)


def test_session_builds_one_snapshot_after_model_enqueue(monkeypatch):
    from enumeraite.core import discovery_session
    from enumeraite.core.discovery_session import DiscoverySession, ScanOptions

    session = DiscoverySession(TARGET, ScanOptions(max_candidates=200, max_per_host=5), object())
    session.model_enabled = True
    evidence = ["ny-01-iphone.example.test", "ca-02-ipad.example.test"]
    for host in evidence:
        session._enqueue(host, source="seed")
        session.nodes[host].update(status="verified", attempts=1)
    pending = "model-only.example.test"

    def model_candidates(verified):
        session._enqueue(pending, source="model")
        return 1

    built = []
    used = []
    original_build = discovery_session.build_pattern_context
    original_expand = discovery_session.expand_host

    def build(*args, **kwargs):
        snapshot = original_build(*args, **kwargs)
        built.append(snapshot)
        return snapshot

    def expand(*args, **kwargs):
        used.append(kwargs["context"])
        assert args[1] is None
        assert "exclude" not in kwargs
        return original_expand(*args, **kwargs)

    monkeypatch.setattr(session, "_model_candidates", model_candidates)
    monkeypatch.setattr(discovery_session, "build_pattern_context", build)
    monkeypatch.setattr(discovery_session, "expand_host", expand)
    assert session._expand()
    assert len(built) == 1
    assert len(used) == len(evidence)
    assert all(snapshot is built[0] for snapshot in used)
    assert pending in built[0].excluded
    assert pending not in built[0].known
