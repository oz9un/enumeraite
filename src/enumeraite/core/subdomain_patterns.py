"""Bounded, evidence-guided DNS naming hypotheses for recursive discovery.

The scores here express relative proposal strength, not calibrated DNS hit
probabilities. In particular, an abbreviation such as ``ny`` is only a possible
state/region token; one hostname cannot establish what its owner meant by it.
"""

from __future__ import annotations

import heapq
import re
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from itertools import islice
from types import MappingProxyType
from typing import Any, Collection, Iterable, Iterator, Mapping, Sequence

_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_PIECES = re.compile(r"[a-z]+|[0-9]+|[-.]")
_STATES = tuple(
    "ca ny tx wa va fl or ma nj il az co ga nc oh pa md mi mn mo nv tn ut wi "
    "al ak ar ct de hi id in ia ks ky la me ms mt ne nh nm nd ok ri sc sd vt wv wy dc".split()
)
_LEXICONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "environment",
        tuple(
            "prod dev test stage staging qa uat sandbox beta demo production development".split()
        ),
    ),
    (
        "product",
        tuple(
            "iphone ipad mac watch icloud airpods macbook imac ios ipados macos tvos vision".split()
        ),
    ),
    ("state_or_region", _STATES),
    ("region", tuple("us eu ap emea apac au sa".split())),
    ("region", tuple("east west north south central".split())),
)
_CHILD_LABELS = ("api", "www", "dev", "admin", "mail")
_Anchor = tuple[int, str | tuple[str, tuple[str, ...]]]


@dataclass(frozen=True)
class PatternProposal:
    """An unverified candidate and the host whose expansion suggested it."""

    value: str
    parent: str
    pattern: str
    reason: str
    confidence: float


@dataclass(frozen=True)
class _Host:
    value: str
    pieces: tuple[str, ...]


@dataclass(frozen=True)
class _SlotIndex:
    """Only varying slots need their own tuple; fixed slots share the base."""

    base: tuple[str, ...]
    varying: Mapping[int, tuple[str, ...]]
    size: int


@dataclass(frozen=True)
class _PatternGroup:
    size: int
    anchors: Mapping[_Anchor, _SlotIndex]


@dataclass(frozen=True)
class PatternContext:
    """Immutable, normalized evidence/history snapshot for one planning wave.

    Construct with :func:`build_pattern_context`. Only caller-verified evidence
    contributes vocabulary; exclusions merely block proposals. Reuse this exact
    snapshot until evidence/history changes, then build another one. Indexed
    ``expand_host`` calls must pass ``evidence=None`` and no exclusions, making
    this snapshot authoritative rather than silently ignoring fresh arguments.
    """

    target: str
    known: frozenset[str]
    excluded: frozenset[str]
    blocked: frozenset[str]
    groups: Mapping[tuple[str, ...], _PatternGroup]
    children: Mapping[tuple[str, ...], Mapping[_Anchor, tuple[str, ...]]]


def build_pattern_context(
    evidence: Sequence[str], target: str, *, exclude: Collection[str] = ()
) -> PatternContext:
    """Prepare reusable immutable indexes without making DNS or provider calls."""
    apex = _target(target)
    known = frozenset(_evidence(evidence, apex))
    if isinstance(exclude, (str, bytes)) or not isinstance(exclude, Collection):
        raise ValueError("exclude must be a collection of DNS names")
    excluded = frozenset(
        name for value in exclude if (name := _normalize(value)) and _in_scope(name, apex)
    )
    groups: dict[tuple[str, ...], list[_Host]] = {}
    children: dict[str, set[str]] = {}
    for value in sorted(known - {apex}):
        parsed = _parse(value, apex)
        groups.setdefault(_shape(parsed), []).append(parsed)
        child, parent = value.split(".", 1)
        children.setdefault(parent, set()).add(child)
    return PatternContext(
        apex,
        known,
        excluded,
        known | excluded,
        MappingProxyType({shape: _index_group(group) for shape, group in groups.items()}),
        _index_children(children, apex),
    )


def expand_host(
    host: str,
    evidence: Sequence[str] | None,
    target: str,
    *,
    count: int = 32,
    max_number: int = 99,
    max_depth: int = 4,
    exclude: Collection[str] = (),
    context: PatternContext | None = None,
) -> list[PatternProposal]:
    """Propose scoped sibling/child names without querying DNS.

    ``max_depth`` counts DNS labels below ``target``. Numeric alternatives are
    bounded inclusively by ``max_number`` and retain the observed minimum digit
    width. Zero is proposed only when zero was already observed in that slot.
    Evidence must be supplied by the caller; this function never treats its own
    proposals as discoveries. Candidate count is capped at 5,000 per expansion.
    ``exclude`` supplies already-tried names for paging. They do not consume the
    candidate count and never contribute naming vocabulary or positive evidence.
    For repeated expansions, build one ``context`` and pass ``evidence=None``
    without ``exclude``. The context's evidence/history snapshot is authoritative
    and its normalized target must match. Standalone calls build the same index.
    """
    _limit("count", count, 0, 5_000)
    _limit("max_number", max_number, 0, 999_999)
    _limit("max_depth", max_depth, 0, 32)
    apex = _target(target)
    if context is None:
        if evidence is None:
            raise ValueError("evidence must be a sequence of DNS names")
        context = build_pattern_context(evidence, apex, exclude=exclude)
    else:
        if not isinstance(context, PatternContext) or context.target != apex:
            raise ValueError("context must be a PatternContext for the same normalized target")
        if evidence is not None or exclude != ():
            raise ValueError("context is authoritative; pass evidence=None and omit exclude")
    normalized = _normalize(host)
    if not normalized or not _in_scope(normalized, apex):
        return []
    if not count or not max_depth or _depth(normalized, apex) > max_depth:
        return []
    current = _parse(normalized, apex)
    observed = _indexed_slots(current, context)
    pattern = _pattern(current, observed, apex)
    seen = {normalized}
    result: list[PatternProposal] = []

    def accept(pieces: tuple[str, ...], reason: str, confidence: float) -> None:
        relative = "".join(pieces)
        value = f"{relative}.{apex}" if relative else apex
        if (
            value in context.blocked
            or value in seen
            or _depth(value, apex) > max_depth
            or not _normalize(value)
        ):
            return
        seen.add(value)
        result.append(PatternProposal(value, normalized, pattern, reason, confidence))

    # Direct substitutions observed in equivalent slots take priority over
    # speculative lexicons. Round-robin ordering gives each dimension a turn.
    observed_streams = [
        _replacements(current.pieces, index, values)
        for index, values in _observed_alternatives(current, observed, max_number)
    ]
    for pieces in _round_robin(observed_streams):
        accept(pieces, "token observed in the same naming slot", 0.82)
        if len(result) >= count:
            return result

    # Allow traversal past earlier pages without making history positive
    # evidence. Keep the existing maximum work ceilings even for a large scan.
    work_count = min(5_000, count + len(context.excluded))
    streams = [
        _cross_combinations(current, observed, max_number, work_count),
        _numeric_changes(current, observed, max_number),
        _lexical_changes(current),
        _child_changes(current, context, apex, max_depth),
    ]
    for pieces, reason, confidence in islice(_round_robin(streams), max(256, work_count * 32)):
        accept(pieces, reason, confidence)
        if len(result) >= count:
            break
    return result


def describe_patterns(evidence: Sequence[str], target: str) -> list[dict[str, Any]]:
    """Describe inferred patterns as JSON records, including their uncertainty."""
    apex = _target(target)
    known = sorted(_evidence(evidence, apex) - {apex})
    groups: dict[tuple[str, ...], list[_Host]] = {}
    for value in known:
        parsed = _parse(value, apex)
        groups.setdefault(_shape(parsed), []).append(parsed)
    records: dict[str, dict[str, Any]] = {}
    for group in groups.values():
        buckets, changed_slots = _peer_buckets(group)
        summaries: dict[tuple[_Anchor, ...], tuple[dict[int, tuple[str, ...]], list[str]]] = {}
        for current in group:
            # A bare arbitrary word provides no useful pattern by itself.
            if len(current.pieces) == 1 and not (
                current.pieces[0].isdigit() or _family(current.pieces[0])
            ):
                continue
            anchors = _anchors(current)
            varying = {index for anchor in anchors for index in changed_slots[anchor]}
            pattern = _render_pattern(current, varying, apex)
            if "<" not in pattern or pattern in records:
                continue
            # Determine the template from posting metadata before materializing
            # its peers. Thousands of arbitrary words sharing `-api` therefore
            # summarize their shared template once, not once per arbitrary word.
            universal = next(
                (anchor for anchor in anchors if len(buckets[anchor]) == len(group)), None
            )
            selected = (
                (universal,)
                if universal
                else tuple(anchor for anchor in anchors if len(buckets[anchor]) > 1) or anchors[:1]
            )
            if selected not in summaries:
                peers = {peer.value: peer for anchor in selected for peer in buckets[anchor]}
                summaries[selected] = (
                    _observed_slots(current, list(peers.values())),
                    sorted(peers),
                )
            observed, examples = summaries[selected]
            slots = []
            for index, values in observed.items():
                token = current.pieces[index]
                family = _family(token)
                if token.isdigit() and varying and index not in varying:
                    continue
                if not token.isdigit() and not family and len(values) < 2:
                    continue
                kind = "number" if token.isdigit() else family[0] if family else "observed_token"
                slots.append(
                    {
                        "position": index,
                        "label_index": current.pieces[:index].count("."),
                        "kind": kind,
                        "observed": list(values),
                        "padding": len(token) if token.isdigit() else None,
                        "tentative": bool(family),
                    }
                )
            records[pattern] = {
                "pattern": pattern,
                "evidence_count": len(examples),
                "examples": examples[:5],
                "slots": slots,
                "confidence": 0.82 if len(examples) > 1 else 0.45,
                "note": "Semantic slot labels are hypotheses; DNS validation is still required.",
            }
    return [records[key] for key in sorted(records)]


def _normalize(value: object) -> str | None:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    normalized = value.lower().removesuffix(".")
    if len(normalized) > 253 or "." not in normalized:
        return None
    if not all(_DNS_LABEL.fullmatch(label) for label in normalized.split(".")):
        return None
    if all(label.isdigit() for label in normalized.split(".")):
        return None
    return normalized


def _target(value: str) -> str:
    normalized = _normalize(value)
    if not normalized:
        raise ValueError("target must be a valid DNS domain")
    return normalized


def _limit(name: str, value: int, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")


def _evidence(evidence: Sequence[str], target: str) -> set[str]:
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise ValueError("evidence must be a sequence of DNS names")
    return {name for value in evidence if (name := _normalize(value)) and _in_scope(name, target)}


def _in_scope(value: str, target: str) -> bool:
    return value == target or value.endswith("." + target)


def _depth(value: str, target: str) -> int:
    return value.count(".") - target.count(".")


def _parse(value: str, target: str) -> _Host:
    relative = "" if value == target else value[: -(len(target) + 1)]
    return _Host(value, tuple(_PIECES.findall(relative)))


def _shape(host: _Host) -> tuple[str, ...]:
    return tuple(
        f"number:{len(piece)}" if piece.isdigit() else "word" if piece.isalpha() else piece
        for piece in host.pieces
    )


def _family(token: str) -> tuple[str, tuple[str, ...]] | None:
    return next((family for family in _LEXICONS if token in family[1]), None)


def _anchors(host: _Host) -> tuple[_Anchor, ...]:
    return tuple(
        (index, _family(piece) or piece)
        for index, piece in enumerate(host.pieces)
        if piece.isalpha()
    ) or ((-1, "numbers"),)


def _index_group(group: list[_Host]) -> _PatternGroup:
    buckets: dict[_Anchor, list[_Host]] = {}
    for host in group:
        for anchor in _anchors(host):
            buckets.setdefault(anchor, []).append(host)
    indexed: dict[_Anchor, _SlotIndex] = {}
    universal: _SlotIndex | None = None
    singletons: dict[str, _SlotIndex] = {}
    for anchor, peers in buckets.items():
        if len(peers) == len(group) and universal is not None:
            indexed[anchor] = universal
            continue
        if len(peers) == 1 and peers[0].value in singletons:
            indexed[anchor] = singletons[peers[0].value]
            continue
        base = peers[0].pieces
        varying: dict[int, set[str]] = {}
        for peer in peers[1:]:
            for index, (first, other) in enumerate(zip(base, peer.pieces, strict=True)):
                if first != other:
                    varying.setdefault(index, {first}).add(other)
        summary = _SlotIndex(
            base,
            MappingProxyType({index: tuple(sorted(values)) for index, values in varying.items()}),
            len(peers),
        )
        indexed[anchor] = summary
        if len(peers) == len(group):
            universal = summary
        if len(peers) == 1:
            singletons[peers[0].value] = summary
    return _PatternGroup(len(group), MappingProxyType(indexed))


def _index_children(
    children: dict[str, set[str]], target: str
) -> Mapping[tuple[str, ...], Mapping[_Anchor, tuple[str, ...]]]:
    groups: dict[tuple[str, ...], list[_Host]] = {}
    for parent in children:
        parsed = _parse(parent, target)
        groups.setdefault(_shape(parsed), []).append(parsed)
    indexed = {}
    for shape, group in groups.items():
        buckets: dict[_Anchor, list[str]] = {}
        for parent_host in group:
            for anchor in _anchors(parent_host):
                buckets.setdefault(anchor, []).append(parent_host.value)
        anchors = {}
        universal: tuple[str, ...] | None = None
        singletons: dict[str, tuple[str, ...]] = {}
        for anchor, parents in buckets.items():
            if len(parents) == len(group) and universal is not None:
                labels = universal
            elif len(parents) == 1 and parents[0] in singletons:
                labels = singletons[parents[0]]
            else:
                labels = tuple(sorted({label for parent in parents for label in children[parent]}))
            anchors[anchor] = labels
            if len(parents) == len(group):
                universal = labels
            if len(parents) == 1:
                singletons[parents[0]] = labels
        indexed[shape] = MappingProxyType(anchors)
    return MappingProxyType(indexed)


def _indexed_slots(host: _Host, context: PatternContext) -> dict[int, tuple[str, ...]]:
    group = context.groups.get(_shape(host))
    buckets = (
        [group.anchors[anchor] for anchor in _anchors(host) if anchor in group.anchors]
        if group
        else []
    )
    # A universal posting already contains every compatible peer. Avoid unioning
    # and sorting its potentially large numeric vocabulary for every parent.
    universal = next((bucket for bucket in buckets if group and bucket.size == group.size), None)
    if universal is not None:
        buckets = [universal]
    observed = {}
    for index, token in enumerate(host.pieces):
        if token in {"-", "."}:
            continue
        if len(buckets) == 1:
            bucket = buckets[0]
            values = bucket.varying.get(index, (bucket.base[index],))
            position = bisect_left(values, token)
            observed[index] = (
                values
                if position < len(values) and values[position] == token
                else (*values[:position], token, *values[position:])
            )
        else:
            observed[index] = tuple(
                sorted(
                    {
                        token,
                        *(
                            value
                            for bucket in buckets
                            for value in bucket.varying.get(index, (bucket.base[index],))
                        ),
                    }
                )
            )
    return observed


def _peer_buckets(
    group: list[_Host],
) -> tuple[dict[_Anchor, list[_Host]], dict[_Anchor, set[int]]]:
    """Index exact shared anchors; do not merge transitive compatibility chains."""
    buckets: dict[_Anchor, list[_Host]] = {}
    for host in group:
        for anchor in _anchors(host):
            buckets.setdefault(anchor, []).append(host)
    changed = {
        anchor: {
            index
            for peer in peers[1:]
            for index, (first, other) in enumerate(zip(peers[0].pieces, peer.pieces, strict=True))
            if first != other
        }
        for anchor, peers in buckets.items()
    }
    return buckets, changed


def _compatible(left: _Host, right: _Host) -> bool:
    if _shape(left) != _shape(right):
        return False
    # A shared literal or recognized semantic family is an anchor. Identical
    # punctuation alone does not establish that two arbitrary names are peers.
    return any(
        lpart.isalpha()
        and (lpart == rpart or (_family(lpart) is not None and _family(lpart) == _family(rpart)))
        for lpart, rpart in zip(left.pieces, right.pieces, strict=True)
    ) or all(not part.isalpha() for part in left.pieces)


def _observed_slots(host: _Host, peers: list[_Host]) -> dict[int, tuple[str, ...]]:
    return {
        index: tuple(sorted({token, *(peer.pieces[index] for peer in peers)}))
        for index, token in enumerate(host.pieces)
        if token not in {"-", "."}
    }


def _pattern(host: _Host, observed: dict[int, tuple[str, ...]], target: str) -> str:
    return _render_pattern(
        host, {index for index, values in observed.items() if len(values) > 1}, target
    )


def _render_pattern(host: _Host, varying: set[int], target: str) -> str:
    pieces = list(host.pieces)
    for index, token in enumerate(pieces):
        family = _family(token)
        if token.isdigit() and (index in varying or not varying):
            pieces[index] = f"<number:0{len(token)}d>"
        elif family:
            pieces[index] = f"<{family[0]}?>"
        elif index in varying:
            pieces[index] = "<observed_token>"
    return "".join(pieces) + "." + target if pieces else target


def _replace(pieces: tuple[str, ...], index: int, value: str) -> tuple[str, ...]:
    return (*pieces[:index], value, *pieces[index + 1 :])


def _replacements(
    pieces: tuple[str, ...], index: int, values: Iterable[str]
) -> Iterator[tuple[str, ...]]:
    for value in values:
        yield _replace(pieces, index, value)


def _observed_alternatives(
    host: _Host, observed: dict[int, tuple[str, ...]], max_number: int
) -> list[tuple[int, list[str]]]:
    return [
        (
            index,
            [
                value
                for value in values
                if value != host.pieces[index] and (not value.isdigit() or int(value) <= max_number)
            ],
        )
        for index, values in observed.items()
    ]


def _round_robin(streams: Iterable[Iterable[Any]]) -> Iterator[Any]:
    pending = deque(iter(stream) for stream in streams)
    while pending:
        stream = pending.popleft()
        try:
            yield next(stream)
        except StopIteration:
            continue
        pending.append(stream)


def _cross_combinations(
    host: _Host, observed: dict[int, tuple[str, ...]], max_number: int, count: int
) -> Iterator[tuple[tuple[str, ...], str, float]]:
    slots = [
        (index, [host.pieces[index], *values])
        for index, values in _observed_alternatives(host, observed, max_number)
        if values
    ]
    if len(slots) < 2:
        return
    # Best-first traversal covers short combinations across every dimension.
    # The frontier is capped, so a many-dimensional grid cannot explode.
    initial = (0,) * len(slots)
    frontier = [(0, initial)]
    seen = {initial}
    work_limit = min(20_000, max(64, count * 8))
    while frontier:
        _, offsets = heapq.heappop(frontier)
        if sum(offset > 0 for offset in offsets) >= 2:
            pieces = host.pieces
            for (index, values), offset in zip(slots, offsets, strict=True):
                pieces = _replace(pieces, index, values[offset])
            yield pieces, "combination of tokens observed in equivalent naming slots", 0.72
        if len(seen) >= work_limit:
            continue
        for dimension, (_, values) in enumerate(slots):
            if offsets[dimension] + 1 >= len(values):
                continue
            changed = (*offsets[:dimension], offsets[dimension] + 1, *offsets[dimension + 1 :])
            if changed not in seen:
                seen.add(changed)
                heapq.heappush(frontier, (sum(changed), changed))
                if len(seen) >= work_limit:
                    break


def _numeric_changes(
    host: _Host, observed: dict[int, tuple[str, ...]], max_number: int
) -> Iterator[tuple[tuple[str, ...], str, float]]:
    def numbers(index: int, token: str) -> Iterator[tuple[str, ...]]:
        value = int(token)
        if value > max_number:
            return
        minimum = 0 if any(int(item) == 0 for item in observed[index]) else 1
        for distance in range(1, max_number + 1):
            for adjacent in (value + distance, value - distance):
                if minimum <= adjacent <= max_number:
                    yield _replace(host.pieces, index, str(adjacent).zfill(len(token)))

    for pieces in _round_robin(
        numbers(index, token) for index, token in enumerate(host.pieces) if token.isdigit()
    ):
        yield pieces, "numeric-slot hypothesis preserving observed zero padding", 0.65


def _lexical_changes(host: _Host) -> Iterator[tuple[tuple[str, ...], str, float]]:
    def replacements(index: int, family: tuple[str, tuple[str, ...]]) -> Iterator[Any]:
        for value in family[1]:
            if value != host.pieces[index]:
                yield (
                    _replace(host.pieces, index, value),
                    f"tentative {family[0]} slot inferred from an ambiguous naming token",
                    0.35,
                )

    yield from _round_robin(
        replacements(index, family)
        for index, token in enumerate(host.pieces)
        if (family := _family(token))
    )


def _child_changes(
    host: _Host, context: PatternContext, target: str, max_depth: int
) -> Iterator[tuple[tuple[str, ...], str, float]]:
    if _depth(host.value, target) >= max_depth:
        return
    anchors = context.children.get(_shape(host), {})
    # Several anchors can point to the same universal/single-parent tuple.
    # Traverse that potentially large child vocabulary only once.
    buckets = {id(labels): labels for anchor in _anchors(host) if (labels := anchors.get(anchor))}
    observed_children = {child for labels in buckets.values() for child in labels}
    for child in sorted(observed_children):
        yield (
            (child, ".", *host.pieces) if host.pieces else (child,),
            "child label observed below a host with the same naming pattern",
            0.70,
        )
    existing_labels = set("".join(host.pieces).split("."))
    for child in _CHILD_LABELS:
        if child not in existing_labels and child not in observed_children:
            yield (
                (child, ".", *host.pieces) if host.pieces else (child,),
                "tentative deeper DNS label; delegation or existence is not assumed",
                0.25,
            )
