"""Small, deterministic enumeration plans for a decision model to prioritize.

Models can rank these plans, but cannot change their scope, range or DNS truth.
Numeric families vary one observed slot at a time and retain literal anchors.
The existing node history is the range cursor, including negative results.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from .subdomain_patterns import PatternProposal

_NUMBERS = re.compile(r"(\d+)")
_IPV4_LABEL = re.compile(r"([0-9]{1,3})-([0-9]{1,3})-([0-9]{1,3})-([0-9]{1,3})\Z")


@dataclass
class DiscoveryPlan:
    id: str
    description: str
    proposals: list[PatternProposal] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    feedback: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5

    def context(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "evidence": self.evidence[:5],
            "feedback": self.feedback,
            "next_examples": [item.value for item in self.proposals[:3]],
        }


@dataclass(frozen=True)
class _NumericExpansion:
    """Immutable recipe; never retains caller nodes or a live generator frame."""

    key: tuple[str, ...]
    slot: int
    width: int
    spans: tuple[range, ...]
    seen: frozenset[int]
    excluded: frozenset[str]
    target: str
    max_depth: int
    parent: str
    pattern: str

    def candidates(
        self, start_span: int = 0, start_number: int | None = None
    ) -> Iterator[tuple[PatternProposal, int, int]]:
        for index in range(start_span, len(self.spans)):
            span = self.spans[index]
            beginning = (
                max(span.start, start_number)
                if index == start_span and start_number is not None
                else span.start
            )
            for number in range(beginning, span.stop):
                if number in self.seen:
                    continue
                pieces = list(self.key)
                pieces[self.slot] = str(number).zfill(self.width)
                value = "".join(pieces) + "." + self.target
                # Preserve the eager path's global width-crossing exclusions
                # and exact depth filtering without retaining mutable history.
                if value in self.excluded:
                    continue
                if value.count(".") - self.target.count(".") > self.max_depth:
                    continue
                yield (
                    PatternProposal(
                        value,
                        self.parent,
                        self.pattern,
                        "Systematic numeric family enumeration; fixed tokens preserved",
                        0.9,
                    ),
                    index,
                    number,
                )


@dataclass(frozen=True)
class NumericPlanBlueprint:
    """Eligible numeric-plan metadata plus a repeatable bounded expansion.

    Use ``materialize()`` after admission to obtain the ordinary mutable
    ``DiscoveryPlan`` expected by prediction, UI previews, and enqueueing.
    Metadata and exclusions describe the catalog's authoritative snapshot.
    """

    id: str
    description: str
    evidence: tuple[str, ...]
    feedback: Mapping[str, Any]
    priority: float
    first_proposal: PatternProposal
    _count: int = field(repr=False)
    _expansion: _NumericExpansion = field(repr=False)
    _cursor: tuple[int, int] = field(repr=False)

    def materialize(self) -> DiscoveryPlan:
        """Build this page without mutating the catalog or borrowing input data."""
        proposals = [self.first_proposal]
        if not len(proposals) >= self._count:
            for proposal, _, _ in self._expansion.candidates(*self._cursor):
                proposals.append(proposal)
                if len(proposals) >= self._count:
                    break
        return DiscoveryPlan(
            self.id,
            self.description,
            proposals=proposals,
            evidence=list(self.evidence),
            feedback=dict(self.feedback),
            priority=self.priority,
        )


@dataclass(frozen=True)
class NumericPlanCatalog:
    """One planning-wave snapshot, including only non-exhausted numeric plans.

    Rebuild after evidence, history or scan limits change. Neither catalog
    construction nor materialization queries DNS or changes persistent state.
    """

    plans: tuple[NumericPlanBlueprint, ...]
    membership: Mapping[str, str]


def plan_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _numeric_buckets(
    nodes: dict[str, dict], target: str
) -> Iterator[tuple[tuple[str, ...], list[tuple[int, dict]]]]:
    """Bridge canonical width boundaries without merging different padding.

    A width must occur in verified evidence before it can define a family.
    Longer spellings join that family only when formatting with its minimum
    width reproduces them exactly: 9 and 10 can meet, but 9 and 010 cannot.
    Pure width-specific families keep their identities. A newly observed lower
    width can supersede a redundant canonical-only higher width; actual padded
    evidence always retains its own spelling convention.
    """
    axes: dict[tuple[str, ...], list[tuple[str, int, dict]]] = {}
    for node in nodes.values():
        value = node["value"]
        if not value.endswith("." + target):
            continue
        pieces = _NUMBERS.split(value[: -(len(target) + 1)])
        # Bound the cross-slot index even for unusually numeric DNS labels.
        for index in range(1, min(len(pieces), 33), 2):
            token = pieces[index]
            if len(token) > 5:
                continue
            key = tuple([*pieces[:index], "<number>", *pieces[index + 1 :]])
            axes.setdefault(key, []).append((token, int(token), node))
    for axis, items in axes.items():
        slot = axis.index("<number>")
        widths = sorted({len(token) for token, _, node in items if node["status"] == "verified"})
        supported: list[set[str]] = []
        for width in widths:
            compatible = [
                (token, number, node)
                for token, number, node in items
                if str(number).zfill(width) == token
            ]
            verified = [item for item in compatible if item[2]["status"] == "verified"]
            if len({number for _, number, _ in verified}) < 2:
                continue
            members = {node["value"] for _, _, node in verified}
            has_padding = any(len(token) > 1 and token.startswith("0") for token, _, _ in verified)
            if not has_padding and any(members <= lower for lower in supported):
                # app9/app10/app11 supports unpadded numbers, not an additional
                # app01 convention merely because two values have two digits.
                continue
            supported.append(members)
            key = (*axis[:slot], f"<number:0{width}d>", *axis[slot + 1 :])
            yield key, [(number, node) for _, number, node in compatible]


def _ipv4_encoding_matches(verified: list[tuple[int, dict]], slot: int) -> int:
    """Return support for an IPv4 range hypothesis, never an existence claim.

    Only a numeric axis inside an exact four-octet leftmost label qualifies.
    Every verified member must agree with its decoded A address; missing,
    contradictory, or out-of-range evidence disables this optional inference.
    """
    if slot not in (1, 3, 5, 7):
        return 0
    addresses = set()
    for _, node in verified:
        match = _IPV4_LABEL.fullmatch(node["value"].split(".", 1)[0])
        if match is None:
            return 0
        octets = [int(value) for value in match.groups()]
        if any(value > 255 for value in octets):
            return 0
        address = ".".join(str(value) for value in octets)
        if address not in node.get("addresses", []):
            return 0
        addresses.add(address)
    return len(addresses) if len(addresses) >= 2 else 0


def numeric_plans(
    nodes: dict[str, dict], target: str, *, max_number: int, max_depth: int, count: int
) -> tuple[list[DiscoveryPlan], dict[str, str]]:
    """Find independently supported numeric axes; never infer a Cartesian product.

    At least two verified values with otherwise identical literals support an axis.
    Padding is a minimum width, not a numeric upper bound. Verified address
    mappings may support an inferred IPv4-octet range; user limits always win.
    """
    catalog = build_numeric_catalog(
        nodes, target, max_number=max_number, max_depth=max_depth, count=count
    )
    return [plan.materialize() for plan in catalog.plans], dict(catalog.membership)


def build_numeric_catalog(
    nodes: dict[str, dict], target: str, *, max_number: int, max_depth: int, count: int
) -> NumericPlanCatalog:
    """Snapshot numeric metadata once; construct full pages only after admission.

    Checking one candidate per family preserves the eager function's exact
    eligibility, so an exhausted family cannot displace productive work. The
    rest of each page is an immutable recipe sharing one exclusion snapshot.
    ``numeric_plans`` remains the eager convenience wrapper.
    """
    plans = []
    membership: dict[str, str] = {}
    excluded = frozenset(nodes)
    for key, items in _numeric_buckets(nodes, target):
        verified = [(number, node) for number, node in items if node["status"] == "verified"]
        numbers = {number for number, _ in verified}
        if len(numbers) < 2:
            continue
        anchors = [node for _, node in verified if node["depth"] < max_depth]
        if not anchors:
            continue
        pattern = "".join(key) + "." + target
        identity = plan_id(pattern)
        for _, node in verified:
            membership.setdefault(node["value"], identity)
        seen = frozenset(number for number, _ in items)
        anchor = min(anchors, key=lambda node: (node["depth"], node["round"], node["value"]))
        slot = next(i for i, piece in enumerate(key) if piece.startswith("<number:"))
        width = int(key[slot][len("<number:0") : -2])
        encoding_matches = _ipv4_encoding_matches(verified, slot)
        minimum = 0 if encoding_matches or 0 in numbers else 1
        maximum = min(max_number, 255) if encoding_matches else max_number
        description = f"Enumerate {pattern} within {minimum}–{maximum}; keep other tokens fixed"
        feedback: dict[str, Any] = {
            "verified": len(verified),
            "tested": sum(node["attempts"] > 0 for _, node in items),
            "miss": sum(node["status"] == "miss" for _, node in items),
            "inconclusive": sum(node["status"] == "inconclusive" for _, node in items),
            "range_min": minimum,
            "range_max": maximum,
            "observed_min": min(numbers),
            "observed_max": max(numbers),
        }
        if encoding_matches:
            description += (
                f"; inferred IPv4 encoding from {encoding_matches} matching DNS answers, "
                "not a claim that untested names exist"
            )
            feedback.update(
                range_basis="verified_ipv4_encoding",
                range_evidence_count=encoding_matches,
                user_range_max=max_number,
            )
        # Exhaustion is always a range/limit claim, not proof of DNS absence.
        spans = (
            range(min(numbers), min(max(numbers), maximum) + 1),
            range(max(numbers) + 1, maximum + 1),
            range(minimum, min(min(numbers), maximum + 1)),
        )
        expansion = _NumericExpansion(
            key, slot, width, spans, seen, excluded, target, max_depth, anchor["value"], pattern
        )
        first = next(expansion.candidates(), None)
        if first is not None:
            proposal, span_index, number = first
            plans.append(
                NumericPlanBlueprint(
                    identity,
                    description,
                    tuple(node["value"] for _, node in verified[:5]),
                    MappingProxyType(feedback),
                    0.9,
                    proposal,
                    count,
                    expansion,
                    (span_index, number + 1),
                )
            )
    return NumericPlanCatalog(
        tuple(sorted(plans, key=lambda plan: (-plan.feedback["verified"], plan.id))),
        MappingProxyType(membership),
    )


def allow_child_trial(
    proposal: PatternProposal, family: str, membership: Mapping[str, str], nodes: dict[str, dict]
) -> bool:
    """Two representative hosts per child label unless a child actually resolved.

    Timeouts are not negative evidence. They consume the bounded trial allocation
    but do not label the rule a failure; normal resume retry rules still apply.
    """
    label = proposal.value[: -(len(proposal.parent) + 1)]
    prefix = label + "."
    trials = [
        node
        for node in nodes.values()
        if node["value"].startswith(prefix)
        and membership.get(node["value"][len(prefix) :]) == family
    ]
    return any(node["status"] == "verified" for node in trials) or len(trials) < 2
