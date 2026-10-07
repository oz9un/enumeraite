"""Resumable breadth-first DNS discovery with shared resource limits."""

from __future__ import annotations

import copy
import ipaddress
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import groupby
from typing import Any, Callable, Sequence, TypeVar

from anthropic import APITimeoutError as AnthropicTimeoutError
from openai import APITimeoutError as OpenAITimeoutError

from .discovery_context import model_batch_context
from .discovery_metrics import discovery_method_counts
from .discovery_plans import (
    DiscoveryPlan,
    NumericPlanBlueprint,
    allow_child_trial,
    build_numeric_catalog,
    numeric_plans,
    plan_id,
)
from .discovery_vocabulary import VOCABULARY_PROTOCOL_VERSION, build_vocabulary_layouts
from .dns_evidence import validate_dns_evidence
from .engine import GenerationEngine
from .origin_evidence import validate_origin_evidence
from .passive import _sample_names, collect_passive, normalize_domain
from .prompt_spec import SUBDOMAIN_SPEC
from .scan_budget import BudgetExhausted, ScanPaused, SpendLedger, attach_scan_guard
from .strategies import SubdomainDiscoveryStrategy
from .subdomain_patterns import build_pattern_context, describe_patterns, expand_host
from .text_safety import safe_exception_text, safe_model_text

_AdmittedPlan = TypeVar("_AdmittedPlan", bound=DiscoveryPlan | NumericPlanBlueprint)

BOOTSTRAP_LABELS = (
    "www",
    "mail",
    "api",
    "dev",
    "staging",
    "cdn",
    "blog",
    "shop",
    "docs",
    "support",
    "status",
    "portal",
    "auth",
    "admin",
    "test",
    "vpn",
    "m",
    "static",
    "assets",
    "login",
    # A small, target-independent fallback broadens evidence when public
    # sources are unavailable. These are root-scan hypotheses, never labels
    # prepended to every discovered host. Keep the original first 20 stable.
    "www2",
    "web",
    "app",
    "apps",
    "mobile",
    "beta",
    "demo",
    "sandbox",
    "stage",
    "prod",
    "uat",
    "qa",
    "sso",
    "id",
    "idp",
    "account",
    "accounts",
    "secure",
    "services",
    "service",
    "gateway",
    "developer",
    "developers",
    "download",
    "downloads",
    "images",
    "img",
    "media",
    "video",
    "files",
    "ftp",
    "sftp",
    "upload",
    "email",
    "webmail",
    "smtp",
    "imap",
    "pop",
    "pop3",
    "autodiscover",
    "autoconfig",
    "ns",
    "ns1",
    "ns2",
    "ns3",
    "dns",
    "dns1",
    "dns2",
    "mx",
    "mx1",
    "mx2",
    "mail1",
    "mail2",
    "remote",
    "intranet",
    "help",
    "helpdesk",
    "wiki",
    "git",
    "gitlab",
    "monitor",
    "monitoring",
    "metrics",
    "search",
    "cloud",
    "store",
    "connect",
    "community",
    "news",
    "events",
)
STATE_VERSION = 1
PATTERN_PAGING_VERSION = 2
_NUMERIC_RUNS = re.compile(r"([0-9]+)")
_LITERAL_WORDS = re.compile(r"[a-z]+")


def normalize_target(value: str) -> str:
    return normalize_domain(value)


@dataclass
class ScanOptions:
    # Zero disables each total-work limit; per-wave and DNS safeguards still apply.
    max_candidates: int = 1000
    max_rounds: int = 10
    find: int | None = None
    time_limit: float = 300
    workers: int = 20
    rate: float = 50
    dns_timeout: float = 2
    batch_size: int = 40
    known_limit: int = 200
    max_per_host: int = 32
    max_depth: int = 4
    # Ceiling for changed deterministic numeric slots, not every number in a
    # hostname. Explicit/model/passive names and fixed numeric literals survive.
    max_number: int = 99
    # DNS misses are not evidence that the remaining naming space is exhausted.
    # Zero keeps breadth-first exploration running within the other hard limits.
    stale_rounds: int = 0
    passive: bool = True
    # Optional complementary AI vocabulary, never a replacement for DNS truth.
    vocabulary_breadth: bool = False

    def __post_init__(self) -> None:
        bounds = {
            "workers": (1, 100),
            "batch_size": (1, 200),
            "known_limit": (1, 5000),
            "max_per_host": (1, 200),
            "max_depth": (1, 10),
            "max_number": (1, 9999),
            "stale_rounds": (0, 20),
        }
        for name, (minimum, maximum) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
        for name in ("max_candidates", "max_rounds"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer (0 means no limit)")
        if self.find is not None and (type(self.find) is not int or self.find < 1):
            raise ValueError("find must be a positive integer")
        if (
            isinstance(self.time_limit, bool)
            or not isinstance(self.time_limit, (int, float))
            or not math.isfinite(self.time_limit)
            or self.time_limit < 0
        ):
            raise ValueError("time_limit must be finite and nonnegative (0 means no limit)")
        for name, maximum in (("rate", 1000), ("dns_timeout", 30)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"{name} must be positive and at most {maximum}")
        for name in ("passive", "vocabulary_breadth"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")


def validate_checkpoint(state: dict, target: str) -> None:
    if state.get("version") != STATE_VERSION or type(state.get("version")) is not int:
        raise ValueError("Unsupported scan checkpoint version")
    if state.get("target") != target:
        raise ValueError("Checkpoint target does not match this scan")
    try:
        ScanOptions(**state["options"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint has invalid scan options") from exc
    nodes = state.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Checkpoint has invalid discovery nodes")
    seen: set[str] = set()
    depths: dict[str, int] = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Invalid checkpoint node")
        value = node.get("value")
        if (
            not isinstance(value, str)
            or not SUBDOMAIN_SPEC.validator(value)
            or value != value.lower().rstrip(".")
            or not value.endswith("." + target)
            or value in seen
        ):
            raise ValueError("Checkpoint contains invalid, duplicate, or out-of-scope names")
        parent = node.get("parent")
        if parent is not None and parent not in seen:
            raise ValueError("Checkpoint contains an invalid or cyclic discovery parent")
        seen.add(value)
        if node.get("status") not in {"pending", "verified", "miss", "wildcard", "inconclusive"}:
            raise ValueError("Checkpoint contains an invalid validation status")
        for field in ("round", "depth", "expanded_revision", "attempts"):
            if type(node.get(field)) is not int or node[field] < 0:
                raise ValueError("Checkpoint contains invalid discovery counters")
        if node["depth"] > 10 or node["attempts"] > 2:
            raise ValueError("Checkpoint discovery depth or retry count exceeds its limit")
        if parent is not None and node["depth"] != depths[parent] + 1:
            raise ValueError("Checkpoint discovery depths do not match parent edges")
        depths[value] = node["depth"]
        if node["status"] != "pending" and node["attempts"] == 0:
            raise ValueError("Checkpoint contains validation without a DNS attempt")
        for field in ("source", "pattern", "reason", "error"):
            if not isinstance(node.get(field), str) or len(node[field]) > 1000:
                raise ValueError("Checkpoint contains invalid node metadata")
        addresses = node.get("addresses")
        if not isinstance(addresses, list) or len(addresses) > 100:
            raise ValueError("Checkpoint contains invalid DNS addresses")
        for address in addresses:
            ipaddress.ip_address(address)
        if "dns_evidence" in node:
            validate_dns_evidence(node["dns_evidence"], value, target)
        if "origin_evidence" in node:
            validate_origin_evidence(node["origin_evidence"], value, target)
    for field in ("round", "attempted", "stale"):
        if type(state.get(field)) is not int or state[field] < 0:
            raise ValueError("Checkpoint contains invalid session counters")
    model_batches = state.get("model_batches", 0)
    if type(model_batches) is not int or model_batches < 0:
        raise ValueError("Checkpoint contains invalid model batch counter")
    vocabulary_version = state.get("vocabulary_protocol_version", VOCABULARY_PROTOCOL_VERSION)
    if type(vocabulary_version) is not int or vocabulary_version != VOCABULARY_PROTOCOL_VERSION:
        raise ValueError("Checkpoint contains unsupported vocabulary protocol version")
    if "pattern_paging_version" in state and (
        type(state["pattern_paging_version"]) is not int
        or not 1 <= state["pattern_paging_version"] <= PATTERN_PAGING_VERSION
    ):
        raise ValueError("Checkpoint contains unsupported pattern paging version")
    if state["attempted"] != sum(node["attempts"] > 0 for node in nodes):
        raise ValueError("Checkpoint attempt count does not match its validation records")
    if type(state.get("initialized")) is not bool:
        raise ValueError("Checkpoint contains invalid initialization state")
    if not isinstance(state.get("stop_reason"), str) or len(state["stop_reason"]) > 1000:
        raise ValueError("Checkpoint contains an invalid stop reason")
    if type(state.get("passive_truncated")) is not bool:
        raise ValueError("Checkpoint contains invalid passive metadata")
    sources = state.get("passive_sources")
    if not isinstance(sources, dict) or len(sources) > 10:
        raise ValueError("Checkpoint contains invalid passive sources")
    for name, values in sources.items():
        if (
            not isinstance(name, str)
            or len(name) > 100
            or not isinstance(values, list)
            or len(values) > 5000
        ):
            raise ValueError("Checkpoint contains invalid passive source records")
        if any(
            not isinstance(value, str)
            or not SUBDOMAIN_SPEC.validator(value)
            or not value.endswith("." + target)
            for value in values
        ):
            raise ValueError("Checkpoint contains out-of-scope passive evidence")
    provider = state.get("provider")
    if provider is not None and (
        not isinstance(provider, dict)
        or set(provider) != {"name", "model"}
        or any(
            not isinstance(value, str) or not value or len(value) > 200
            for value in provider.values()
        )
    ):
        raise ValueError("Checkpoint provider is invalid")
    if state.get("ledger") is not None:
        try:
            SpendLedger.from_dict(state["ledger"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Checkpoint model budget is invalid") from exc
    elif provider is not None:
        raise ValueError("Checkpoint is missing its model budget")
    predictor = state.get("predictor")
    if predictor is not None and predictor != {"name": "jev", "model": "jev-1.13.0"}:
        raise ValueError("Checkpoint predictor is invalid")
    if state.get("prediction_ledger") is not None:
        if predictor is None:
            raise ValueError("Checkpoint prediction budget has no predictor")
        try:
            SpendLedger.from_dict(state["prediction_ledger"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Checkpoint prediction budget is invalid") from exc
    elif predictor is not None:
        raise ValueError("Checkpoint is missing its prediction budget")
    predictions = state.get("predictions", [])
    if not isinstance(predictions, list) or len(predictions) > 60:
        raise ValueError("Checkpoint prediction history is invalid")
    for prediction in predictions:
        if (
            not isinstance(prediction, dict)
            or set(prediction) != {"round", "plan_id", "description", "score", "status", "count"}
            or type(prediction["round"]) is not int
            or prediction["round"] < 0
            or type(prediction["count"]) is not int
            or not 0 <= prediction["count"] <= 200
            or prediction["status"] not in {"scored", "fallback"}
            or not isinstance(prediction["plan_id"], str)
            or not 1 <= len(prediction["plan_id"]) <= 80
            or not isinstance(prediction["description"], str)
            or len(prediction["description"]) > 500
        ):
            raise ValueError("Checkpoint prediction record is invalid")
        score = prediction["score"]
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (float, int))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Checkpoint prediction score is invalid")
        if (score is not None) != (prediction["status"] == "scored"):
            raise ValueError("Checkpoint prediction status is inconsistent")
    elapsed = state.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise ValueError("Checkpoint elapsed time is invalid")
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Checkpoint elapsed time is invalid")
    if (
        not isinstance(state.get("warnings"), list)
        or not all(isinstance(value, str) and len(value) <= 1000 for value in state["warnings"])
        or len(state["warnings"]) > 100
    ):
        raise ValueError("Checkpoint warnings are invalid")


class DiscoverySession:
    def __init__(
        self,
        target: str,
        options: ScanOptions,
        validator: Any,
        *,
        passive_fetch: Callable = collect_passive,
        provider: Any = None,
        ledger: SpendLedger | None = None,
        predictor: Any = None,
        prediction_ledger: SpendLedger | None = None,
        checkpoint: Callable[[dict], None] | None = None,
        event: Callable[[str, dict], None] | None = None,
        state: dict | None = None,
        clock: Callable[[], float] = time.monotonic,
        stop_requested: Callable[[], bool] | None = None,
    ):
        self.target = normalize_target(target)
        self.options = options
        self.validator = validator
        self.passive_fetch = passive_fetch
        self.provider = provider
        self.ledger = ledger
        self.predictor = predictor
        self.prediction_ledger = prediction_ledger
        self.prediction_enabled = predictor is not None
        self.checkpoint = checkpoint or (lambda state: None)
        self.event = event or (lambda kind, payload: None)
        self.stop_requested = stop_requested or (lambda: False)
        self.clock = clock
        if state is not None:
            validate_checkpoint(state, self.target)
        expansion_changed = state is not None and (
            state.get("pattern_paging_version") != PATTERN_PAGING_VERSION
            or any(
                state["options"][name] != getattr(options, name)
                for name in ("max_number", "max_depth", "max_per_host")
            )
        )
        self.state: dict[str, Any] = (
            state
            if state is not None
            else {
                "version": STATE_VERSION,
                "target": self.target,
                "options": asdict(options),
                "initialized": False,
                "nodes": [],
                "round": 0,
                "attempted": 0,
                "stale": 0,
                "elapsed_seconds": 0.0,
                "stop_reason": "starting",
                "warnings": [],
                "passive_sources": {},
                "passive_truncated": False,
                "provider": None,
                "ledger": None,
            }
        )
        self.state["options"] = asdict(options)
        self.state.setdefault("predictor", None)
        self.state.setdefault("prediction_ledger", None)
        self.state.setdefault("predictions", [])
        self.state.setdefault("model_batches", 0)
        self.state.setdefault("vocabulary_protocol_version", VOCABULARY_PROTOCOL_VERSION)
        # Revisit exhausted parents after recipe or paging changes, retaining
        # checked names, request counters, spending, and user limits.
        self.state["pattern_paging_version"] = PATTERN_PAGING_VERSION
        self.nodes = {node["value"]: node for node in self.state["nodes"]}
        if expansion_changed:
            self.state["stale"] = 0
            for node in self.nodes.values():
                if node["status"] == "verified":
                    node["expanded_revision"] = 0
        self.started = clock()
        self.deadline = self.started + options.time_limit if options.time_limit else None
        self.prior_elapsed = self.state["elapsed_seconds"]
        self.model_enabled = provider is not None
        self._model_stop_reason: str | None = None
        if provider is not None:
            if ledger is None:
                raise ValueError("AI scanning requires a spend ledger")
            identity = {"name": provider.get_provider_name(), "model": str(provider.model)}
            if self.state.get("provider") not in (None, identity):
                raise ValueError("Resume requires the same provider and model as the saved scan")
            self.state["provider"] = identity
            attach_scan_guard(
                provider, ledger, self.deadline, self.save, stop_requested=self.stop_requested
            )
        if predictor is not None:
            if prediction_ledger is None:
                raise ValueError("Jev prediction requires a separate spend ledger")
            identity = {"name": "jev", "model": str(predictor.model)}
            if identity["model"] != "jev-1.13.0":
                raise ValueError("Unsupported prediction model")
            if self.state.get("predictor") not in (None, identity):
                raise ValueError("Resume requires the same prediction model")
            self.state["predictor"] = identity

    @property
    def findings(self) -> list[str]:
        return [value for value, node in self.nodes.items() if node["status"] == "verified"]

    def save(self) -> None:
        self.state["elapsed_seconds"] = self.prior_elapsed + max(0, self.clock() - self.started)
        if self.ledger is not None:
            self.state["ledger"] = self.ledger.to_dict()
        if self.prediction_ledger is not None:
            self.state["prediction_ledger"] = self.prediction_ledger.to_dict()
        self.checkpoint(self.state)

    def _warn(self, value: str) -> None:
        value = safe_model_text(value, 1000)
        if value not in self.state["warnings"] and len(self.state["warnings"]) < 100:
            self.state["warnings"].append(value)
            self.event("warning", {"message": value})

    def _node_event(self, kind: str, node: dict) -> None:
        # Event consumers can retain or decorate snapshots without changing the
        # checkpoint, including nested DNS observation records.
        self.event(kind, copy.deepcopy(node))

    def _enqueue_dns_aliases(self, node: dict) -> None:
        evidence = node.get("dns_evidence")
        if evidence is None:
            return
        for alias in evidence.get("aliases", ()):
            if alias not in self.nodes:
                self._enqueue(
                    alias,
                    source="dns",
                    parent=node["value"],
                    reason="CNAME target observed in a DNS answer; independently verified next",
                    depth=node["depth"] + 1,
                    origin_evidence={"evidence": [node["value"]]},
                )

    def _enqueue(
        self,
        value: str,
        *,
        source: str,
        parent: str | None = None,
        pattern: str = "",
        reason: str = "",
        depth: int = 0,
        origin_evidence: dict[str, Any] | None = None,
    ) -> bool:
        value = value.strip().lower().rstrip(".")
        if (
            value in self.nodes
            or not SUBDOMAIN_SPEC.validator(value)
            or not value.endswith("." + self.target)
            or depth > self.options.max_depth
            or len(value[: -(len(self.target) + 1)].split(".")) > self.options.max_depth
            or not self._candidate_capacity(1)
        ):
            return False
        origin = copy.deepcopy(origin_evidence) if origin_evidence is not None else {}
        origin["queued_at"] = datetime.now(timezone.utc).isoformat()
        validate_origin_evidence(origin, value, self.target)
        node = {
            "value": value,
            "parent": parent,
            "source": safe_model_text(source, 1000),
            "pattern": safe_model_text(pattern, 1000),
            "reason": safe_model_text(reason, 1000),
            "origin_evidence": origin,
            "round": self.state["round"],
            "depth": depth,
            "expanded_revision": 0,
            "status": "pending",
            "addresses": [],
            "error": "",
            "attempts": 0,
        }
        self.nodes[value] = node
        self.state["nodes"].append(node)
        self._node_event("candidate", node)
        return True

    def _candidate_capacity(self, count: int) -> int:
        """Fit one bounded wave within the optional total candidate cap."""
        if not self.options.max_candidates:
            return count
        return min(count, max(0, self.options.max_candidates - len(self.nodes)))

    def _pending_allowed(self, node: dict) -> bool:
        """Keep saved work dormant when current depth/numeric limits exclude it."""
        return (
            node["status"] == "pending"
            and node["attempts"] < 2
            and node["depth"] <= self.options.max_depth
            and len(node["value"][: -(len(self.target) + 1)].split(".")) <= self.options.max_depth
            and self._pattern_numbers_allowed(node)
        )

    def _pattern_numbers_allowed(self, node: dict) -> bool:
        """Reapply the numeric ceiling without treating literals as range axes.

        Existing checkpoints already retain each deterministic proposal's
        parent. Corresponding digit runs identify changed numeric slots even
        across padding-width boundaries or simultaneous lexical substitutions.
        A child label is vocabulary, not a numeric mutation of its parent.
        Missing/ambiguous legacy provenance restricts only explicitly numeric
        rules; it must not invent a blanket ban on numbered hostnames.
        """
        if node.get("source") != "pattern":
            return True
        value = node.get("value")
        suffix = "." + self.target
        if not isinstance(value, str) or not value.endswith(suffix):
            return False
        relative = value[: -len(suffix)]
        parts = _NUMERIC_RUNS.split(relative)
        numbers = [int(number) for number in parts[1::2]]
        if all(number <= self.options.max_number for number in numbers):
            return True
        reason = node.get("reason", "")
        explicitly_numeric = isinstance(reason, str) and reason.startswith(
            ("numeric-slot hypothesis", "Systematic numeric family enumeration")
        )
        parent_name = node.get("parent")
        if not isinstance(parent_name, str):
            return not explicitly_numeric
        parent = self.nodes.get(parent_name)
        if (
            not isinstance(parent, dict)
            or parent.get("value") != parent_name
            or not SUBDOMAIN_SPEC.validator(parent_name)
            or not parent_name.endswith(suffix)
        ):
            return not explicitly_numeric
        if value.endswith("." + parent_name):
            # Observed v200 children are vocabulary, not a request to enumerate
            # 1..200. Only an explicitly numeric legacy rule could imply that.
            return not explicitly_numeric
        previous = _NUMERIC_RUNS.split(parent_name[: -len(suffix)])
        same_structure = len(parts) == len(previous) and all(
            _LITERAL_WORDS.sub("<word>", left) == _LITERAL_WORDS.sub("<word>", right)
            for left, right in zip(parts[::2], previous[::2], strict=False)
        )
        if not same_structure:
            return not explicitly_numeric
        return all(
            current == int(before) or current <= self.options.max_number
            for current, before in zip(numbers, previous[1::2], strict=True)
        )

    def _stop(self) -> str | None:
        if self.stop_requested():
            return "paused; scan saved"
        if self.options.find is not None and len(self.findings) >= self.options.find:
            return "verified target reached"
        if self.deadline is not None and self.clock() >= self.deadline:
            return "time limit reached"
        if self.options.max_candidates and self.state["attempted"] >= self.options.max_candidates:
            retry_pending = any(
                self._pending_allowed(node) and node["attempts"] > 0 for node in self.nodes.values()
            )
            if not retry_pending:
                return "candidate limit reached"
        return None

    def _initialize(self, seeds: Sequence[str], *, prior_names: Sequence[str] = ()) -> None:
        for seed in seeds:
            self._enqueue(
                seed, source="seed", reason="User-supplied name queued for DNS verification"
            )
        for name in prior_names:
            self._enqueue(
                name,
                source="history",
                reason="Previously DNS-verified in a saved scan; queued for fresh DNS verification",
            )
        if self.options.passive:
            self.event(
                "phase",
                {"message": "Collecting known subdomains from passive sources", "stage": "passive"},
            )
            try:
                result = self.passive_fetch(
                    self.target,
                    timeout=min(10.0, self.options.time_limit) if self.options.time_limit else 10.0,
                    max_results=(
                        min(self.options.known_limit, self.options.max_candidates)
                        if self.options.max_candidates
                        else self.options.known_limit
                    ),
                    deadline=self.deadline,
                )
                self.state["passive_sources"] = result.sources
                eligible = {
                    name
                    for name in result.names
                    if name not in self.nodes
                    and SUBDOMAIN_SPEC.validator(name)
                    and name.endswith("." + self.target)
                    and len(name[: -(len(self.target) + 1)].split(".")) <= self.options.max_depth
                }
                capacity = self._candidate_capacity(len(eligible))
                selected = _sample_names(eligible, self.target, capacity) if capacity else []
                self.state["passive_truncated"] = result.truncated or len(eligible) > capacity
                # Preserve provenance for existing seeds, then fairly allocate
                # only the remaining candidate slots across eligible families.
                retained = [name for name in result.names if name in self.nodes]
                for name in (*retained, *selected):
                    records = result.evidence.get(name, [])
                    origin = {"passive": records} if records else None
                    if name in self.nodes and records:
                        # A supplied seed may also be independently present in
                        # public sources. Preserve its first-queued method.
                        node = self.nodes[name]
                        combined = {
                            **node.get("origin_evidence", {}),
                            "passive": copy.deepcopy(records),
                        }
                        validate_origin_evidence(combined, name, self.target)
                        node["origin_evidence"] = combined
                        self._node_event("candidate", node)
                    else:
                        self._enqueue(
                            name,
                            source="passive",
                            reason="Previously observed by passive sources",
                            origin_evidence=origin,
                        )
                for error in result.errors:
                    self._warn(error)
                if self.state["passive_truncated"]:
                    self._warn(
                        "Passive results were capped; this is a partial view of known names."
                    )
                self.event("passive", {"count": len(result.names), "sources": result.sources})
            except Exception as exc:
                self._warn("Passive discovery failed: " + safe_exception_text(exc))
        else:
            self.event(
                "phase",
                {
                    "message": (
                        "Verifying supplied seed names before expanding their patterns"
                        if seeds
                        else "Starting from common service names"
                    ),
                    "stage": "seeds",
                },
            )
        # Seeded discovery must leave room for hypotheses derived from the user's
        # actual naming examples, even with a small candidate cap. Root scans and
        # explicitly passive-first scans still retain common service fallbacks.
        if self.options.passive or not seeds:
            # Preserve the original small-scan allowance. A larger fallback
            # must not fill a tight candidate budget before evidence-guided
            # discovery gets a turn; spend at most 10% beyond that allowance.
            bootstrap_limit = len(BOOTSTRAP_LABELS)
            if self.options.max_candidates:
                remaining = self.options.max_candidates - len(self.nodes)
                bootstrap_limit = min(bootstrap_limit, max(20, remaining // 10))
            for label in BOOTSTRAP_LABELS[:bootstrap_limit]:
                self._enqueue(
                    f"{label}.{self.target}", source="bootstrap", reason="Common service hypothesis"
                )
        self.state["initialized"] = True
        self.save()

    def _validate_batch(self, pending: list[str]) -> int:
        pending = [value for value in pending if self._pending_allowed(self.nodes[value])]
        count = self.options.batch_size
        if self.options.find is not None:
            count = min(count, self.options.find - len(self.findings))
        retries = [value for value in pending if self.nodes[value]["attempts"] > 0]
        new_values = [value for value in pending if self.nodes[value]["attempts"] == 0]
        available = (
            max(0, self.options.max_candidates - self.state["attempted"])
            if self.options.max_candidates
            else count
        )
        values = (retries + new_values[:available])[:count]
        if not values or self.stop_requested():
            return 0
        self.event(
            "phase",
            {
                "message": f"Verifying {len(values)} queued candidates",
                "stage": "dns",
                "count": len(values),
            },
        )
        outcomes = self.validator.validate(values, deadline=self.deadline)
        by_value = {result.subdomain: result for result in outcomes}
        hits = 0
        for value in values:
            node = self.nodes[value]
            outcome = by_value.get(value)
            evidence = getattr(outcome, "dns_evidence", None)
            if evidence is not None:
                node["dns_evidence"] = copy.deepcopy(evidence)
            if getattr(outcome, "cancelled", False) or (outcome is None and self.stop_requested()):
                # A paused rate waiter or partially checked record is unfinished
                # work. It must not become a miss or consume the retry allowance.
                self._node_event("result", node)
                continue
            if evidence is None:
                # A generic validator result cannot substantiate old evidence
                # for a newly completed attempt.
                node.pop("dns_evidence", None)
            if node["attempts"] == 0:
                self.state["attempted"] += 1
            node["attempts"] += 1
            node["error"] = ""
            if outcome is None or outcome.error:
                node["status"] = "inconclusive"
                node["error"] = safe_model_text(
                    outcome.error if outcome else "missing result", 1000
                )
            elif outcome.wildcard:
                node["status"] = "wildcard"
            elif outcome.exists:
                node["status"] = "verified"
                node["addresses"] = list(outcome.ip_addresses)[:100]
                hits += 1
                self._node_event("finding", node)
                self._enqueue_dns_aliases(node)
            else:
                node["status"] = "miss"
            self._node_event("result", node)
        self.event("progress", self.summary())
        self.save()
        return hits

    def _expand(self) -> bool:
        evidence = self.findings
        if not evidence:
            return False
        self.state["round"] += 1
        added = 0
        # Give an explicitly selected model a bounded first turn on verified
        # evidence. Otherwise a broad deterministic frontier can consume every
        # remaining candidate slot before hosted generation ever gets a chance.
        if self.model_enabled and not self._stop() and self._candidate_capacity(1):
            added += self._model_candidates(evidence)
        self.event(
            "phase",
            {
                "message": f"Expanding verified patterns — round {self.state['round']}",
                "stage": "patterns",
            },
        )
        # A family is one work item, not one full brute-force job per host.
        # History supplies the cursor, so misses are not regenerated on resume.
        wave_size = min(200, max(40, self.options.batch_size))
        numeric_catalog = build_numeric_catalog(
            self.nodes,
            self.target,
            max_number=self.options.max_number,
            max_depth=self.options.max_depth,
            count=wave_size,
        )
        membership = numeric_catalog.membership
        offered: list[DiscoveryPlan | NumericPlanBlueprint] = list(numeric_catalog.plans)
        # New hits go first, while older evidence is revisited after vocabulary grows.
        frontier = sorted(
            (
                node
                for node in self.nodes.values()
                if node["status"] == "verified"
                and node["depth"] < self.options.max_depth
                and node["expanded_revision"] < len(evidence)
            ),
            key=lambda node: (node["expanded_revision"], node["round"], node["value"]),
        )
        if len(frontier) > 200:
            # A nonempty first page must not permanently hide later parents.
            # The saved round supplies a deterministic cursor without changing
            # old checkpoints or confusing "considered" with "exhausted".
            offset = ((self.state["round"] - 1) * 200) % len(frontier)
            frontier = frontier[offset:] + frontier[:offset]
        grouped: dict[str, DiscoveryPlan] = {}
        page_groups = []
        # Snapshot verified vocabulary and all tried/pending exclusions once,
        # after the model pass has enqueued its candidates. Pattern proposals
        # are not enqueued until planning ends, so this snapshot stays current.
        pattern_context = (
            build_pattern_context(evidence, self.target, exclude=self.nodes)
            if frontier and self._candidate_capacity(1) and not self._stop()
            else None
        )
        # Empty pages must not hide productive later parents. Bound planning
        # CPU and paid context independently of the maximum DNS candidate cap.
        for offset in range(0, len(frontier), 200):
            if not self._candidate_capacity(1) or self._stop():
                break
            groups = []
            for node in frontier[offset : offset + 200]:
                if self._stop():
                    break
                groups.append(
                    (
                        node,
                        expand_host(
                            node["value"],
                            None,
                            self.target,
                            count=self.options.max_per_host,
                            max_number=self.options.max_number,
                            max_depth=self.options.max_depth,
                            context=pattern_context,
                        ),
                    )
                )
            page_groups.extend(groups)
            for node, proposals in groups:
                for proposal in proposals:
                    if proposal.value in self.nodes:
                        continue
                    family = membership.get(node["value"])
                    if family and proposal.reason.startswith("numeric-slot"):
                        continue  # Supported varying slots are covered by family plans.
                    child = proposal.value.endswith("." + proposal.parent)
                    if (
                        family
                        and child
                        and not allow_child_trial(proposal, family, membership, self.nodes)
                    ):
                        continue
                    identity = plan_id((family or proposal.pattern) + proposal.reason)
                    plan = grouped.setdefault(
                        identity,
                        DiscoveryPlan(
                            identity,
                            f"{proposal.pattern}: {proposal.reason}"[:500],
                            priority=proposal.confidence,
                        ),
                    )
                    if len(plan.proposals) < wave_size:
                        plan.proposals.append(proposal)
                    if node["value"] not in plan.evidence and len(plan.evidence) < 5:
                        plan.evidence.append(node["value"])
            if grouped:
                break
        for node in self.nodes.values():
            family = membership.get(node.get("parent"))
            identity = plan_id((family or node["pattern"]) + node["reason"])
            if identity in grouped:
                feedback = grouped[identity].feedback
                status = node["status"]
                feedback[status] = feedback.get(status, 0) + 1
        offered.extend(grouped.values())
        allowance = self._candidate_capacity(wave_size)
        selected = self._admit_plans(offered, min(24, allowance))
        if self._stop() or not self._candidate_capacity(1):
            self.save()
            return bool(added)
        # Admission depends only on metadata. Full numeric pages are needed
        # only for these selected plans, before prediction previews/UI counts.
        plans = []
        for selected_plan in selected:
            if self._stop():
                self.save()
                return bool(added)
            plans.append(
                selected_plan.materialize()
                if isinstance(selected_plan, NumericPlanBlueprint)
                else selected_plan
            )
        plans = self._rank_plans(plans)
        # Keep an exploration share even if Jev assigns one plan a very high
        # score. Scores change scheduling, never DNS truth or hard scan limits.
        positions = {plan.id: 0 for plan in plans}
        remaining = allowance
        while remaining > 0 and plans and not self._stop():
            progress = False
            for index, plan in enumerate(plans):
                exploration = min(len(plans) - index - 1, remaining - 1)
                quota = min(max(1, math.ceil(remaining * 0.75)), remaining - exploration)
                accepted = 0
                while positions[plan.id] < len(plan.proposals) and accepted < quota:
                    proposal = plan.proposals[positions[plan.id]]
                    positions[plan.id] += 1
                    family = membership.get(proposal.parent)
                    if (
                        family
                        and proposal.value.endswith("." + proposal.parent)
                        and not allow_child_trial(proposal, family, membership, self.nodes)
                    ):
                        continue
                    if self._stop():
                        break
                    parent = self.nodes[proposal.parent]
                    if self._enqueue(
                        proposal.value,
                        source="pattern",
                        parent=proposal.parent,
                        pattern=proposal.pattern,
                        reason=proposal.reason,
                        depth=parent["depth"] + 1,
                        origin_evidence={
                            "plan_id": plan.id,
                            "plan_description": plan.description,
                            "evidence": plan.evidence[:5],
                            "confidence": proposal.confidence,
                            "numeric_ceiling": self.options.max_number,
                        },
                    ):
                        added += 1
                        accepted += 1
                        remaining -= 1
                        progress = True
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break
            if not progress:
                break
        for node, proposals in page_groups:
            # A filled page is not an exhausted family. The next expansion must
            # skip attempted names and visit the next page, even after misses.
            if not proposals:
                node["expanded_revision"] = len(evidence)
        self.save()
        return bool(added)

    def _admit_plans(self, plans: list[_AdmittedPlan], limit: int) -> list[_AdmittedPlan]:
        """Protect admission across families before asking a predictor to rank.

        Most slots retain deterministic proposal priorities. At least one quarter
        walk all eligible plans, so low-priority families are not permanently
        excluded by the model-context cap. Equal-priority leaders also rotate,
        sharing the large first-plan allocation across rounds. A stable predictor
        sort may still move a genuinely higher-scored admitted plan to the front.
        The saved round is the only cursor; node history remains the DNS cursor.
        """
        if not plans or limit <= 0:
            return []
        turn = max(0, self.state["round"] - 1)
        ordered = []
        for _, group in groupby(
            sorted(plans, key=lambda plan: (-plan.priority, plan.id)),
            key=lambda plan: plan.priority,
        ):
            tied = list(group)
            offset = turn % len(tied)
            ordered.extend(tied[offset:] + tied[:offset])
        if len(ordered) <= limit:
            return ordered
        if limit == 1:
            # A final single candidate has no room for an exploration share.
            # Preserve the strongest available hypothesis at a tight hard cap.
            return ordered[:1]
        exploration = max(1, math.ceil(limit * 0.25))
        selected = {plan.id for plan in ordered[: limit - exploration]}
        by_identity = sorted(plans, key=lambda plan: plan.id)
        offset = (turn * exploration) % len(by_identity)
        for index in range(len(by_identity)):
            selected.add(by_identity[(offset + index) % len(by_identity)].id)
            if len(selected) >= limit:
                break
        return [plan for plan in ordered if plan.id in selected]

    def _rank_plans(self, plans: list[DiscoveryPlan]) -> list[DiscoveryPlan]:
        if not plans or not self.prediction_enabled or self._stop():
            return plans
        self.event(
            "phase", {"message": "Jev is prioritizing discovery plans", "stage": "prediction"}
        )
        scores: dict[str, float] = {}
        try:
            scores = self.predictor.score_plans(
                {
                    "target": self.target,
                    "verified_examples": self.findings[:20],
                    "recent_results": [
                        {"value": node["value"], "status": node["status"]}
                        for node in self.state["nodes"][-40:]
                        if node["status"] != "pending"
                    ],
                    "remaining_candidates": (
                        max(0, self.options.max_candidates - len(self.nodes))
                        if self.options.max_candidates
                        else None
                    ),
                    "round": self.state["round"],
                },
                [plan.context() for plan in plans],
                ledger=self.prediction_ledger,
                deadline=self.deadline,
                checkpoint=self.save,
                stop_requested=self.stop_requested,
            )
            if set(scores) != {plan.id for plan in plans} or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
                for value in scores.values()
            ):
                raise ValueError("Invalid prediction scores")
        except ScanPaused:
            return plans
        except BudgetExhausted:
            self.prediction_enabled = False
            scores = {}
            self._warn("Jev prediction budget exhausted; using deterministic plan priorities.")
        except Exception:
            # Raw provider errors can contain secrets or target data. The
            # adapter accounts for uncertain usage; do not retry paid requests.
            self.prediction_enabled = False
            scores = {}
            self._warn("Jev prediction unavailable; using deterministic plan priorities.")
        for plan in plans:
            self.state["predictions"].append(
                {
                    "round": self.state["round"],
                    "plan_id": plan.id,
                    "description": plan.description[:500],
                    "score": scores.get(plan.id),
                    "status": "scored" if scores else "fallback",
                    "count": len(plan.proposals),
                }
            )
        self.state["predictions"] = self.state["predictions"][-60:]
        self.save()
        return sorted(plans, key=lambda plan: -scores.get(plan.id, plan.priority))

    def _model_candidates(self, evidence: list[str]) -> int:
        if not self._candidate_capacity(1) or self._stop():
            return 0
        # Three different questions are more useful than repeatedly refilling
        # one answer. Keep room for cheap Python expansion in the same wave.
        allowance = 60
        if self.options.max_candidates:
            remaining = self.options.max_candidates - len(self.nodes)
            allowance = min(allowance, remaining - min(40, remaining // 3))
        added = 0
        limit_calls = getattr(self.provider, "set_runtime_batch_limit", None)
        try:
            if callable(limit_calls):
                limit_calls(1)
            for _ in range(3):
                if not self.model_enabled or self._stop() or added >= allowance:
                    break
                count = self._candidate_capacity(min(20, allowance - added))
                added += self._model_batch(evidence, count)
                # Persist successful names before another paid request can
                # fail, hit a budget boundary, or be interrupted.
                self.save()
        finally:
            self.provider.set_runtime_context("")
            self.provider.set_runtime_exclusions(())
            if callable(limit_calls):
                limit_calls()
        return added

    def _model_batch(self, evidence: list[str], count: int) -> int:
        if self.ledger is not None and (
            self.ledger.blocked
            or self.ledger.reserved_usd > 0
            or (
                self.ledger.remaining_usd <= 0
                and (
                    self.ledger.pricing.input_usd_per_million > 0
                    or self.ledger.pricing.output_usd_per_million > 0
                )
            )
        ):
            self.model_enabled = False
            self._model_stop_reason = "AI budget exhausted"
            self._warn("AI budget exhausted; continuing free pattern discovery.")
            return 0
        batch_index = self.state["model_batches"]
        seeds, instructions, focus = model_batch_context(evidence, self.target, batch_index)
        if not seeds:
            return 0
        # Replace an existing semantic pass only when Python can supply a
        # witnessed finite layout. Empty/invalid answers never trigger paid retries.
        layouts = (
            build_vocabulary_layouts(seeds, self.target)
            if self.options.vocabulary_breadth and batch_index % 3 == 2
            else ()
        )
        if layouts:
            focus = "grounded naming vocabulary"
        self.state["model_batches"] += 1
        self.event(
            "phase",
            {
                "message": f"Exploring {focus} — AI batch {self.state['model_batches']}",
                "stage": "model",
                "focus": focus,
                "batch": self.state["model_batches"],
            },
        )
        self.provider.set_runtime_exclusions(list(self.nodes))
        recent = self.state["nodes"][-100:]
        feedback = "\n".join(
            f"{node['status']}: {node['value']}" for node in recent if node["status"] != "pending"
        )
        families, _ = numeric_plans(
            self.nodes,
            self.target,
            max_number=self.options.max_number,
            max_depth=self.options.max_depth,
            count=1,
        )
        if families:
            feedback += (
                "\nPython already enumerates these numeric families. Do not list consecutive "
                "numbers for them; suggest distinct naming hypotheses instead:\n"
                + "\n".join(plan.description for plan in families[:12])
            )
        self.provider.set_runtime_context(instructions + "\n\n" + feedback)
        try:
            # SDK retries are disabled by the spend guard. Recover timeouts here
            # so every attempt gets its own durable reservation and uncertain
            # charge. Three attempts bound even free, open-ended scans.
            for attempt in range(3):
                if self._stop():
                    return 0
                try:
                    if layouts:
                        result = self.provider.generate_subdomain_vocabulary(
                            layouts,
                            self.target,
                            count,
                            instructions=instructions + "\n\n" + feedback,
                            exclusions=list(self.nodes),
                        )
                    else:
                        result = GenerationEngine(self.provider).generate_paths(
                            seeds,
                            self.target,
                            count,
                            strategy=SubdomainDiscoveryStrategy(),
                        )
                    break
                except (TimeoutError, OpenAITimeoutError, AnthropicTimeoutError):
                    if self._stop():
                        return 0
                    if attempt == 2:
                        raise
                    self._warn(
                        "AI request timed out; retrying within the remaining scan budget "
                        f"(attempt {attempt + 2}/3)."
                    )
                    self.save()
            rationales = result.metadata.get("rationales", {})
            if not isinstance(rationales, dict):
                rationales = {}
            provenance = result.metadata.get("provenance", {})
            if not isinstance(provenance, dict):
                provenance = {}
            witnessed = {layout.plan_id: layout for layout in layouts}
            added = 0
            for index, value in enumerate(result.paths[:count]):
                proof = provenance.get(value, {})
                plan_id = proof.get("plan_id") if isinstance(proof, dict) else None
                layout = witnessed.get(plan_id) if isinstance(plan_id, str) else None
                word = proof.get("word") if isinstance(proof, dict) else None
                pattern = ""
                origin = {
                    "provider": self.state["provider"]["name"],
                    "model": self.state["provider"]["model"],
                    "focus": focus,
                    "batch": self.state["model_batches"],
                    "evidence": seeds,
                    "confidence": result.confidence_scores[index],
                }
                if (
                    layout is not None
                    and isinstance(word, str)
                    and layout.template.replace("{value}", word) + "." + self.target == value
                ):
                    pattern = layout.template + "." + self.target
                    origin.update(
                        evidence=list(layout.evidence),
                        plan_id=layout.plan_id,
                        layout=layout.template,
                        word=word,
                    )
                added += self._enqueue(
                    value,
                    source="model",
                    pattern=pattern,
                    reason=(
                        rationales[value]
                        if isinstance(rationales.get(value), str)
                        else "Model hypothesis from verified evidence"
                    ),
                    depth=0,
                    origin_evidence=origin,
                )
            self.event(
                "phase",
                {
                    "message": f"Queued {added} AI hypotheses for DNS validation",
                    "stage": "model_complete",
                    "count": added,
                },
            )
            return added
        except ScanPaused:
            self.model_enabled = False
        except BudgetExhausted:
            self.model_enabled = False
            self._model_stop_reason = "AI budget exhausted"
            self._warn("AI budget exhausted; continuing free pattern discovery.")
        except (TimeoutError, OpenAITimeoutError, AnthropicTimeoutError) as exc:
            self.model_enabled = False
            self._model_stop_reason = "AI timeout recovery exhausted"
            self._warn(
                "AI timeout recovery exhausted; continuing free pattern discovery: "
                + safe_exception_text(exc)
            )
        except Exception as exc:
            self.model_enabled = False
            self._model_stop_reason = "AI provider failed"
            self._warn("AI generation stopped: " + safe_exception_text(exc))
        return 0

    def run(self, seeds: Sequence[str] = (), *, prior_names: Sequence[str] = ()) -> dict:
        try:
            known = self.findings
            for value in known:
                self._node_event("finding", self.nodes[value])
            if self.stop_requested():
                self.state["stop_reason"] = "paused; scan saved"
                return self.summary()
            if not self.state["initialized"]:
                self._initialize(seeds, prior_names=prior_names)
            else:
                # Retry transient DNS failures once on resume, without treating
                # timeouts as evidence that a hostname does not exist.
                for node in self.nodes.values():
                    if node["status"] == "inconclusive" and node["attempts"] < 2:
                        node["status"] = "pending"
                for seed in seeds:
                    self._enqueue(seed, source="seed", reason="Additional user-supplied seed")
            for value in known:
                self._enqueue_dns_aliases(self.nodes[value])
            round_hits = 0
            round_validated = False
            while True:
                reason = self._stop()
                if reason:
                    self.state["stop_reason"] = reason
                    break
                pending = [
                    value for value, node in self.nodes.items() if self._pending_allowed(node)
                ]
                if pending:
                    round_hits += self._validate_batch(pending)
                    round_validated = True
                    continue
                if self.state["round"] and round_validated:
                    self.state["stale"] = 0 if round_hits else self.state["stale"] + 1
                round_hits = 0
                round_validated = False
                if self.options.stale_rounds and self.state["stale"] >= self.options.stale_rounds:
                    self.state["stop_reason"] = "diminishing returns"
                    break
                if self.options.max_rounds and self.state["round"] >= self.options.max_rounds:
                    self.state["stop_reason"] = "round limit reached"
                    break
                if not self._expand():
                    dormant = any(node["status"] == "pending" for node in self.nodes.values())
                    self.state["stop_reason"] = self._stop() or (
                        "remaining queued work is outside current limits"
                        if dormant
                        else (
                            self._model_stop_reason + "; deterministic frontier exhausted"
                            if self._model_stop_reason
                            else "frontier exhausted"
                        )
                    )
                    break
        except KeyboardInterrupt:
            self.state["stop_reason"] = "interrupted; scan saved"
        finally:
            self.save()
        return self.summary()

    def summary(self) -> dict:
        return {
            "target": self.target,
            "findings": self.findings,
            "verified": len(self.findings),
            "known_verified": sum(
                node["status"] == "verified" and node["source"] in {"seed", "passive", "history"}
                for node in self.nodes.values()
            ),
            "attempted": self.state["attempted"],
            "queued": sum(node["status"] == "pending" for node in self.nodes.values()),
            "inconclusive": sum(node["status"] == "inconclusive" for node in self.nodes.values()),
            "wildcard": sum(node["status"] == "wildcard" for node in self.nodes.values()),
            "round": self.state["round"],
            "model_batches": self.state["model_batches"],
            "discovery_methods": discovery_method_counts(self.nodes.values()),
            "stop_reason": self.state["stop_reason"],
            "elapsed_seconds": self.prior_elapsed + max(0, self.clock() - self.started),
            "ledger": self.ledger.to_dict() if self.ledger else self.state.get("ledger"),
            "predictor": self.state.get("predictor"),
            "prediction_ledger": (
                self.prediction_ledger.to_dict()
                if self.prediction_ledger
                else self.state.get("prediction_ledger")
            ),
            "predictions": list(self.state.get("predictions", [])),
            "warnings": self.state["warnings"],
            "passive_sources": self.state["passive_sources"],
            "passive_truncated": self.state["passive_truncated"],
        }

    def report(self) -> dict:
        return {
            **self.summary(),
            "patterns": describe_patterns(self.findings, self.target),
            "nodes": self.state["nodes"],
        }
