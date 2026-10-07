"""Bounded, optional observations from the scanner's existing DNS queries."""

from __future__ import annotations

import ipaddress
import math
import re
from datetime import datetime, timezone
from typing import Any

from .prompt_spec import SUBDOMAIN_SPEC

_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)")
_RECORD_STATUSES = {"answered", "no_answer", "nxdomain", "error", "not_checked"}
_WILDCARD_STATUSES = {
    "not_checked",
    "not_detected",
    "matched",
    "different_answer",
    "inconclusive",
}


def new_dns_evidence(name: str) -> dict[str, Any]:
    """Start an observation only when a candidate lookup is actually attempted."""
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": 0.0,
        "records": {
            kind: {"status": "not_checked", "addresses": [], "error": ""} for kind in ("A", "AAAA")
        },
        "wildcard": {"status": "not_checked", "parent": name.split(".", 1)[1]},
    }


def validate_dns_evidence(evidence: Any, name: str, target: str) -> None:
    """Reject malformed saved evidence without inventing it for older nodes."""
    required = {"checked_at", "duration_ms", "records", "wildcard"}
    if (
        not isinstance(evidence, dict)
        or not required.issubset(evidence)
        or set(evidence) - required - {"aliases"}
    ):
        raise ValueError("Checkpoint contains invalid DNS evidence")
    if "aliases" in evidence:
        aliases = evidence["aliases"]
        if not isinstance(aliases, list) or len(aliases) > 16:
            raise ValueError("Checkpoint contains invalid DNS aliases")
        seen = set()
        for alias in aliases:
            if (
                not isinstance(alias, str)
                or alias != alias.strip().lower().rstrip(".")
                or not SUBDOMAIN_SPEC.validator(alias)
                or alias == name
                or not alias.endswith("." + target)
                or alias in seen
            ):
                raise ValueError("Checkpoint contains invalid or out-of-scope DNS aliases")
            seen.add(alias)
    checked_at = evidence["checked_at"]
    if not isinstance(checked_at, str) or not _UTC_TIMESTAMP.fullmatch(checked_at):
        raise ValueError("Checkpoint contains an invalid DNS observation timestamp")
    try:
        datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Checkpoint contains an invalid DNS observation timestamp") from exc
    duration = evidence["duration_ms"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        raise ValueError("Checkpoint contains an invalid DNS observation duration")
    try:
        finite_duration = math.isfinite(duration)
    except OverflowError:
        finite_duration = False
    if not finite_duration:
        raise ValueError("Checkpoint contains an invalid DNS observation duration")
    records = evidence["records"]
    if not isinstance(records, dict) or set(records) != {"A", "AAAA"}:
        raise ValueError("Checkpoint contains invalid DNS record evidence")
    for kind, version in (("A", 4), ("AAAA", 6)):
        record = records[kind]
        if not isinstance(record, dict) or set(record) != {"status", "addresses", "error"}:
            raise ValueError("Checkpoint contains invalid DNS record evidence")
        status, addresses, error = record["status"], record["addresses"], record["error"]
        if (
            not isinstance(status, str)
            or status not in _RECORD_STATUSES
            or not isinstance(addresses, list)
            or len(addresses) > 100
            or not isinstance(error, str)
            or len(error) > 180
        ):
            raise ValueError("Checkpoint contains invalid DNS record evidence")
        if (status == "answered") != bool(addresses) or (error and status != "error"):
            raise ValueError("Checkpoint contains inconsistent DNS record evidence")
        for address in addresses:
            if not isinstance(address, str) or len(address) > 45 or "%" in address:
                raise ValueError("Checkpoint contains invalid DNS evidence addresses")
            try:
                parsed = ipaddress.ip_address(address)
            except ValueError as exc:
                raise ValueError("Checkpoint contains invalid DNS evidence addresses") from exc
            if parsed.version != version:
                raise ValueError("Checkpoint DNS evidence address has the wrong record family")
    wildcard = evidence["wildcard"]
    if not isinstance(wildcard, dict) or set(wildcard) != {"status", "parent"}:
        raise ValueError("Checkpoint contains invalid wildcard evidence")
    status, parent = wildcard["status"], wildcard["parent"]
    if (
        not isinstance(status, str)
        or status not in _WILDCARD_STATUSES
        or not isinstance(parent, str)
        or not SUBDOMAIN_SPEC.validator(parent)
        or not name.endswith("." + target)
        or parent != name.split(".", 1)[1]
    ):
        raise ValueError("Checkpoint contains invalid or out-of-scope wildcard evidence")
