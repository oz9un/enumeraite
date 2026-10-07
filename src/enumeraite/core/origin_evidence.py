"""Recorded discovery provenance, separate from later DNS observations."""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from .passive import normalize_domain
from .prompt_spec import SUBDOMAIN_SPEC

_ORIGIN_TEXT_LIMITS = {
    "provider": 100,
    "model": 500,
    "focus": 1000,
    "plan_id": 160,
    "plan_description": 1000,
    "layout": 253,
    "word": 32,
}
_PASSIVE_TEXT_LIMITS = {
    "source": 100,
    "query": 255,
    "matched_name": 255,
    "certificate_id": 32,
    "issuer": 500,
    "not_before": 64,
    "not_after": 64,
    "entry_timestamp": 64,
    "address": 45,
    "subject_common_name": 255,
    "first_seen": 40,
}


def _timestamp(value: Any) -> None:
    if not isinstance(value, str) or len(value) > 40 or "T" not in value:
        raise ValueError("Checkpoint contains an invalid origin observation timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Checkpoint contains an invalid origin observation timestamp") from exc
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("Checkpoint origin observation timestamps must be UTC")


def _text_fields(value: dict, limits: dict[str, int]) -> None:
    for name, limit in limits.items():
        if name in value and (
            not isinstance(value[name], str) or not 0 < len(value[name]) <= limit
        ):
            raise ValueError("Checkpoint contains invalid origin text metadata")


def validate_origin_evidence(evidence: Any, name: str, target: str) -> None:
    """Validate optional saved provenance without backfilling legacy observations."""
    allowed = {
        *_ORIGIN_TEXT_LIMITS,
        "queued_at",
        "passive",
        "evidence",
        "confidence",
        "batch",
        "numeric_ceiling",
    }
    if not isinstance(evidence, dict) or set(evidence) - allowed:
        raise ValueError("Checkpoint contains invalid origin evidence")
    _text_fields(evidence, _ORIGIN_TEXT_LIMITS)
    if "queued_at" in evidence:
        _timestamp(evidence["queued_at"])
    if "evidence" in evidence:
        names = evidence["evidence"]
        if (
            not isinstance(names, list)
            or len(names) > 60
            or any(
                not isinstance(value, str)
                or not SUBDOMAIN_SPEC.validator(value)
                or value != value.lower().rstrip(".")
                or not value.endswith("." + target)
                for value in names
            )
            or len(set(names)) != len(names)
        ):
            raise ValueError("Checkpoint contains invalid origin evidence names")
    if "confidence" in evidence:
        score = evidence["confidence"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= score <= 1
            or not math.isfinite(score)
        ):
            raise ValueError("Checkpoint contains an invalid proposal confidence")
    if "batch" in evidence and (type(evidence["batch"]) is not int or evidence["batch"] < 1):
        raise ValueError("Checkpoint contains an invalid origin batch")
    if "numeric_ceiling" in evidence and (
        type(evidence["numeric_ceiling"]) is not int or not 1 <= evidence["numeric_ceiling"] <= 9999
    ):
        raise ValueError("Checkpoint contains an invalid origin numeric ceiling")
    if "passive" not in evidence:
        return
    records = evidence["passive"]
    if not isinstance(records, list) or len(records) > 30:
        raise ValueError("Checkpoint contains invalid passive origin records")
    counts: Counter[str] = Counter()
    for record in records:
        if (
            not isinstance(record, dict)
            or not {"source", "collected_at", "query", "matched_name"} <= record.keys()
            or set(record) - {*_PASSIVE_TEXT_LIMITS, "collected_at"}
        ):
            raise ValueError("Checkpoint contains invalid passive origin metadata")
        _text_fields(record, _PASSIVE_TEXT_LIMITS)
        _timestamp(record["collected_at"])
        if "first_seen" in record:
            _timestamp(record["first_seen"])
        counts[record["source"]] += 1
        if counts[record["source"]] > 3:
            raise ValueError("Checkpoint contains too many records for one passive origin")
        try:
            matched = normalize_domain(record["matched_name"].removeprefix("*."))
        except ValueError as exc:
            raise ValueError("Checkpoint contains an invalid matched source name") from exc
        if matched != name:
            raise ValueError("Checkpoint passive origin does not match its hostname")
        if record["source"] in {"crt.sh", "subfinder", "merklemap"}:
            prefix = (
                "%."
                if record["source"] == "crt.sh"
                else ("*." if record["source"] == "merklemap" else "")
            )
            query = prefix + target
            if record["query"] != query:
                raise ValueError("Checkpoint passive origin query does not match its scope")
        if "certificate_id" in record and not re.fullmatch(
            r"[0-9]{1,32}", record["certificate_id"]
        ):
            raise ValueError("Checkpoint contains an invalid certificate record ID")
        if "address" in record:
            try:
                ipaddress.ip_address(record["address"])
            except ValueError as exc:
                raise ValueError(
                    "Checkpoint contains an invalid historical source address"
                ) from exc


def passive_collectors(node: dict, sources: dict) -> list[str]:
    """Combine recorded samples with exact legacy collector membership."""
    origin = node.get("origin_evidence")
    records = origin.get("passive", []) if isinstance(origin, dict) else []
    collectors = {
        item["source"]
        for item in records
        if isinstance(item, dict) and isinstance(item.get("source"), str)
    }
    collectors.update(
        source
        for source, names in sources.items()
        if isinstance(source, str) and isinstance(names, list) and node.get("value") in names
    )
    return sorted(collectors)
