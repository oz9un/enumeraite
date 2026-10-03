"""Read-only discovery accounting by the method that first queued each name."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_METHODS = ("seed", "passive", "bootstrap", "numeric", "pattern", "model", "other")
_STATUSES = ("pending", "verified", "miss", "wildcard", "inconclusive")
_NUMERIC_REASONS = ("numeric-slot hypothesis", "Systematic numeric family enumeration")


def discovery_method_counts(nodes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Count unique nodes, not lookup packets or retries; never infer absence.

    The session's deduplicated nodes retain first-enqueue ownership. These are
    descriptive counts, not exclusive attribution or probabilities. A retried
    node may be both checked previously and pending now. The caller supplies
    validated nodes; unknown origins are bounded to one legacy/other bucket.
    """
    rows: dict[str, dict[str, Any]] = {}
    for node in nodes:
        source = node.get("source")
        method = source if source in _METHODS and source != "numeric" else "other"
        reason = node.get("reason")
        if source == "pattern" and isinstance(reason, str) and reason.startswith(_NUMERIC_REASONS):
            method = "numeric"
        row = rows.setdefault(
            method,
            {"method": method, "candidates": 0, "checked": 0, **dict.fromkeys(_STATUSES, 0)},
        )
        row["candidates"] += 1
        attempts = node.get("attempts", 0)
        if type(attempts) is int and attempts > 0:
            row["checked"] += 1
        status = node.get("status")
        if status in _STATUSES:
            row[status] += 1
    return [rows[method] for method in _METHODS if method in rows]
