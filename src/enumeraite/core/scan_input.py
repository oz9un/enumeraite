"""Resolve a root domain or seed hostname into an explicit DNS scan boundary.

The packaged Public Suffix List is read locally, including private suffixes.
No first-use network request or user-writable suffix cache influences scope.
Unknown suffixes stay at the exact entered boundary unless the user explicitly
provides a scope; a hostname is never promoted by counting labels alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import tldextract

from .passive import normalize_domain

_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)


@dataclass(frozen=True)
class ScanInput:
    target: str
    seeds: list[str]
    input: str
    mode: Literal["seeded", "passive"]
    passive_default: bool


def validate_seed_depth(target: str, seeds: Sequence[str], max_depth: int) -> None:
    """Reject starting evidence that the selected DNS depth would silently skip."""
    if type(max_depth) is not int or not 1 <= max_depth <= 10:
        raise ValueError("Maximum depth must be an integer between 1 and 10")
    target = normalize_domain(target)
    required = 0
    for value in seeds:
        seed = normalize_domain(value)
        if not seed.endswith("." + target):
            raise ValueError("Every seed must be a subdomain within the scan scope")
        required = max(required, len(seed[: -(len(target) + 1)].split(".")))
    if required > max_depth:
        raise ValueError(
            f"Starting subdomain exceeds max-depth (requires {required}, configured {max_depth}); "
            "increase depth or narrow scope"
        )


def _boundary(value: str) -> tuple[str, str]:
    normalized = normalize_domain(value)
    extracted = _EXTRACT(normalized)
    if extracted.suffix and not extracted.domain:
        raise ValueError("Scan scope must be a domain below a public or private suffix")
    registrable = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else normalized
    return normalized, registrable


def resolve_scan_input(
    value: str, *, scope: str | None = None, seeds: list[str] | None = None
) -> ScanInput:
    """Use bare domains for passive-first scans and hostnames for seeded scans.

    ``scope`` pins the boundary exactly, including delegated/private DNS zones.
    Every explicit seed must be a strict descendant of that boundary. The input
    itself becomes the first seed when it is below the selected boundary.
    """
    original, inferred = _boundary(value)
    target = _boundary(scope)[0] if scope is not None else inferred
    if original != target and not original.endswith("." + target):
        raise ValueError("Scan input must equal the scope or be a subdomain within it")
    if seeds is not None and (not isinstance(seeds, list) or len(seeds) > 1000):
        raise ValueError("Seeds must be a list of at most 1000 subdomain names")
    normalized_seeds = []
    seen = set()
    values = ([original] if original != target else []) + (seeds or [])
    for seed in values:
        normalized = normalize_domain(seed)
        if not normalized.endswith("." + target):
            raise ValueError("Every seed must be a subdomain within the scan scope")
        if normalized not in seen:
            seen.add(normalized)
            normalized_seeds.append(normalized)
    if len(normalized_seeds) > 1000:
        raise ValueError("At most 1000 unique seeds may be supplied, including the input hostname")
    return ScanInput(
        target=target,
        seeds=normalized_seeds,
        input=original,
        mode="seeded" if normalized_seeds else "passive",
        passive_default=not normalized_seeds,
    )
