"""Deterministic bounds for seed data sent to model providers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from .bounded_io import read_bounded_regular_file
from .prompt_spec import PATH_SPEC, SUBDOMAIN_SPEC

_TEMPLATE_SEGMENT = re.compile(r"^(?:\{[^{}]+\}|:[^/]+|<[^<>]+>)$")
_MAX_SEED_ITEMS = 100_000
_MAX_SEED_LINE_CHARS = 10_000
_MAX_SEED_FILE_BYTES = 64 * 1024 * 1024


def valid_seed(value: str, kind: str) -> bool:
    """Validate evidence while allowing concrete-safe OpenAPI path templates."""
    if kind == "subdomain":
        return SUBDOMAIN_SPEC.validator(value)
    if kind != "path" or not value.startswith("/"):
        return False
    concrete = "/".join(
        "123" if _TEMPLATE_SEGMENT.fullmatch(part) else part for part in value.split("/")
    )
    return PATH_SPEC.validator(concrete)


def normalize_seed(value: str, kind: str) -> str:
    """Return the canonical evidence representation without erasing templates."""
    value = value.strip()
    return (value.rstrip("/") or "/") if kind == "path" else value.lower().rstrip(".")


def bounded_seed_sample(
    values: Iterable[str],
    limit: int = 500,
    char_limit: int = 50_000,
) -> List[str]:
    """Deduplicate and evenly sample seeds under item and character budgets."""
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or isinstance(char_limit, bool)
        or not isinstance(char_limit, int)
        or limit <= 0
        or char_limit <= 0
    ):
        raise ValueError("seed limits must be positive")
    if isinstance(values, (str, bytes)):
        raise ValueError("values must be an iterable of seed strings")
    materialized: List[str] = []
    for index, value in enumerate(values):
        if index >= _MAX_SEED_ITEMS:
            raise ValueError("seed input exceeds the 100000-item limit")
        if not isinstance(value, str):
            raise ValueError("seed input must contain only strings")
        normalized = value.strip()
        if normalized:
            materialized.append(normalized)
    unique = list(dict.fromkeys(materialized))
    if len(unique) <= limit:
        selected = unique
    elif limit == 1:
        selected = unique[:1]
    else:
        last = len(unique) - 1
        selected = [unique[(index * last) // (limit - 1)] for index in range(limit)]

    result: List[str] = []
    used = 0
    for value in selected:
        additional = len(value) + (1 if result else 0)
        if used + additional > char_limit:
            continue
        result.append(value)
        used += additional
    return result


def read_seed_file(
    path: str | Path,
    *,
    max_bytes: int = _MAX_SEED_FILE_BYTES,
    max_items: int = _MAX_SEED_ITEMS,
) -> List[str]:
    """Read a bounded regular seed file without following special-file streams."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items <= 0:
        raise ValueError("max_items must be a positive integer")
    raw = read_bounded_regular_file(
        Path(path),
        max_bytes,
        not_regular_message="seed path must be a regular file",
        too_large_message="seed file exceeds the input byte limit",
    )
    text = raw.decode("utf-8")
    result: List[str] = []
    for line in text.splitlines():
        if len(line) > _MAX_SEED_LINE_CHARS:
            raise ValueError("seed file contains an oversized line")
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        result.append(item)
        if len(result) > max_items:
            raise ValueError("seed file exceeds the item limit")
    return result
