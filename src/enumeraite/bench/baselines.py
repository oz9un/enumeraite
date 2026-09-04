"""Non-AI controls used by the benchmark harness."""

from __future__ import annotations

import re
from typing import Iterable, List

from ..core.candidate_pool import structural_candidates
from ..core.prompt_spec import PATH_SPEC, SUBDOMAIN_SPEC

PATH_ACTIONS = (
    "list",
    "create",
    "update",
    "delete",
    "get",
    "search",
    "count",
    "export",
)
PATH_SUBRESOURCES = ("id", "status", "history", "config", "settings", "logs", "meta")
SUBDOMAIN_PREFIXES = (
    "api",
    "admin",
    "app",
    "auth",
    "cdn",
    "dev",
    "staging",
    "prod",
    "test",
)
ENVIRONMENTS = ("dev", "test", "qa", "staging", "prod", "sandbox")


def wordlist_baseline(
    words: Iterable[str], apex: str = "", kind: str = "path", count: int | None = None
) -> List[str]:
    if isinstance(words, (str, bytes)):
        raise ValueError("words must be an iterable of strings, not a string")
    if kind not in {"path", "subdomain"}:
        raise ValueError("kind must be 'path' or 'subdomain'")
    if not isinstance(apex, str):
        raise ValueError("apex must be a string")
    if count is not None and (
        isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 5_000
    ):
        raise ValueError("count must be an integer between 0 and 5000")
    if count == 0:
        return []
    materialized = list(words)
    if not all(isinstance(word, str) for word in materialized):
        raise ValueError("words must contain only strings")
    normalized_apex = apex.lower().rstrip(".")
    if kind == "subdomain" and not SUBDOMAIN_SPEC.validator(normalized_apex):
        raise ValueError("subdomain wordlist baseline requires a valid apex")
    spec = SUBDOMAIN_SPEC if kind == "subdomain" else PATH_SPEC
    result: List[str] = []
    seen: set[str] = set()
    for raw in materialized:
        word = raw.strip().strip("/.")
        if not word or word.startswith("#") or " " in word:
            continue
        if kind == "subdomain":
            value = (
                word
                if word.lower().endswith("." + normalized_apex)
                else f"{word}.{normalized_apex}"
            )
        else:
            value = "/" + word
        if not spec.validator(value):
            continue
        value = spec.normalizer(value)
        identity = spec.identity(value)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(value)
        if count is not None and len(result) >= count:
            break
    return result


def mutation_baseline(known: Iterable[str], count: int, kind: str, target: str = "") -> List[str]:
    """Generate the combined structural-plus-mutation reference control."""
    return _mutation_control(known, count, kind, target, include_structural=True)


def mutation_only_baseline(
    known: Iterable[str], count: int, kind: str, target: str = ""
) -> List[str]:
    """Generate lexical mutations without structural gap completion."""
    return _mutation_control(known, count, kind, target, include_structural=False)


def structural_baseline(known: Iterable[str], count: int, kind: str, target: str = "") -> List[str]:
    """Generate only seed-derived structural gap candidates."""
    values, normalized_target = _validated_control_inputs(known, count, kind, target)
    return [
        candidate.value
        for candidate in structural_candidates(
            values,
            kind,
            target=normalized_target,
            count=count,
        )
    ]


def _mutation_control(
    known: Iterable[str],
    count: int,
    kind: str,
    target: str,
    *,
    include_structural: bool,
) -> List[str]:
    values, target = _validated_control_inputs(known, count, kind, target)
    if count == 0:
        return []
    structural = (
        [
            candidate.value
            for candidate in structural_candidates(values, kind, target=target, count=count)
        ]
        if include_structural
        else []
    )
    if kind == "subdomain":
        mutations = _subdomain_mutations(values, count, target)
    else:
        mutations = _path_mutations(values, count)
    spec = SUBDOMAIN_SPEC if kind == "subdomain" else PATH_SPEC
    known_identities = {spec.identity(value) for value in values}
    return _take_unique(
        (
            candidate
            for candidate in [*structural, *mutations]
            if spec.validator(candidate)
            and spec.identity(candidate) not in known_identities
            and (
                kind != "subdomain"
                or not target
                or candidate.lower().rstrip(".").endswith("." + target)
            )
        ),
        count,
    )


def _validated_control_inputs(
    known: Iterable[str], count: int, kind: str, target: str
) -> tuple[List[str], str]:
    if isinstance(known, (str, bytes)):
        raise ValueError("known must be an iterable of strings, not a string")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 5_000:
        raise ValueError("count must be an integer between 0 and 5000")
    if kind not in {"path", "subdomain"}:
        raise ValueError("kind must be 'path' or 'subdomain'")
    if not isinstance(target, str):
        raise ValueError("target must be a string")
    values = list(known)
    if not all(isinstance(value, str) for value in values):
        raise ValueError("known must contain only strings")
    if kind == "subdomain" and target and not SUBDOMAIN_SPEC.validator(target):
        raise ValueError("subdomain target must be a valid DNS apex")
    if kind == "subdomain":
        target = target.lower().rstrip(".")
    return values, target


def _path_mutations(known: Iterable[str], count: int) -> List[str]:
    result: List[str] = []
    for path in known:
        # OpenAPI seeds commonly contain router templates. A control that emits
        # literal ``{userId}`` requests is artificially weak, so instantiate
        # each parameter before applying ordinary path mutations.
        clean = _concretize_path(path.rstrip("/") or "/")
        parent, _, leaf = clean.rpartition("/")
        parent = parent or ""
        if leaf and not _looks_like_identifier(leaf):
            if leaf.endswith("s") and len(leaf) > 1:
                result.append(f"{parent}/{leaf[:-1]}")
            else:
                result.append(f"{parent}/{leaf}s")
        version = re.search(r"(?<![a-z])v(\d+)(?!\d)", clean, re.I)
        if version:
            number = int(version.group(1))
            for adjacent in sorted({max(0, number - 1), number + 1}):
                result.append(clean[: version.start(1)] + str(adjacent) + clean[version.end(1) :])
        for action in PATH_ACTIONS:
            result.append(f"{clean}/{action}")
        for resource in PATH_SUBRESOURCES:
            result.append(f"{clean}/{resource}")
        if len(result) >= count * 2:
            break
    return _take_unique(result, count)


def _subdomain_mutations(known: Iterable[str], count: int, target: str) -> List[str]:
    values = [item.lower().rstrip(".") for item in known]
    apex = target or _infer_apex(values)
    result: List[str] = []
    for fqdn in values:
        prefix = (
            fqdn[: -(len(apex) + 1)] if apex and fqdn.endswith("." + apex) else fqdn.split(".")[0]
        )
        leftmost = prefix.split(".")[0]
        tokens = leftmost.split("-")
        for index, token in enumerate(tokens):
            if token in ENVIRONMENTS:
                for environment in ENVIRONMENTS:
                    changed = tokens[:]
                    changed[index] = environment
                    result.append(f"{'-'.join(changed)}.{apex}")
        numbered = re.search(r"(\d+)$", leftmost)
        if numbered:
            number = int(numbered.group(1))
            width = len(numbered.group(1))
            adjacent_numbers = ([number - 1] if number > 1 else []) + [number + 1]
            for adjacent in adjacent_numbers:
                label = leftmost[: numbered.start(1)] + str(adjacent).zfill(width)
                result.append(f"{label}.{apex}")
        for common in SUBDOMAIN_PREFIXES:
            result.append(f"{common}.{apex}")
        if len(result) >= count * 2:
            break
    return _take_unique(result, count)


def wordlist_set(paths: Iterable[str]) -> set[str]:
    return {
        line.strip().lower().strip("/.")
        for line in paths
        if line.strip() and not line.lstrip().startswith("#")
    }


def _take_unique(values: Iterable[str], count: int) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))[:count]


def _infer_apex(values: List[str]) -> str:
    if len(values) < 2:
        return ""
    reversed_parts = [list(reversed(value.split("."))) for value in values]
    common: List[str] = []
    # Deliberately stop at the shortest hostname while finding the shared suffix.
    for parts in zip(*reversed_parts, strict=False):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    return ".".join(reversed(common)) if len(common) >= 2 else ""


def _concretize_path(value: str) -> str:
    return "/".join(
        "123" if re.fullmatch(r"\{[^{}]+\}|:[^/]+|<[^<>]+>", part) else part
        for part in value.split("/")
    )


def _looks_like_identifier(value: str) -> bool:
    return bool(
        re.fullmatch(r"\d+", value)
        or re.fullmatch(r"[0-9a-f]{8,}", value, re.IGNORECASE)
        or re.fullmatch(r"[0-9a-f-]{32,}", value, re.IGNORECASE)
    )
