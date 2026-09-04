"""Deterministic, seed-derived proposals for the hybrid discovery pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from .prompt_spec import PATH_SPEC, SUBDOMAIN_SPEC


@dataclass(frozen=True)
class StructuralCandidate:
    """A candidate justified by an observed structural relationship."""

    value: str
    confidence: float
    rationale: str


def structural_candidates(
    known: Iterable[str], kind: str, target: str = "", count: int = 100
) -> List[StructuralCandidate]:
    """Infer high-evidence gaps without relying on a generic wordlist."""
    if isinstance(known, (str, bytes)):
        raise ValueError("known must be an iterable of seed strings, not a string")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 5_000:
        raise ValueError("count must be an integer between 0 and 5000")
    if not isinstance(target, str):
        raise ValueError("target must be a string")
    known_values = list(known)
    if not all(isinstance(value, str) for value in known_values):
        raise ValueError("known must contain only seed strings")
    if count == 0:
        return []
    if kind == "path":
        proposals = _path_candidates(known_values)
        spec = PATH_SPEC
    elif kind == "subdomain":
        normalized_target = target.lower().rstrip(".")
        if target and not SUBDOMAIN_SPEC.validator(normalized_target):
            raise ValueError("subdomain target must be a valid DNS apex")
        scoped_values = (
            [value for value in known_values if _within_apex(value, normalized_target)]
            if normalized_target
            else known_values
        )
        proposals = _subdomain_candidates(scoped_values, normalized_target)
        known_values = scoped_values
        spec = SUBDOMAIN_SPEC
    else:
        raise ValueError("kind must be 'path' or 'subdomain'")

    known_identities = {spec.identity(value) for value in known_values}
    seen = set(known_identities)
    result: List[StructuralCandidate] = []
    for candidate in proposals:
        identity = spec.identity(candidate.value)
        if identity in seen or not spec.validator(candidate.value):
            continue
        if kind == "subdomain" and target and not _within_apex(candidate.value, target):
            continue
        seen.add(identity)
        result.append(candidate)
        if len(result) >= count:
            break
    return result


def _path_candidates(known: Iterable[str]) -> List[StructuralCandidate]:
    values = [value.rstrip("/") or "/" for value in known if value.startswith("/")]
    proposals: List[StructuralCandidate] = []

    # Missing immediate parents are strong evidence: a known item/action route
    # almost always implies that the router also has a collection or item root.
    for value in values:
        parts = value.strip("/").split("/")
        for cut in range(len(parts) - 1, 0, -1):
            parent_parts = parts[:cut]
            if _prefix_only(parent_parts[-1]):
                break
            removed = parts[cut] if cut < len(parts) else ""
            parent_is_item = bool(re.fullmatch(r"\{[^{}]+\}|:[^/]+|<[^<>]+>|\d+", parent_parts[-1]))
            removed_is_item = bool(re.fullmatch(r"\{[^{}]+\}|:[^/]+|<[^<>]+>|\d+", removed))
            if parent_is_item:
                confidence = 0.96
            elif removed_is_item:
                confidence = 0.94
            elif cut == len(parts) - 1:
                confidence = 0.86
            else:
                confidence = 0.72
            proposals.append(
                StructuralCandidate(
                    _concretize_path("/" + "/".join(parent_parts)),
                    confidence,
                    "missing ancestor of observed route",
                )
            )

    # Complete a version × top-level-resource grid learned entirely from the
    # seed. This captures v2/users when v2/orders and v1/users are visible.
    version_roots: dict[tuple[str, ...], set[str]] = {}
    resources: dict[tuple[str, ...], set[str]] = {}
    for value in values:
        parts = value.strip("/").split("/")
        for index, part in enumerate(parts):
            if re.fullmatch(r"v\d+(?:\.\d+)?", part, re.IGNORECASE) and index + 1 < len(parts):
                root = tuple(parts[:index])
                version_roots.setdefault(root, set()).add(part)
                resources.setdefault(root, set()).add(parts[index + 1])
                break
    for root in sorted(version_roots):
        versions = version_roots[root]
        if len(versions) < 2:
            continue
        for version in sorted(versions):
            for resource in sorted(resources[root]):
                value = "/" + "/".join((*root, version, resource))
                proposals.append(
                    StructuralCandidate(value, 0.82, "missing version-resource grid cell")
                )
    return proposals


def _subdomain_candidates(known: Iterable[str], target: str) -> List[StructuralCandidate]:
    values = [value.lower().rstrip(".") for value in known if "." in value]
    apex = target.lower().rstrip(".") or _infer_apex(values)
    proposals: List[StructuralCandidate] = []
    environments = {
        "dev",
        "development",
        "test",
        "qa",
        "stage",
        "staging",
        "uat",
        "prod",
        "production",
        "sandbox",
    }
    observed_environments = {
        token
        for value in values
        for token in re.split(r"[-_.]", _relative(value, apex))
        if token in environments
    }
    # Emit the higher-confidence environment grid before lower-confidence
    # adjacent-instance extrapolation. The LLM hybrid ranks by confidence, and
    # the deterministic baseline should preserve the same evidence ordering.
    for value in values:
        relative = _relative(value, apex)
        labels = relative.split(".")
        tokens = labels[0].split("-")
        for index, token in enumerate(tokens):
            if token not in observed_environments:
                continue
            for replacement in sorted(observed_environments - {token}):
                changed = tokens[:]
                changed[index] = replacement
                changed_relative = ".".join(["-".join(changed), *labels[1:]])
                proposals.append(
                    StructuralCandidate(
                        f"{changed_relative}.{apex}",
                        0.84,
                        "missing observed environment grid cell",
                    )
                )
    for value in values:
        relative = _relative(value, apex)
        labels = relative.split(".")
        numbered = re.search(r"(\d+)$", labels[0])
        if numbered:
            number = int(numbered.group(1))
            width = len(numbered.group(1))
            adjacent_numbers = ([number - 1] if number > 1 else []) + [number + 1]
            for adjacent in adjacent_numbers:
                changed_label = labels[0][: numbered.start(1)] + str(adjacent).zfill(width)
                changed_relative = ".".join([changed_label, *labels[1:]])
                proposals.append(
                    StructuralCandidate(
                        f"{changed_relative}.{apex}",
                        0.8,
                        "adjacent observed host instance",
                    )
                )
    return proposals


def _prefix_only(value: str) -> bool:
    return value in {"api", "rest", "service", "services"} or bool(
        re.fullmatch(r"v\d+(?:\.\d+)?", value, re.IGNORECASE)
    )


def _concretize_path(value: str) -> str:
    return "/".join(
        "123" if re.fullmatch(r"\{[^{}]+\}|:[^/]+|<[^<>]+>", part) else part
        for part in value.split("/")
    )


def _infer_apex(values: List[str]) -> str:
    if len(values) < 2:
        return ""
    reversed_parts = [list(reversed(value.split("."))) for value in values]
    common: List[str] = []
    for parts in zip(*reversed_parts, strict=False):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    return ".".join(reversed(common)) if len(common) >= 2 else ""


def _relative(value: str, apex: str) -> str:
    return value[: -(len(apex) + 1)] if apex and value.endswith("." + apex) else value


def _within_apex(value: str, target: str) -> bool:
    candidate = value.lower().rstrip(".")
    apex = target.lower().rstrip(".")
    return candidate != apex and candidate.endswith("." + apex)
