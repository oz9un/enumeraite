"""Deterministic target-context inference from user-supplied seeds."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


@dataclass
class ContextBundle:
    """A collection of concise observations suitable for prompt injection."""

    observations: List[str] = field(default_factory=list)

    def add(self, observation: str) -> None:
        value = observation.strip()
        if value and value not in self.observations:
            self.observations.append(value)

    def render(self) -> str:
        if not self.observations:
            return ""
        items = "\n".join(f"- {item}" for item in self.observations)
        return "Observed conventions on this target (use these; they are specific to it):\n" + items


def collect_path_context(paths: Iterable[str]) -> ContextBundle:
    values = [value.strip() for value in paths if value and value.strip().startswith("/")]
    bundle = ContextBundle()
    if len(values) < 2:
        return bundle

    segments = [[part for part in value.strip("/").split("/") if part] for value in values]
    depths = [len(parts) for parts in segments]
    common_depth = Counter(depths).most_common(1)[0][0]
    bundle.add(
        f"path depth ranges {min(depths)}-{max(depths)} segments; most common is {common_depth}"
    )

    flat = [segment for parts in segments for segment in parts]
    multiword = [segment for segment in flat if "-" in segment or "_" in segment]
    separators = Counter("kebab (-)" if "-" in value else "snake (_)" for value in multiword)
    if separators:
        bundle.add(f"multi-word segments mainly use {separators.most_common(1)[0][0]}")

    alpha = [segment for segment in flat if any(char.isalpha() for char in segment)]
    casing = Counter(_casing(segment) for segment in alpha)
    if casing:
        bundle.add(f"segment casing is mainly {casing.most_common(1)[0][0]}")

    versions = sorted(
        {match.group(0).lower() for value in flat if (match := re.fullmatch(r"v\d+", value, re.I))},
        key=lambda item: int(item[1:]),
    )
    if versions:
        bundle.add(
            f"API versioning is present ({', '.join(versions)}); adjacent versions may exist"
        )

    prefix = _common_leading(segments)
    if prefix:
        bundle.add("common path prefix: /" + "/".join(prefix))

    first_segments = Counter(parts[0].lower() for parts in segments if parts)
    if first_segments and first_segments.most_common(1)[0][1] >= 2:
        bundle.add(f"most paths begin with /{first_segments.most_common(1)[0][0]}")

    if any(
        segment.isdigit()
        or re.fullmatch(r"[0-9a-f]{8,}", segment, re.I)
        or re.fullmatch(r"\{[^{}]+\}|:[^/]+|<[^<>]+>", segment)
        for segment in flat
    ):
        bundle.add("templated or concrete identifiers appear in path segments")

    ignored = {"api", "rest", "service", "services"} | set(versions)
    nouns = Counter(
        segment.lower()
        for segment in flat
        if segment.lower() not in ignored
        and not segment.isdigit()
        and len(segment) > 2
        and re.fullmatch(r"[a-zA-Z][a-zA-Z_-]*", segment)
    )
    recurring = [name for name, count in nouns.most_common(5) if count >= 2]
    if recurring:
        bundle.add("recurring resource names: " + ", ".join(recurring))
    return bundle


def collect_subdomain_context(subdomains: Iterable[str]) -> ContextBundle:
    values = [value.strip().lower().rstrip(".") for value in subdomains if value and "." in value]
    bundle = ContextBundle()
    if len(values) < 2:
        return bundle

    apex = _infer_apex(values)
    labels = [
        value[: -(len(apex) + 1)] if value.endswith("." + apex) else value for value in values
    ]
    leftmost = [label.split(".")[0] for label in labels]
    if apex:
        bundle.add(f"shared apex: {apex}")
    if sum("-" in label for label in leftmost) >= 2:
        bundle.add("compound host labels use hyphens")

    env_terms = (
        "dev",
        "development",
        "test",
        "qa",
        "stage",
        "staging",
        "prod",
        "production",
        "uat",
        "sandbox",
    )
    found_envs = sorted(
        {term for label in leftmost for term in re.split(r"[-_.]", label) if term in env_terms}
    )
    if found_envs:
        bundle.add(
            f"environment tiers are present ({', '.join(found_envs)}); sibling tiers may exist"
        )

    tokens = [re.split(r"[-_]", label) for label in leftmost]
    leading = _common_leading(tokens)
    if leading:
        bundle.add("shared leading host token(s): " + "-".join(leading))
    if any(re.search(r"\d+$", label) for label in leftmost):
        bundle.add("numbered host instances are present")

    ignored = set(env_terms) | {"www"}
    services = Counter(
        token
        for label in leftmost
        for token in re.split(r"[-_]", label)
        if token and token not in ignored and not token.isdigit()
    )
    recurring = [name for name, count in services.most_common(5) if count >= 2]
    if recurring:
        bundle.add("recurring service labels: " + ", ".join(recurring))
    return bundle


def collect_context(items: Iterable[str], kind: str) -> ContextBundle:
    if kind == "path":
        return collect_path_context(items)
    if kind == "subdomain":
        return collect_subdomain_context(items)
    raise ValueError(f"Unsupported discovery kind: {kind!r}")


def _infer_apex(values: Sequence[str]) -> str:
    """Infer the longest shared DNS suffix with at least two labels."""
    split = [value.split(".") for value in values]
    reversed_parts = [list(reversed(parts)) for parts in split]
    common: List[str] = []
    for parts in zip(*reversed_parts, strict=False):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    if len(common) < 2:
        # Last two labels are a useful fallback; callers can override target.
        common = list(reversed(split[0][-2:]))
    return ".".join(reversed(common))


def _common_leading(sequences: Sequence[Sequence[str]]) -> List[str]:
    if not sequences:
        return []
    common: List[str] = []
    for values in zip(*sequences, strict=False):
        lowered = {value.lower() for value in values}
        if len(lowered) != 1:
            break
        common.append(values[0].lower())
    return common


def _casing(value: str) -> str:
    if value.isupper():
        return "UPPERCASE"
    if re.search(r"[a-z][A-Z]", value):
        return "camelCase"
    return "lowercase"
