"""Deterministic benchmark dataset loaders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Literal, Tuple, cast
from urllib.parse import urlunsplit

import yaml

from ..core.bounded_io import read_bounded_regular_file
from ..core.prompt_spec import PATH_SPEC, SUBDOMAIN_SPEC
from ..core.seeds import valid_seed
from ..core.url_safety import parse_http_url

Kind = Literal["path", "subdomain"]
_MAX_DATASET_FILE_BYTES = 64 * 1024 * 1024


@dataclass
class BenchCase:
    """One target with visible seed and held-out ground truth."""

    target: str
    kind: Kind
    known: List[str]
    hidden: List[str]
    universe: List[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("Benchmark target must be a non-empty string")
        if self.kind not in {"path", "subdomain"}:
            raise ValueError(f"Unsupported benchmark kind: {self.kind!r}")
        if not all(
            isinstance(values, list) and all(isinstance(item, str) for item in values)
            for values in (self.known, self.hidden, self.universe)
        ):
            raise ValueError("Benchmark known, hidden, and universe values must be string lists")
        if not isinstance(self.meta, dict):
            raise ValueError("Benchmark metadata must be an object")
        self.known = _unique(_normalize(item, self.kind) for item in self.known)
        self.hidden = _unique(_normalize(item, self.kind) for item in self.hidden)
        self.universe = _unique(_normalize(item, self.kind) for item in self.universe)
        known_identities = {_identity(item, self.kind) for item in self.known}
        hidden_identities = {_identity(item, self.kind) for item in self.hidden}
        if len(hidden_identities) != len(self.hidden):
            raise ValueError("Held-out ground truth contains duplicate route shapes")
        leaked = known_identities & hidden_identities
        if leaked:
            raise ValueError(f"Held-out route shape leaked into seed: {sorted(leaked)[:3]}")
        if not self.hidden:
            raise ValueError("A benchmark case must contain at least one held-out item")
        universe_identities = {_identity(item, self.kind) for item in self.universe}
        if not ((known_identities | hidden_identities) <= universe_identities):
            raise ValueError("Known and hidden items must be members of the case universe")


def _seeded_split(
    items: Iterable[str],
    seed_frac: float,
    salt: str,
    max_seed: int | None = None,
) -> Tuple[List[str], List[str]]:
    """Split deterministically by salted SHA-256 ordering."""
    if isinstance(items, (str, bytes)):
        raise ValueError("benchmark items must be an iterable of strings, not a string")
    if (
        isinstance(seed_frac, bool)
        or not isinstance(seed_frac, (int, float))
        or not 0 < seed_frac < 1
    ):
        raise ValueError("seed_frac must be between 0 and 1")
    if not isinstance(salt, str) or not salt:
        raise ValueError("benchmark salt must be a non-empty string")
    if max_seed is not None and (
        isinstance(max_seed, bool) or not isinstance(max_seed, int) or max_seed <= 0
    ):
        raise ValueError("max_seed must be a positive integer")
    materialized = list(items)
    if not all(isinstance(item, str) for item in materialized):
        raise ValueError("benchmark items must contain only strings")
    unique = _unique(item.strip() for item in materialized if item.strip())
    if len(unique) < 2:
        raise ValueError("At least two unique items are required for a benchmark split")
    ordered = sorted(
        unique,
        key=lambda item: hashlib.sha256(f"{salt}:{item}".encode("utf-8")).hexdigest(),
    )
    seed_count = max(1, min(len(ordered) - 1, int(len(ordered) * seed_frac)))
    if max_seed is not None:
        seed_count = max(1, min(seed_count, max_seed))
    return ordered[:seed_count], ordered[seed_count:]


def load_openapi_dataset(
    specs_dir: str | Path,
    seed_frac: float = 0.2,
    salt: str = "enumeraite-v1",
    limit: int | None = None,
    max_seed: int | None = None,
) -> List[BenchCase]:
    """Load paths from JSON OpenAPI documents below ``specs_dir``."""
    _validate_loader_limits(limit, max_seed)
    root = Path(specs_dir)
    cases: List[BenchCase] = []
    spec_paths = sorted(
        path for path in root.rglob("*") if path.suffix.lower() in {".json", ".yaml", ".yml"}
    )
    for spec_path in spec_paths:
        try:
            raw_bytes, source_sha256 = _read_dataset_file(spec_path)
            raw = raw_bytes.decode("utf-8")
            data = json.loads(raw) if spec_path.suffix.lower() == ".json" else yaml.safe_load(raw)
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        paths = data.get("paths")
        if not isinstance(paths, dict) or len(paths) < 4:
            continue
        items = _unique_path_shapes(
            path for path in paths if isinstance(path, str) and valid_seed(path, "path")
        )
        if len(items) < 4:
            continue
        relative_source = spec_path.relative_to(root).as_posix()
        split_salt = f"{salt}:{relative_source}"
        known, hidden = _seeded_split(items, seed_frac, split_salt, max_seed=max_seed)
        raw_info = data.get("info")
        raw_servers = data.get("servers")
        info = cast(dict[str, Any], raw_info) if isinstance(raw_info, dict) else {}
        servers = cast(list[Any], raw_servers) if isinstance(raw_servers, list) else []
        target = _openapi_target(data, info, servers, spec_path)
        cases.append(
            BenchCase(
                target=target,
                kind="path",
                known=known,
                hidden=hidden,
                universe=items,
                meta={
                    "source": str(spec_path),
                    "source_sha256": source_sha256,
                    "title": info.get("title", ""),
                    "spec_version": data.get("openapi", data.get("swagger", "")),
                    "split_salt": split_salt,
                },
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def load_subdomain_dataset(
    truth_dir: str | Path,
    seed_frac: float = 0.5,
    salt: str = "enumeraite-v1",
    limit: int | None = None,
    max_seed: int | None = None,
) -> List[BenchCase]:
    """Load one complete ``<apex>.txt`` truth set per target."""
    _validate_loader_limits(limit, max_seed)
    root = Path(truth_dir)
    cases: List[BenchCase] = []
    for truth_path in sorted(root.glob("*.txt")):
        apex = truth_path.stem.lower().rstrip(".")
        if not SUBDOMAIN_SPEC.validator(apex):
            continue
        try:
            raw_bytes, source_sha256 = _read_dataset_file(truth_path)
            items = [
                normalized
                for line in raw_bytes.decode("utf-8").splitlines()
                if line.strip()
                and not line.lstrip().startswith("#")
                and SUBDOMAIN_SPEC.validator(normalized := line.strip().lower().rstrip("."))
                and normalized.endswith("." + apex)
            ]
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if len(set(items)) < 2:
            continue
        split_salt = f"{salt}:{truth_path.name}"
        known, hidden = _seeded_split(items, seed_frac, split_salt, max_seed=max_seed)
        cases.append(
            BenchCase(
                target=apex,
                kind="subdomain",
                known=known,
                hidden=hidden,
                universe=items,
                meta={
                    "source": str(truth_path),
                    "source_sha256": source_sha256,
                    "split_salt": split_salt,
                },
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def _normalize(value: str, kind: Kind) -> str:
    value = str(value).strip()
    return (value.rstrip("/") or "/") if kind == "path" else value.lower().rstrip(".")


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _unique_path_shapes(values: Iterable[str]) -> List[str]:
    """Choose one stable representative for each equivalent router shape."""
    result: dict[str, str] = {}
    for value in sorted(_unique(str(item).strip() for item in values if str(item).strip())):
        result.setdefault(PATH_SPEC.identity(value), value)
    return list(result.values())


def _identity(value: str, kind: Kind) -> str:
    return PATH_SPEC.identity(value) if kind == "path" else value


def _read_dataset_file(path: Path) -> tuple[bytes, str]:
    raw = read_bounded_regular_file(
        path,
        _MAX_DATASET_FILE_BYTES,
        not_regular_message="dataset source must be a bounded regular file",
        too_large_message="dataset source exceeds the input byte limit",
    )
    return raw, hashlib.sha256(raw).hexdigest()


def _validate_loader_limits(limit: int | None, max_seed: int | None) -> None:
    for name, value in (("limit", limit), ("max_seed", max_seed)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")


def _openapi_target(
    data: dict[str, Any], info: dict[str, Any], servers: list[Any], spec_path: Path
) -> str:
    """Use an OpenAPI 3 server or Swagger 2 host/basePath when available."""
    if servers and isinstance(servers[0], dict) and servers[0].get("url"):
        target = _safe_openapi_url(str(servers[0]["url"]))
        if target:
            return target
    host = data.get("host")
    if isinstance(host, str) and host.strip():
        schemes = data.get("schemes")
        scheme = (
            str(schemes[0]) if isinstance(schemes, list) and schemes and schemes[0] else "https"
        )
        base_path = str(data.get("basePath") or "").strip()
        if base_path and not base_path.startswith("/"):
            base_path = "/" + base_path
        target = _safe_openapi_url(f"{scheme}://{host.strip()}{base_path}")
        if target:
            return target
    return str(info.get("title") or spec_path.stem)


def _safe_openapi_url(value: str) -> str:
    """Strip query/fragment data and refuse credential-bearing absolute URLs."""
    if "://" not in value:
        return value[:2_048]
    try:
        parsed = parse_http_url(value, label="OpenAPI server URL")
    except ValueError:
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))[:2_048]
