"""Fast, side-effect-free discovery kind and provider detection."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import socket
from typing import Iterable, Tuple

from ..core.config import Config
from ..core.prompt_spec import SUBDOMAIN_SPEC
from ..core.url_safety import parse_http_url

PROVIDER_PREFERENCE = (
    "anthropic",
    "openai",
    "openrouter",
    "deepseek",
    "minimax",
    "groq",
    "together",
    "ollama",
    "vllm",
    "llamacpp",
    "local",
    "claude",
    "huggingface",
)
HOSTED_PROVIDERS = {
    "anthropic",
    "claude",
    "openai",
    "openrouter",
    "deepseek",
    "minimax",
    "groq",
    "together",
}


def detect_kind(items: Iterable[str]) -> Tuple[str, float]:
    values = [item.strip() for item in _bounded_string_items(items) if item.strip()]
    if not values:
        return "path", 0.0
    paths = sum(value.startswith("/") for value in values)
    domains = sum(SUBDOMAIN_SPEC.validator(value) for value in values)
    if domains > paths:
        return "subdomain", domains / len(values)
    return "path", paths / len(values)


def detect_provider(config: Config) -> str | None:
    preferred = config.default_provider
    ordered = ([preferred] if preferred in PROVIDER_PREFERENCE else []) + [
        name for name in PROVIDER_PREFERENCE if name != preferred
    ]
    for name in ordered:
        if _provider_usable(config, name):
            return name
    return None


def _provider_usable(config: Config, name: str) -> bool:
    provider_config = config.get_provider_config(name)
    if not provider_config:
        return False
    if name in HOSTED_PROVIDERS and not provider_config.get("api_key"):
        return False
    if name == "huggingface":
        return _hf_usable()
    if name in {"ollama", "local", "vllm", "llamacpp"}:
        base_url = provider_config.get("base_url", "")
        if not isinstance(base_url, str):
            return False
        endpoint = _local_endpoint(base_url)
        return bool(endpoint and local_server_reachable(*endpoint))
    return True


def local_server_reachable(host: str, port: int, timeout: float = 0.15) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def infer_apex(items: Iterable[str]) -> str:
    materialized = _bounded_string_items(items)
    values = list(
        dict.fromkeys(
            item.lower().rstrip(".")
            for item in materialized
            if item and SUBDOMAIN_SPEC.validator(item)
        )
    )
    # One host does not reveal where the registrable/delegated zone begins;
    # guessing the last two labels is wrong for public suffixes such as co.uk.
    if len(values) < 2:
        return ""
    reversed_parts = [list(reversed(value.split("."))) for value in values]
    common = []
    for parts in zip(*reversed_parts, strict=False):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    if len(common) < 2:
        return ""
    return ".".join(reversed(common))


def _bounded_string_items(items: Iterable[str]) -> list[str]:
    if isinstance(items, (str, bytes)):
        raise ValueError("items must be an iterable of strings")
    result: list[str] = []
    for index, item in enumerate(items):
        if index >= 100_000:
            raise ValueError("items exceed the 100000-item detection limit")
        if not isinstance(item, str):
            raise ValueError("items must contain only strings")
        result.append(item)
    return result


def _hf_usable() -> bool:
    minimums = {
        "torch": (2, 13, 0),
        "transformers": (5, 10, 1),
        "accelerate": (0, 26, 0),
    }
    return all(
        importlib.util.find_spec(package) is not None
        and _installed_version_at_least(package, minimum)
        for package, minimum in minimums.items()
    )


def _installed_version_at_least(package: str, minimum: tuple[int, ...]) -> bool:
    try:
        raw = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    match = re.match(r"^(\d+(?:\.\d+)*)", raw)
    if not match:
        return False
    current = tuple(int(part) for part in match.group(1).split("."))
    width = max(len(current), len(minimum))
    return current + (0,) * (width - len(current)) >= minimum + (0,) * (width - len(minimum))


def _local_endpoint(base_url: str) -> tuple[str, int] | None:
    try:
        parsed = parse_http_url(base_url, label="local provider base URL")
        hostname = parsed.hostname
        if hostname is None:
            return None
        return hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None
