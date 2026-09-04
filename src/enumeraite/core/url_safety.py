"""Shared parsing rules for URLs that may carry credentials or trigger requests."""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from urllib.parse import SplitResult, urlsplit


def parse_http_url(value: str, *, label: str = "URL") -> SplitResult:
    """Parse a bounded HTTP(S) URL and reject ambiguous authority syntax."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a valid HTTP(S) URL")
    text = value.strip()
    if (
        not text
        or len(text) > 2_048
        or "\\" in text
        or any(char.isspace() or unicodedata.category(char).startswith("C") for char in text)
    ):
        raise ValueError(f"{label} must be a valid HTTP(S) URL")
    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid HTTP(S) URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise ValueError(f"{label} must be a valid HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not contain credentials")
    try:
        _ = parsed.port  # Access performs urllib's range and syntax validation.
    except ValueError as exc:
        raise ValueError(f"{label} contains an invalid port") from exc
    if not valid_url_host(parsed.hostname):
        raise ValueError(f"{label} contains an invalid hostname")
    return parsed


def valid_url_host(hostname: str) -> bool:
    """Accept literal IPs, localhost-style names, and valid IDNA DNS names."""
    if not isinstance(hostname, str):
        return False
    normalized = hostname.lower().rstrip(".")
    try:
        ipaddress.ip_address(normalized)
        return True
    except ValueError:
        pass
    try:
        ascii_host = normalized.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    return bool(
        len(ascii_host) <= 253
        and re.fullmatch(
            r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*" r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
            ascii_host,
            re.IGNORECASE,
        )
    )


def is_loopback_host(hostname: str) -> bool:
    if not isinstance(hostname, str):
        return False
    normalized = hostname.lower().rstrip(".")
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False
