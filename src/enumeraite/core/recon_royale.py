"""Safety-gated client and submission preparation for Recon Royale."""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence, cast
from urllib.parse import urlunsplit

import requests

from .dns_validator import DNSValidator
from .url_safety import is_loopback_host, parse_http_url

_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$",
    re.IGNORECASE,
)
_MAX_API_RESPONSE_BYTES = 1_000_000
_MAX_SUBMISSION_CANDIDATES = 10_000
_MAX_SCOPE_INPUTS = 100_000


@dataclass
class RoyaleSubmission:
    target: str
    submitted_candidates: int
    generated_candidates: int
    valid_subdomains: list[str]
    passive_valid: list[str]
    enumeraite_valid: list[str]
    marginal_enumeraite: list[str]
    valid_rate: float
    enumeraite_valid_rate: float
    eligible: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReconRoyaleClient:
    """Minimal API-key client matching Recon Royale's documented API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://recon-royale.com",
        timeout: float = 20.0,
        session=None,
    ):
        if (
            not isinstance(api_key, str)
            or not api_key
            or len(api_key) > 4_096
            or any(char.isspace() or unicodedata.category(char).startswith("C") for char in api_key)
        ):
            raise ValueError("Recon Royale API key is required")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Recon Royale timeout must be finite and positive")
        self.api_key = api_key
        self.base_url = _service_base_url(base_url)
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        }

    def current_target(self) -> str:
        started = time.perf_counter()
        response = self.session.get(
            self.base_url + "/api/current-target-via-apikey",
            headers=self.headers,
            timeout=self.timeout,
            allow_redirects=False,
            stream=True,
        )
        try:
            _reject_redirect(response)
            response.raise_for_status()
            payload = _bounded_response_payload(response, deadline=started + self.timeout)
        finally:
            _close_response(response)
        if isinstance(payload, str):
            target = payload
        elif isinstance(payload, dict):
            target = next(
                (
                    str(payload[key])
                    for key in ("target", "domain", "current_target")
                    if payload.get(key)
                ),
                "",
            )
        else:
            target = ""
        target = target.strip().lower().rstrip(".")
        if not _DOMAIN_RE.fullmatch(target):
            raise ValueError(f"Recon Royale returned an invalid target: {target!r}")
        return target

    def submit(self, subdomains: Sequence[str]) -> dict[str, Any]:
        if isinstance(subdomains, (str, bytes)) or not isinstance(subdomains, Sequence):
            raise ValueError("subdomains must be a sequence of DNS names")
        values = _bounded_dns_values(
            subdomains,
            label="subdomains",
            max_items=_MAX_SUBMISSION_CANDIDATES,
        )
        if not values:
            raise ValueError("Refusing to submit an empty subdomain list")
        started = time.perf_counter()
        response = self.session.post(
            self.base_url + "/api/submit-via-apikey",
            headers={**self.headers, "Content-Type": "application/json"},
            json={"subdomains": values},
            timeout=self.timeout,
            allow_redirects=False,
            stream=True,
        )
        try:
            _reject_redirect(response)
            response.raise_for_status()
            payload = _bounded_response_payload(response, deadline=started + self.timeout)
            return payload if isinstance(payload, dict) else {"response": payload}
        finally:
            _close_response(response)


def prepare_submission(
    target: str,
    passive: Iterable[str],
    generated: Iterable[str],
    validator: DNSValidator | None = None,
    min_valid_rate: float = 0.20,
) -> RoyaleSubmission:
    """Scope, resolve, and score a potential submission without sending it."""
    if (
        isinstance(min_valid_rate, bool)
        or not isinstance(min_valid_rate, (int, float))
        or not math.isfinite(min_valid_rate)
        or not 0.10 <= min_valid_rate <= 1.0
    ):
        raise ValueError("min_valid_rate must be between 0.10 and 1.0")
    if not isinstance(target, str):
        raise ValueError("target must be a valid DNS apex")
    apex = target.lower().rstrip(".")
    if not _DOMAIN_RE.fullmatch(apex):
        raise ValueError("target must be a valid DNS apex")
    passive_scoped = filter_in_scope(passive, apex)
    generated_scoped = filter_in_scope(generated, apex)
    union = list(dict.fromkeys(passive_scoped + generated_scoped))
    if len(union) > _MAX_SUBMISSION_CANDIDATES:
        raise ValueError("submission exceeds the 10000-candidate validation limit")
    if not union:
        return RoyaleSubmission(
            target=apex,
            submitted_candidates=0,
            generated_candidates=0,
            valid_subdomains=[],
            passive_valid=[],
            enumeraite_valid=[],
            marginal_enumeraite=[],
            valid_rate=0.0,
            enumeraite_valid_rate=0.0,
            eligible=False,
            reason="no in-scope candidates",
        )

    dns = validator or DNSValidator()
    results = dns.validate_subdomains(union, detect_wildcard=True, apex=apex)
    union_set = set(union)
    valid = list(
        dict.fromkeys(
            normalized
            for result in results
            if result.exists
            and not result.wildcard
            and (normalized := result.subdomain.lower().rstrip(".")) in union_set
        )
    )
    valid_set = set(valid)
    passive_valid = [value for value in passive_scoped if value in valid_set]
    generated_valid = [value for value in generated_scoped if value in valid_set]
    passive_set = set(passive_valid)
    marginal = [value for value in generated_valid if value not in passive_set]
    valid_rate = len(valid) / len(union)
    enumeraite_valid_rate = (
        len(generated_valid) / len(generated_scoped) if generated_scoped else 0.0
    )
    eligible = bool(marginal) and enumeraite_valid_rate >= min_valid_rate
    if not generated_scoped:
        reason = "no in-scope Enumeraite candidates"
    elif enumeraite_valid_rate < min_valid_rate:
        reason = (
            f"Enumeraite valid rate {enumeraite_valid_rate:.1%} is below "
            f"{min_valid_rate:.1%} threshold"
        )
    elif not marginal:
        reason = "no marginal validated Enumeraite findings beyond the passive baseline"
    else:
        reason = ""
    return RoyaleSubmission(
        target=apex,
        submitted_candidates=len(union),
        generated_candidates=len(generated_scoped),
        valid_subdomains=valid,
        passive_valid=passive_valid,
        enumeraite_valid=generated_valid,
        marginal_enumeraite=marginal,
        valid_rate=valid_rate,
        enumeraite_valid_rate=enumeraite_valid_rate,
        eligible=eligible,
        reason=reason,
    )


def filter_in_scope(values: Iterable[str], apex: str) -> list[str]:
    """Return normalized proper subdomains below the exact apex."""
    if isinstance(values, (str, bytes)):
        raise ValueError("values must be an iterable of DNS names, not a string")
    if not isinstance(apex, str):
        raise ValueError("apex must be a valid DNS name")
    apex = apex.strip().lower().rstrip(".")
    if not _DOMAIN_RE.fullmatch(apex):
        raise ValueError("apex must be a valid DNS name")
    return [
        value
        for value in _bounded_dns_values(
            values,
            label="values",
            max_items=_MAX_SCOPE_INPUTS,
        )
        if value.endswith("." + apex)
    ]


def _bounded_dns_values(
    values: Iterable[str],
    *,
    label: str,
    max_items: int,
) -> list[str]:
    """Normalize a finite DNS iterable without silently coercing objects."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be an iterable of DNS names, not a string")
    normalized_values: list[str] = []
    for index, value in enumerate(values):
        if index >= max_items:
            raise ValueError(f"{label} exceed the {max_items}-item limit")
        if not isinstance(value, str):
            raise ValueError(f"{label} must contain only DNS-name strings")
        normalized = value.strip().lower().rstrip(".")
        if normalized and _DOMAIN_RE.fullmatch(normalized):
            normalized_values.append(normalized)
    return list(dict.fromkeys(normalized_values))


def _service_base_url(value: str) -> str:
    parsed = parse_http_url(value, label="Recon Royale base URL")
    if parsed.query or parsed.fragment:
        raise ValueError("Recon Royale base URL must not contain a query or fragment")
    if parsed.scheme != "https" and not is_loopback_host(parsed.hostname or ""):
        raise ValueError("Recon Royale base URL must use HTTPS for a remote service")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _reject_redirect(response) -> None:
    status = int(getattr(response, "status_code", 200) or 200)
    if 300 <= status < 400:
        raise ValueError("Recon Royale API redirect refused to protect the API key")


def _bounded_response_payload(response, *, deadline: float | None = None) -> Any:
    """Decode a bounded JSON/text response from the authenticated service."""
    headers = getattr(response, "headers", {}) or {}
    try:
        declared = int(headers.get("Content-Length", 0) or 0)
    except (TypeError, ValueError):
        declared = 0
    if declared > _MAX_API_RESPONSE_BYTES:
        raise ValueError("Recon Royale API response exceeds the 1 MB limit")

    body = bytearray()
    iterator = getattr(response, "iter_content", None)
    if callable(iterator):
        raw = getattr(response, "raw", None)
        read_one = getattr(raw, "read1", None)
        stream = (
            None if callable(read_one) else iter(cast(Iterable[Any], iterator(chunk_size=16_384)))
        )
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Recon Royale API response body deadline exceeded")
            try:
                chunk = (
                    read_one(16_384, decode_content=False)
                    if callable(read_one)
                    else next(stream)  # type: ignore[arg-type]
                )
            except StopIteration:
                break
            if not chunk:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Recon Royale API response body deadline exceeded")
            if not isinstance(chunk, (bytes, bytearray)):
                raise ValueError("Recon Royale API returned a non-byte response chunk")
            if len(chunk) > _MAX_API_RESPONSE_BYTES - len(body):
                raise ValueError("Recon Royale API response exceeds the 1 MB limit")
            body.extend(chunk)
        encoding = getattr(response, "encoding", None) or "utf-8"
        text = bytes(body).decode(encoding, errors="replace").strip()
    else:  # Compatibility with lightweight injected sessions.
        text = str(getattr(response, "text", "")).strip()
        if len(text.encode("utf-8")) > _MAX_API_RESPONSE_BYTES:
            raise ValueError("Recon Royale API response exceeds the 1 MB limit")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _close_response(response) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()
