"""Bounded discovery of historical subdomain names from public sources.

These names are evidence for a scan, not a claim that DNS currently resolves.
No source is exhaustive; failures and known response caps remain visible.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess  # nosec B404
import tempfile
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from urllib3.exceptions import HTTPError as Urllib3HTTPError
from urllib3.exceptions import TimeoutError as Urllib3TimeoutError

from .config import load_merklemap_api_key

_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_COLLECTION_SECONDS = 60
_MAX_MERKLEMAP_PAGES = 100
_MAX_CRTSH_ATTEMPTS = 3
_SUBFINDER_OBSERVATION_MULTIPLIER = 8
_MAX_SUBFINDER_OBSERVATIONS = 50_000
_CRTSH_RETRY_STATUSES = frozenset({404, 408, 500, 502, 503, 504})
_SOURCES = ("crt.sh",)
_DOMAIN_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_DIGIT_RUN = re.compile(r"[0-9]+")


@dataclass
class PassiveResult:
    names: list[str]
    sources: dict[str, list[str]]
    errors: list[str]
    truncated: bool = False
    evidence: dict[str, list[dict[str, str]]] = field(default_factory=dict)


class _SourceError(ValueError):
    """An expected service failure with a message safe to show to the user."""


class _ResponseTooLarge(_SourceError):
    pass


class _HTTPSourceError(_SourceError):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"service returned HTTP {status}")


@dataclass
class _SourceResult:
    source: str
    names: list[str]
    error: str | None = None
    truncated: bool = False
    # Retain actual source membership while selecting a small candidate sample.
    # This private set is bounded by the response-byte limit and is not persisted.
    observed_names: set[str] | None = None
    evidence: dict[str, list[dict[str, str]]] = field(default_factory=dict)


def normalize_domain(value: str) -> str:
    """Return an ASCII domain, rejecting URLs, IPs, wildcards and ambiguous input."""
    if not isinstance(value, str) or len(value) > 1_024:
        raise ValueError("target must be a DNS domain such as example.com")
    value = value.strip().lower()
    if value.endswith("."):
        value = value[:-1]
    if not value or any(
        char.isspace() or unicodedata.category(char).startswith("C") for char in value
    ):
        raise ValueError("target must be a DNS domain such as example.com")
    try:
        domain = value.encode("idna").decode("ascii").lower()
        # The stdlib fast path accepts ASCII without validating A-labels.
        for label in domain.split("."):
            if label.startswith("xn--"):
                label.encode("ascii").decode("idna")
    except UnicodeError as exc:
        raise ValueError("target must be a valid DNS domain") from exc
    if (
        len(domain) > 253
        or "." not in domain
        or not all(_DOMAIN_RE.fullmatch(label) for label in domain.split("."))
    ):
        raise ValueError("target must be a DNS domain such as example.com")
    try:
        ipaddress.ip_address(domain)
    except ValueError:
        return domain
    raise ValueError("target must be a DNS domain, not an IP address")


def _scoped_name(value: Any, target: str) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.startswith("*."):
        value = value[2:]
    try:
        name = normalize_domain(value)
    except ValueError:
        return None
    return name if name.endswith("." + target) else None


def collect_passive(
    target: str,
    *,
    timeout: float = 10,
    max_results: int = 5000,
    deadline: float | None = None,
) -> PassiveResult:
    """Query crt.sh, configured MerkleMap, and optional subfinder concurrently.

    ``sources`` maps service names to their retained, scoped names. A failed
    source still has an entry and an explanatory ``errors`` item. The result is
    deterministic regardless of which service replies first. Dedicated daemon
    workers also bound caller latency when a platform DNS lookup ignores the
    HTTP client's socket timeout. Optional subfinder uses only keyless sources
    with isolated temporary configuration, never the user's saved provider keys.
    ``MERKLEMAP_API_KEY`` in the selected JSON config or environment opts into
    authenticated MerkleMap search; its token goes only to the fixed API endpoint
    and is never passed to subfinder.
    Explicit timeouts up to 60 seconds are honored, unless ``deadline`` is earlier.
    Truncated samples take turns across naming families before returning sorted
    names, so large numeric families do not hide unrelated observed vocabulary.
    """
    target = normalize_domain(target)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 < timeout <= _MAX_COLLECTION_SECONDS
    ):
        raise ValueError("passive timeout must be positive, finite, and at most 60 seconds")
    if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
        raise ValueError("max_results must be a positive integer")
    if deadline is not None and (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise ValueError("deadline must be a finite monotonic timestamp")
    stop_at = min(time.monotonic() + timeout, deadline if deadline is not None else math.inf)
    executable = _find_subfinder()
    merklemap_key = load_merklemap_api_key()
    http_sources = (*_SOURCES, "merklemap") if merklemap_key else _SOURCES
    source_order = (*http_sources, "subfinder") if executable else http_sources
    replies: queue.Queue[_SourceResult] = queue.Queue(maxsize=len(http_sources))
    # A single immutable snapshot survives a later page's uninterruptible lookup.
    partial_merklemap: deque[_SourceResult] = deque(maxlen=1)
    if time.monotonic() < stop_at:
        for source in http_sources:
            threading.Thread(
                target=_source_worker,
                args=(source, target, max_results, stop_at, replies),
                kwargs=(
                    {"api_key": merklemap_key, "partial": partial_merklemap}
                    if source == "merklemap"
                    else {}
                ),
                name=f"enumeraite-passive-{source}",
                daemon=True,
            ).start()

    received: dict[str, _SourceResult] = {}
    # The HTTP workers continue while this bounded local process streams names.
    # Keeping subprocess ownership here guarantees it is killed/reaped before
    # returning, rather than abandoning it in a daemon collector thread.
    if executable and time.monotonic() < stop_at:
        received["subfinder"] = _run_subfinder(executable, target, max_results, stop_at)
    while len(received) < len(source_order):
        remaining = stop_at - time.monotonic()
        if remaining <= 0:
            break
        try:
            result = replies.get(timeout=remaining)
        except queue.Empty:
            break
        received[result.source] = result

    # Also retain completed replies already queued at the boundary.
    while True:
        try:
            result = replies.get_nowait()
        except queue.Empty:
            break
        received[result.source] = result
    if "merklemap" not in received and partial_merklemap:
        received["merklemap"] = partial_merklemap.pop()

    sources: dict[str, list[str]] = {}
    errors: list[str] = []
    truncated = False
    for source in source_order:
        source_result = received.get(source)
        if source_result is None:
            sources[source] = []
            errors.append(f"{source}: discovery deadline exceeded; results are incomplete")
            truncated = True
        else:
            sources[source] = source_result.names
            if source_result.error:
                errors.append(f"{source}: {source_result.error}")
            truncated = truncated or source_result.truncated
    all_names = {name for values in sources.values() for name in values}
    if len(all_names) > max_results:
        truncated = True
    names = _sample_names(all_names, target, max_results)
    evidence: dict[str, list[dict[str, str]]] = {}
    for source in source_order:
        source_result = received.get(source)
        observed = (
            source_result.observed_names
            if source_result is not None and source_result.observed_names is not None
            else set(sources[source])
        )
        sources[source] = [name for name in names if name in observed]
        if source_result is not None:
            for name in sources[source]:
                records = source_result.evidence.get(name)
                if records:
                    evidence.setdefault(name, []).extend(records)
    return PassiveResult(
        names=names, sources=sources, errors=errors, truncated=truncated, evidence=evidence
    )


def _sample_names(names: set[str], target: str, limit: int) -> list[str]:
    """Sample already normalized names fairly; never invent naming evidence.

    All input is bounded by HTTP/subprocess byte limits. Sort once, group numeric
    variants, then take at most ``limit`` turns rather than building combinations.
    Family grouping is only an allocation heuristic, not a semantic inference.
    """
    ordered = sorted(names)
    if len(ordered) <= limit:
        return ordered
    families: dict[str, list[str]] = {}
    for name in ordered:
        relative = name[: -(len(target) + 1)]
        families.setdefault(_DIGIT_RUN.sub("#", relative), []).append(name)
    pending = deque(iter(values) for values in families.values())
    selected: list[str] = []
    while pending and len(selected) < limit:
        family = pending.popleft()
        try:
            selected.append(next(family))
        except StopIteration:
            continue
        pending.append(family)
    return sorted(selected)


def _find_subfinder() -> str | None:
    """Find an installed binary even when a dashboard has a minimal PATH."""
    executable = shutil.which("subfinder")
    if executable:
        # A relative PATH entry must survive the child's isolated working directory.
        return os.path.abspath(executable)
    directories = []
    if gobin := os.environ.get("GOBIN"):
        directories.append(Path(gobin))
    directories.extend(
        Path(root) / "bin" for root in os.environ.get("GOPATH", "").split(os.pathsep) if root
    )
    directories.extend(
        (Path.home() / "go" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin"))
    )
    for directory in directories:
        executable = shutil.which(str(directory / "subfinder"))
        if executable:
            return os.path.abspath(executable)
    return None


def _subfinder_environment(directory: Path, config: Path, providers: Path) -> dict[str, str]:
    """No model/provider credentials, proxy settings, or user config are inherited."""
    environment = {
        name: os.environ[name] for name in ("PATH", "SYSTEMROOT", "WINDIR") if name in os.environ
    }
    environment.update(
        {
            "HOME": str(directory),
            "USERPROFILE": str(directory),
            "XDG_CONFIG_HOME": str(directory),
            "XDG_CACHE_HOME": str(directory),
            "APPDATA": str(directory),
            "LOCALAPPDATA": str(directory),
            "TMPDIR": str(directory),
            "TEMP": str(directory),
            "TMP": str(directory),
            "SUBFINDER_CONFIG": str(config),
            "SUBFINDER_PROVIDER_CONFIG": str(providers),
        }
    )
    return environment


def _run_subfinder(
    executable: str, target: str, max_results: int, deadline: float
) -> _SourceResult:
    """Stream bounded stdout/stderr; retain partial scoped names on every stop."""
    names: set[str] = set()
    evidence: dict[str, list[dict[str, str]]] = {}
    # Observe beyond final retention so an early numeric family cannot consume
    # every slot before the family sampler sees later names. The byte/deadline
    # bounds still apply, even when output repeats or contains no scoped names.
    observation_limit = min(
        max_results * _SUBFINDER_OBSERVATION_MULTIPLIER, _MAX_SUBFINDER_OBSERVATIONS
    )
    malformed = False
    truncated = False
    error: str | None = None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return _SourceResult("subfinder", [], "discovery timed out; results are incomplete", True)
    # Reserve part of the same deadline for killing/reaping and pipe cleanup.
    run_deadline = deadline - min(0.25, remaining * 0.1)
    chunks: queue.Queue[tuple[str, bytes | None]] = queue.Queue(maxsize=16)
    finished = threading.Event()
    process: subprocess.Popen | None = None
    readers: list[threading.Thread] = []
    buffer = bytearray()
    dropping_line = False
    consumed = 0

    def read_pipe(label: str, pipe: Any) -> None:
        try:
            while not finished.is_set():
                chunk = pipe.read(4096)
                while not finished.is_set():
                    try:
                        chunks.put((label, chunk or None), timeout=0.02)
                        break
                    except queue.Full:
                        continue
                if not chunk:
                    break
        except (OSError, ValueError):
            # Closing a pipe during cancellation is expected; never surface its
            # contents or a platform exception that could contain local paths.
            return

    def retain_line(line: bytes) -> bool:
        nonlocal malformed
        if not line.strip():
            return False
        try:
            matched_name = line.decode("utf-8").strip()
            value = matched_name
            if value.startswith("*."):
                value = value[2:]
            value = normalize_domain(value)
        except (UnicodeError, ValueError):
            malformed = True
            return False
        if value.endswith("." + target):
            names.add(value)
            _retain_evidence(
                evidence,
                value,
                {
                    "source": "subfinder",
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "query": target,
                    "matched_name": _safe_text(matched_name, 255),
                },
            )
        return len(names) >= observation_limit

    try:
        with tempfile.TemporaryDirectory(prefix="enumeraite-subfinder-") as temporary:
            directory = Path(temporary)
            config = directory / "config.yaml"
            providers = directory / "provider-config.yaml"
            config.write_text("{}\n", encoding="utf-8")
            providers.write_text("{}\n", encoding="utf-8")
            if time.monotonic() >= run_deadline:
                return _SourceResult(
                    "subfinder", [], "discovery timed out; results are incomplete", True
                )
            try:
                # Arguments are fixed except for an already normalized DNS name;
                # no shell, active scanning, updates, or saved source keys.
                process = subprocess.Popen(  # nosec B603
                    [
                        executable,
                        "-d",
                        target,
                        "-silent",
                        "-duc",
                        "-es",
                        "hackertarget",
                        "-timeout",
                        str(max(1, min(10, math.ceil(run_deadline - time.monotonic())))),
                        "-max-time",
                        "1",
                        "-config",
                        str(config),
                        "-pc",
                        str(providers),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    bufsize=0,
                    cwd=directory,
                    env=_subfinder_environment(directory, config, providers),
                    start_new_session=os.name != "nt",
                )
                for label, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                    reader = threading.Thread(
                        target=read_pipe,
                        args=(label, pipe),
                        name=f"enumeraite-subfinder-{label}",
                        daemon=True,
                    )
                    reader.start()
                    readers.append(reader)
                ended: set[str] = set()
                limit_reached = False
                while len(ended) < 2:
                    remaining = run_deadline - time.monotonic()
                    if remaining <= 0:
                        error, truncated = "discovery timed out; results are incomplete", True
                        break
                    try:
                        label, chunk = chunks.get(timeout=min(0.05, remaining))
                    except queue.Empty:
                        continue
                    if chunk is None:
                        ended.add(label)
                        if label == "stdout" and buffer and not dropping_line:
                            limit_reached = retain_line(bytes(buffer))
                            buffer.clear()
                    else:
                        allowed = max(0, _MAX_RESPONSE_BYTES - consumed)
                        consumed += len(chunk)
                        if label == "stdout":
                            for byte in chunk[:allowed]:
                                if byte == 10:
                                    if not dropping_line:
                                        limit_reached = retain_line(bytes(buffer))
                                    buffer.clear()
                                    dropping_line = False
                                    if limit_reached:
                                        break
                                elif not dropping_line:
                                    if len(buffer) >= 4096:
                                        malformed, dropping_line = True, True
                                        buffer.clear()
                                    else:
                                        buffer.append(byte)
                        if consumed > _MAX_RESPONSE_BYTES:
                            error, truncated = (
                                "output exceeded the 8 MiB safety limit; results are incomplete",
                                True,
                            )
                            break
                    if limit_reached:
                        error, truncated = (
                            "observation limit reached; results are incomplete",
                            True,
                        )
                        break
                if error is None:
                    try:
                        code = process.wait(timeout=max(0.001, run_deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        error, truncated = "discovery timed out; results are incomplete", True
                    else:
                        if code != 0:
                            error, truncated = (
                                "optional collector exited unsuccessfully; results are incomplete",
                                True,
                            )
                        elif malformed:
                            error, truncated = (
                                "ignored malformed collector output; results are incomplete",
                                True,
                            )
            finally:
                finished.set()
                if process is not None:
                    if process.poll() is None:
                        try:
                            if os.name == "nt":
                                process.kill()
                            else:
                                os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            process.poll()  # The child already exited before cancellation.
                    # Normally SIGKILL reaps immediately. A rare OS-level
                    # uninterruptible child must not defeat the caller's wall
                    # clock limit; retain a daemon reaper in that case.
                    try:
                        process.wait(timeout=max(0.001, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        error, truncated = "collector cleanup delayed; results are incomplete", True
                        threading.Thread(
                            target=process.wait, name="enumeraite-subfinder-reaper", daemon=True
                        ).start()
                    for pipe in (process.stdout, process.stderr):
                        if pipe is not None:
                            pipe.close()
                    for reader in readers:
                        reader.join(timeout=max(0, min(0.02, deadline - time.monotonic())))
    except (OSError, ValueError):
        error, truncated = "optional collector could not run; results are incomplete", True
    return _SourceResult(
        "subfinder",
        _sample_names(names, target, max_results),
        error,
        truncated or len(names) > max_results,
        observed_names=names,
        evidence=evidence,
    )


def _source_worker(
    source: str,
    target: str,
    max_results: int,
    deadline: float,
    replies: queue.Queue[_SourceResult],
    *,
    api_key: str = "",
    partial: deque[_SourceResult] | None = None,
) -> None:
    names: set[str] = set()
    evidence: dict[str, list[dict[str, str]]] = {}
    error = None
    truncated = False
    try:
        if source == "merklemap":
            truncated = _collect_merklemap(
                target, api_key, max_results, deadline, names, evidence, partial
            )
        elif source == "crt.sh":
            query = f"%.{target}"
            body = _fetch_crtsh(
                {"q": query, "output": "json"},
                deadline=deadline,
            )
            collected_at = datetime.now(timezone.utc).isoformat()
            records, malformed = _parse_crtsh(body)
            if malformed:
                error, truncated = (
                    "ignored malformed certificate data; results are incomplete",
                    True,
                )
            for record in records:
                if time.monotonic() >= deadline:
                    raise TimeoutError("discovery deadline exceeded")
                name = _scoped_name(record["matched_name"], target)
                if name is not None:
                    names.add(name)
                    _retain_evidence(
                        evidence,
                        name,
                        {
                            **record,
                            "source": source,
                            "collected_at": collected_at,
                            "query": query,
                            "matched_name": _safe_text(record["matched_name"], 255),
                        },
                    )
    except _ResponseTooLarge:
        error, truncated = "response exceeded the 8 MiB safety limit", True
    except (TimeoutError, requests.Timeout, Urllib3TimeoutError):
        error, truncated = "discovery timed out; results are incomplete", True
    except _SourceError as exc:
        error, truncated = str(exc), source == "merklemap"
    except (requests.RequestException, Urllib3HTTPError):
        # Request exceptions can contain proxy credentials or the bearer token.
        error, truncated = "service connection failed", source == "merklemap"
    except (ValueError, RecursionError, UnicodeError):
        error, truncated = "service returned an invalid response", source == "merklemap"
    replies.put_nowait(
        _SourceResult(
            source,
            _sample_names(names, target, max_results),
            error=error,
            truncated=truncated or len(names) > max_results,
            observed_names=names,
            evidence=evidence,
        )
    )


def _collect_merklemap(
    target: str,
    api_key: str,
    max_results: int,
    deadline: float,
    names: set[str],
    evidence: dict[str, list[dict[str, str]]],
    partial: deque[_SourceResult] | None,
) -> bool:
    """Fill observations incrementally so a later page failure cannot erase them."""
    query = f"*.{target}"
    received_bytes = 0
    processed = 0
    total = 0
    previous_body = None
    for page in range(_MAX_MERKLEMAP_PAGES):
        body = _fetch_bytes(
            "https://api.merklemap.com/v1/search",
            {"query": query, "type": "wildcard", "page": str(page)},
            deadline=deadline,
            headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
            max_bytes=_MAX_RESPONSE_BYTES - received_bytes,
        )
        received_bytes += len(body)
        if body == previous_body:
            raise _SourceError("service repeated a result page; results are incomplete")
        previous_body = body
        payload = json.loads(body)
        if (
            not isinstance(payload, dict)
            or type(payload.get("count")) is not int
            or payload["count"] < 0
            or not isinstance(payload.get("results"), list)
        ):
            raise _SourceError("service returned invalid search results")
        records = payload["results"]
        total = max(total, payload["count"])
        if not records:
            if processed < total:
                raise _SourceError(
                    "service ended before its reported count; results are incomplete"
                )
            return False
        collected_at = datetime.now(timezone.utc).isoformat()
        for record in records:
            if time.monotonic() >= deadline:
                raise TimeoutError("discovery deadline exceeded")
            if not isinstance(record, dict) or not isinstance(record.get("hostname"), str):
                raise _SourceError("service returned invalid search results")
            name = _scoped_name(record["hostname"], target)
            if name is None:
                continue
            names.add(name)
            metadata = {
                "source": "merklemap",
                "query": query,
                "matched_name": _safe_text(record["hostname"], 255),
                "collected_at": collected_at,
            }
            common_name = record.get("subject_common_name")
            if isinstance(common_name, str) and common_name.strip():
                metadata["subject_common_name"] = _safe_text(common_name, 255)
            first_seen = record.get("first_seen")
            if isinstance(first_seen, str) and 0 < len(first_seen) <= 64:
                try:
                    observed_at = datetime.fromisoformat(first_seen.replace("Z", "+00:00"))
                    if observed_at.tzinfo is not None:
                        metadata["first_seen"] = observed_at.astimezone(timezone.utc).isoformat()
                except (ValueError, OverflowError):
                    pass
            _retain_evidence(evidence, name, metadata)
        processed += len(records)
        if processed >= total:
            return False
        if len(names) >= max_results:
            return True
        if partial is not None:
            partial.append(
                _SourceResult(
                    "merklemap",
                    list(names),
                    "discovery deadline exceeded; results are incomplete",
                    True,
                    observed_names=names.copy(),
                    evidence={name: list(records) for name, records in evidence.items()},
                )
            )
    raise _SourceError("service page limit reached; results are incomplete")


def _fetch_crtsh(params: dict[str, str], *, deadline: float) -> bytes:
    """Retry only free CT transport failures, inside the original shared budget."""
    for attempt in range(_MAX_CRTSH_ATTEMPTS):
        try:
            return _fetch_bytes("https://crt.sh/", params, deadline=deadline)
        except (
            _HTTPSourceError,
            TimeoutError,
            requests.Timeout,
            requests.ConnectionError,
            Urllib3HTTPError,
        ) as exc:
            if isinstance(exc, _HTTPSourceError) and exc.status not in _CRTSH_RETRY_STATUSES:
                raise
            delay = 0.25 * (2**attempt)
            if attempt + 1 == _MAX_CRTSH_ATTEMPTS or deadline - time.monotonic() <= delay:
                raise
            time.sleep(delay)
            if time.monotonic() >= deadline:
                raise
    raise AssertionError("crt.sh attempts exhausted without a result")


def _fetch_bytes(
    url: str,
    params: dict[str, str],
    *,
    deadline: float,
    headers: dict[str, str] | None = None,
    max_bytes: int | None = None,
) -> bytes:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("discovery deadline exceeded")
    limit = _MAX_RESPONSE_BYTES if max_bytes is None else min(max_bytes, _MAX_RESPONSE_BYTES)
    if limit <= 0:
        raise _ResponseTooLarge()
    request_headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "enumeraite/passive-discovery",
    }
    if headers:
        request_headers.update(headers)
    with requests.Session() as session:
        # No implicit .netrc credentials or proxy secrets go to public services.
        session.trust_env = False
        with session.get(
            url,
            params=params,
            timeout=(min(3.0, remaining), min(10.0, remaining)),
            headers=request_headers,
            allow_redirects=False,
            stream=True,
        ) as response:
            status = response.status_code
            if 300 <= status < 400:
                raise _SourceError("service redirect refused")
            if status == 429:
                raise _SourceError("service rate limit reached (HTTP 429)")
            if not 200 <= status < 300:
                raise _HTTPSourceError(status)
            encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
            if encoding not in ("", "identity"):
                raise _SourceError("service ignored the uncompressed-response requirement")
            try:
                content_length = int(response.headers.get("Content-Length", "0"))
            except (TypeError, ValueError):
                content_length = 0
            if content_length > limit:
                raise _ResponseTooLarge()
            return _read_body(response, deadline=deadline, max_bytes=limit)


def _read_body(response: Any, *, deadline: float, max_bytes: int | None = None) -> bytes:
    """Use one socket read per check so drip-fed bodies cannot hide a timeout."""
    limit = _MAX_RESPONSE_BYTES if max_bytes is None else max_bytes
    body = bytearray()
    read_one = getattr(getattr(response, "raw", None), "read1", None)
    stream = None if callable(read_one) else iter(response.iter_content(chunk_size=16_384))
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("discovery deadline exceeded")
        try:
            chunk = (
                read_one(min(16_384, limit - len(body) + 1), decode_content=False)
                if callable(read_one)
                else next(stream)  # type: ignore[arg-type]
            )
        except StopIteration:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("discovery deadline exceeded")
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise _SourceError("service returned a non-byte response")
        if len(body) + len(chunk) > limit:
            raise _ResponseTooLarge()
        body.extend(chunk)
    return bytes(body)


def _safe_text(value: str, limit: int) -> str:
    return "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in value.strip()[:limit]
    )


def _retain_evidence(
    evidence: dict[str, list[dict[str, str]]], name: str, record: dict[str, str]
) -> None:
    records = evidence.setdefault(name, [])
    if len(records) >= 3:
        return
    # Repeated stdout lines are one observation, not three distinct records
    # merely because they were read at different instants.
    if any(
        existing.keys() == record.keys()
        and all(existing[key] == value for key, value in record.items() if key != "collected_at")
        for existing in records
    ):
        return
    records.append(record)


def _parse_crtsh(body: bytes) -> tuple[list[dict[str, str]], bool]:
    payload = json.loads(body)
    if not isinstance(payload, list):
        raise _SourceError("service returned invalid certificate data")
    records: list[dict[str, str]] = []
    malformed = False
    for record in payload:
        if not isinstance(record, dict):
            malformed = True
            continue
        values = []
        for key in ("name_value", "common_name"):
            if key not in record:
                continue
            value = record[key]
            if isinstance(value, str):
                values.append(value)
            else:
                malformed = True
        if not values:
            malformed = True
            continue
        metadata: dict[str, str] = {}
        certificate_id = record.get("id")
        if isinstance(certificate_id, (str, int)) and not isinstance(certificate_id, bool):
            certificate_id = str(certificate_id)
            if re.fullmatch(r"[0-9]{1,32}", certificate_id):
                metadata["certificate_id"] = certificate_id
        issuer = record.get("issuer_name")
        if isinstance(issuer, str) and issuer.strip():
            metadata["issuer"] = _safe_text(issuer, 500)
        for key in ("not_before", "not_after", "entry_timestamp"):
            date = record.get(key)
            if not isinstance(date, str) or not 0 < len(date) <= 64:
                continue
            try:
                datetime.fromisoformat(date)
            except ValueError:
                continue
            if any(unicodedata.category(char).startswith("C") for char in date):
                continue
            metadata[key] = date
        for value in values:
            records.extend({"matched_name": name, **metadata} for name in value.splitlines())
    return records, malformed
