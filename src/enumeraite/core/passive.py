"""Bounded, keyless discovery of historical subdomain names from public sources.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from urllib3.exceptions import HTTPError as Urllib3HTTPError
from urllib3.exceptions import TimeoutError as Urllib3TimeoutError

_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_COLLECTION_SECONDS = 60
_HACKERTARGET_FREE_RESULTS = 50
_SOURCES = ("crt.sh", "hackertarget")
_DOMAIN_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_DIGIT_RUN = re.compile(r"[0-9]+")


@dataclass
class PassiveResult:
    names: list[str]
    sources: dict[str, list[str]]
    errors: list[str]
    truncated: bool = False


class _SourceError(ValueError):
    """An expected service failure with a message safe to show to the user."""


class _ResponseTooLarge(_SourceError):
    pass


@dataclass
class _SourceResult:
    source: str
    names: list[str]
    error: str | None = None
    truncated: bool = False
    # Retain actual source membership while selecting a small candidate sample.
    # This private set is bounded by the response-byte limit and is not persisted.
    observed_names: set[str] | None = None


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
    """Query public services and optional installed subfinder concurrently.

    ``sources`` maps service names to their retained, scoped names. A failed
    source still has an entry and an explanatory ``errors`` item. The result is
    deterministic regardless of which service replies first. Dedicated daemon
    workers also bound caller latency when a platform DNS lookup ignores the
    HTTP client's socket timeout. Optional subfinder uses only keyless sources
    with isolated temporary configuration, never the user's saved provider keys.
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
    executable = shutil.which("subfinder")
    if executable:
        # PATH may contain relative directories; changing the child's cwd must
        # not change which installed executable was selected.
        executable = os.path.abspath(executable)
    source_order = (*_SOURCES, "subfinder") if executable else _SOURCES
    replies: queue.Queue[_SourceResult] = queue.Queue(maxsize=len(_SOURCES))
    if time.monotonic() < stop_at:
        for source in _SOURCES:
            threading.Thread(
                target=_source_worker,
                args=(source, target, max_results, stop_at, replies),
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
    for source in source_order:
        source_result = received.get(source)
        observed = (
            source_result.observed_names
            if source_result is not None and source_result.observed_names is not None
            else set(sources[source])
        )
        sources[source] = [name for name in names if name in observed]
    return PassiveResult(names=names, sources=sources, errors=errors, truncated=truncated)


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
            value = line.decode("utf-8").strip()
            if value.startswith("*."):
                value = value[2:]
            value = normalize_domain(value)
        except (UnicodeError, ValueError):
            malformed = True
            return False
        if value.endswith("." + target):
            names.add(value)
        return len(names) >= max_results

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
                        error, truncated = "result limit reached; results are incomplete", True
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
    return _SourceResult("subfinder", sorted(names), error, truncated)


def _source_worker(
    source: str,
    target: str,
    max_results: int,
    deadline: float,
    replies: queue.Queue[_SourceResult],
) -> None:
    try:
        if source == "crt.sh":
            body = _fetch_bytes(
                "https://crt.sh/", {"q": f"%.{target}", "output": "json"}, deadline=deadline
            )
            raw_names = _parse_crtsh(body)
            service_cap = False
        else:
            body = _fetch_bytes(
                "https://api.hackertarget.com/hostsearch/", {"q": target}, deadline=deadline
            )
            raw_names = _parse_hostsearch(body)
            service_cap = len(raw_names) >= _HACKERTARGET_FREE_RESULTS
        names: set[str] = set()
        timed_out = False
        for value in raw_names:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            name = _scoped_name(value, target)
            if name is not None:
                names.add(name)
        result = _SourceResult(
            source,
            _sample_names(names, target, max_results),
            error="discovery timed out; results are incomplete" if timed_out else None,
            truncated=timed_out or service_cap or len(names) > max_results,
            observed_names=names,
        )
    except _ResponseTooLarge:
        result = _SourceResult(source, [], "response exceeded the 8 MiB safety limit", True)
    except (TimeoutError, requests.Timeout, Urllib3TimeoutError):
        result = _SourceResult(source, [], "discovery timed out; results are incomplete", True)
    except _SourceError as exc:
        result = _SourceResult(source, [], str(exc))
    except (requests.RequestException, Urllib3HTTPError):
        # Request exception strings may include proxy URLs or credentials.
        result = _SourceResult(source, [], "service connection failed")
    except (ValueError, RecursionError, UnicodeError):
        result = _SourceResult(source, [], "service returned an invalid response")
    replies.put_nowait(result)


def _fetch_bytes(url: str, params: dict[str, str], *, deadline: float) -> bytes:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("discovery deadline exceeded")
    with requests.Session() as session:
        # No implicit .netrc credentials or proxy secrets go to public services.
        session.trust_env = False
        with session.get(
            url,
            params=params,
            timeout=(min(3.0, remaining), min(10.0, remaining)),
            headers={"Accept-Encoding": "identity", "User-Agent": "enumeraite/passive-discovery"},
            allow_redirects=False,
            stream=True,
        ) as response:
            status = response.status_code
            if 300 <= status < 400:
                raise _SourceError("service redirect refused")
            if status == 429:
                raise _SourceError("service rate limit reached (HTTP 429)")
            if not 200 <= status < 300:
                raise _SourceError(f"service returned HTTP {status}")
            encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
            if encoding not in ("", "identity"):
                raise _SourceError("service ignored the uncompressed-response requirement")
            try:
                content_length = int(response.headers.get("Content-Length", "0"))
            except (TypeError, ValueError):
                content_length = 0
            if content_length > _MAX_RESPONSE_BYTES:
                raise _ResponseTooLarge()
            return _read_body(response, deadline=deadline)


def _read_body(response: Any, *, deadline: float) -> bytes:
    """Use one socket read per check so drip-fed bodies cannot hide a timeout."""
    body = bytearray()
    read_one = getattr(getattr(response, "raw", None), "read1", None)
    stream = None if callable(read_one) else iter(response.iter_content(chunk_size=16_384))
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("discovery deadline exceeded")
        try:
            chunk = (
                read_one(16_384, decode_content=False)
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
        if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
            raise _ResponseTooLarge()
        body.extend(chunk)
    return bytes(body)


def _parse_crtsh(body: bytes) -> list[str]:
    payload = json.loads(body)
    if not isinstance(payload, list):
        raise _SourceError("service returned invalid certificate data")
    names: list[str] = []
    for record in payload:
        if not isinstance(record, dict):
            raise _SourceError("service returned invalid certificate data")
        value = record.get("name_value", record.get("common_name"))
        if not isinstance(value, str):
            raise _SourceError("service returned invalid certificate data")
        names.extend(value.splitlines())
    return names


def _parse_hostsearch(body: bytes) -> list[str]:
    text = body.decode("utf-8").strip()
    if text.lower() in {"no records found", "no records found.", "no results found"}:
        return []
    if not text or text.lower().startswith(("error", "api count", "api limit", "quota")):
        raise _SourceError("service unavailable or free query quota exhausted")
    names: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        name, separator, address = line.partition(",")
        if not separator:
            raise _SourceError("service returned invalid host records")
        try:
            ipaddress.ip_address(address.strip())
        except ValueError as exc:
            raise _SourceError("service returned invalid host records") from exc
        names.append(name.strip())
    return names
