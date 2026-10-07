"""Bounded, rate-limited DNS workers with per-parent wildcard calibration."""

from __future__ import annotations

import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import dns.exception
import dns.resolver

from .dns_evidence import new_dns_evidence
from .dns_validator import DNSResult
from .prompt_spec import SUBDOMAIN_SPEC
from .text_safety import safe_exception_text


@dataclass
class ScanDNSResult(DNSResult):
    """Cancellation is unfinished work, not negative DNS evidence."""

    cancelled: bool = False
    dns_evidence: dict[str, Any] | None = None


class ScanDNS:
    def __init__(
        self,
        target: str,
        *,
        workers: int = 20,
        rate: float = 50,
        timeout: float = 2,
        cancel_event: threading.Event | None = None,
        event: Callable[[str, dict], None] | None = None,
    ):
        if not SUBDOMAIN_SPEC.validator(target):
            raise ValueError("Invalid DNS target")
        if not 1 <= workers <= 100 or not 0 < rate <= 1000 or not 0 < timeout <= 30:
            raise ValueError("Invalid DNS concurrency, rate, or timeout")
        self.target = target
        self.workers = workers
        self.rate = rate
        self.timeout = timeout
        self._rate_lock = threading.Lock()
        self._next_query = 0.0
        self._last_query: float | None = None
        self._cancel = cancel_event if cancel_event is not None else threading.Event()
        self._owns_cancel = cancel_event is None
        # Worker callbacks may arrive concurrently. Consumers must synchronize
        # their own event sink; no scanner locks are held during callbacks.
        self.event = event or (lambda kind, payload: None)
        self._validation_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._wildcards: dict[str, tuple[set[str], bool, str | None]] = {}
        self._parent_locks: dict[str, threading.Lock] = {}
        self.query_count = 0

    def _slot(self, deadline: float | None) -> bool:
        with self._rate_lock:
            if self._cancel.is_set():
                return False
            now = time.monotonic()
            scheduled = max(now, self._next_query)
            if deadline is not None and scheduled >= deadline:
                return False
            self._next_query = scheduled + 1 / self.rate
        delay = scheduled - time.monotonic()
        if delay > 0 and self._cancel.wait(delay):
            return False
        return not self._cancel.is_set() and (deadline is None or time.monotonic() < deadline)

    def _resolve(
        self,
        name: str,
        deadline: float | None,
        *,
        probe: bool = False,
        candidate: str | None = None,
    ) -> ScanDNSResult:
        started = time.monotonic()
        addresses: set[str] = set()
        errors = []
        cancelled = False
        evidence = None
        resolver = dns.resolver.Resolver()
        for kind in ("A", "AAAA"):
            if not self._slot(deadline):
                cancelled = self._cancel.is_set()
                errors.append(
                    "scan cancelled" if self._cancel.is_set() else "scan deadline reached"
                )
                break
            remaining = (
                min(self.timeout, deadline - time.monotonic())
                if deadline is not None
                else self.timeout
            )
            if remaining <= 0:
                errors.append("scan deadline reached")
                break
            try:
                with self._rate_lock:
                    if self._cancel.is_set():
                        cancelled = True
                        errors.append("scan cancelled")
                        break
                    self.query_count += 1
                    query_id = self.query_count
                    self._last_query = time.monotonic()
                if not probe and evidence is None:
                    evidence = new_dns_evidence(name)
                payload = {
                    "query_id": query_id,
                    "worker_id": threading.current_thread().name,
                    "hostname": name,
                    "candidate": candidate or name,
                    "query_kind": kind,
                    "probe": probe,
                }
                query_started = time.monotonic()
                status, query_addresses, error = "error", [], ""
                self.event("dns_start", payload.copy())
                try:
                    result = resolver.resolve(name, kind, lifetime=remaining, search=False)
                    query_addresses = [str(record) for record in result]
                    addresses.update(query_addresses)
                    status = "answered" if query_addresses else "no_answer"
                    if (
                        evidence is not None
                        and query_addresses
                        and isinstance(result, dns.resolver.Answer)
                    ):
                        # Only the chain dnspython actually followed is evidence;
                        # unrelated answer-section CNAMEs and probe answers are not.
                        aliases = evidence.get("aliases")
                        for rrset in result.chaining_result.cnames:
                            if aliases is not None and len(aliases) >= 16:
                                break
                            alias = rrset[0].target.to_text(omit_final_dot=True).lower()
                            if (
                                alias != name
                                and alias.endswith("." + self.target)
                                and SUBDOMAIN_SPEC.validator(alias)
                                and (aliases is None or alias not in aliases)
                            ):
                                if aliases is None:
                                    aliases = evidence["aliases"] = []
                                aliases.append(alias)
                except dns.resolver.NXDOMAIN:
                    status = "nxdomain"
                    if probe and addresses:
                        # An address followed by name-nonexistence is not a
                        # complete, stable wildcard signature. Preserve the
                        # ordinary candidate's partial-positive semantics.
                        error = "inconsistent wildcard probe: address answer followed by NXDOMAIN"
                        errors.append(error)
                    break
                except dns.resolver.NoAnswer:
                    status = "no_answer"
                    continue
                except (dns.exception.DNSException, OSError) as exc:
                    error = safe_exception_text(exc, 180)
                    errors.append(error)
                finally:
                    if evidence is not None:
                        evidence["records"][kind] = {
                            "status": status,
                            "addresses": list(query_addresses[:100]),
                            "error": error,
                        }
                    self.event(
                        "dns_end",
                        {
                            **payload,
                            "status": status,
                            "addresses": query_addresses,
                            "error": error,
                            "elapsed_seconds": max(0, time.monotonic() - query_started),
                        },
                    )
            except (dns.exception.DNSException, OSError) as exc:
                errors.append(safe_exception_text(exc, 180))
        response_time = max(0.0, time.monotonic() - started)
        if evidence is not None:
            evidence["duration_ms"] = response_time * 1000
        return ScanDNSResult(
            name,
            bool(addresses),
            sorted(addresses),
            response_time,
            # A partial positive candidate still establishes an address, but
            # wildcard calibration requires complete A/AAAA observations. A
            # timed-out record type may conceal a rotating wildcard pool; do
            # not cache the other type's answer as a stable complete signature.
            error="; ".join(errors) if errors and (probe or not addresses) else None,
            cancelled=cancelled,
            dns_evidence=evidence,
        )

    def _wildcard(
        self, parent: str, deadline: float | None, *, candidate: str | None = None
    ) -> tuple[set[str], bool, str | None]:
        with self._cache_lock:
            if parent in self._wildcards:
                return self._wildcards[parent]
            # Overflow parents still receive ordinary conservative calibration,
            # but retain neither a lock nor a result. Worker/rate/deadline bounds
            # continue to apply, and active cached-parent locks are never evicted.
            cacheable = parent in self._parent_locks or len(self._parent_locks) < 1000
            lock = self._parent_locks.get(parent)
            if lock is None:
                lock = threading.Lock()
                if cacheable:
                    self._parent_locks[parent] = lock
        with lock:
            if self._cancel.is_set():
                return set(), False, "scan cancelled"
            if parent in self._wildcards:
                return self._wildcards[parent]
            probes = [
                self._resolve(
                    f"enumeraite-{secrets.token_hex(8)}.{parent}",
                    deadline,
                    probe=True,
                    candidate=candidate,
                )
                for _ in range(2)
            ]
            addresses = {ip for probe in probes for ip in probe.ip_addresses}
            # Stable wildcards need not hide explicit records with distinct IPs.
            # Differing positive answers indicate a rotating wildcard even
            # when they overlap (for example rotating A plus a stable AAAA).
            # A candidate's previously unseen pool address is not evidence of
            # an explicit host, so classify these answers conservatively.
            rotating = all(probe.exists for probe in probes) and set(probes[0].ip_addresses) != set(
                probes[1].ip_addresses
            )
            error = (
                "scan cancelled"
                if any(probe.cancelled for probe in probes)
                else next((probe.error for probe in probes if probe.error), None)
            )
            if not error and probes[0].exists != probes[1].exists:
                # One positive and one conclusive empty observation cannot
                # establish a stable pool or a consistently negative parent.
                # Do not verify an unseen pool address or cache this transient
                # inconsistency as a permanent wildcard classification.
                error = "inconsistent wildcard probe address presence"
            result = addresses, rotating, error
            # Incomplete/inconsistent calibration is retryable, not zone truth.
            if not error and cacheable:
                with self._cache_lock:
                    self._wildcards[parent] = result
            return result

    def _validate(self, name: str, deadline: float | None) -> DNSResult:
        if self._cancel.is_set():
            return ScanDNSResult(name, False, [], 0, error="scan cancelled", cancelled=True)
        if not SUBDOMAIN_SPEC.validator(name) or not name.endswith("." + self.target):
            return DNSResult(name, False, [], 0, error="out of scope or invalid DNS name")
        result = self._resolve(name, deadline)
        if result.cancelled or not result.exists:
            return result
        parent = name.split(".", 1)[1]
        addresses, rotating, error = self._wildcard(parent, deadline, candidate=name)
        if error:
            result.exists = False
            result.error = "wildcard calibration inconclusive: " + error
            result.cancelled = error == "scan cancelled"
            wildcard_status = "inconclusive"
        elif rotating or (addresses and set(result.ip_addresses).issubset(addresses)):
            result.exists = False
            result.wildcard = True
            wildcard_status = "matched"
        else:
            wildcard_status = "different_answer" if addresses else "not_detected"
        if result.dns_evidence is not None:
            result.dns_evidence["wildcard"]["status"] = wildcard_status
        return result

    def validate(self, names: Sequence[str], *, deadline: float | None) -> list[DNSResult]:
        """Share rate/deadline limits and wake queued workers when interrupted.

        A missing scan deadline keeps each resolver call and rate slot bounded.
        Cancellation stops new requests and wakes every reserved rate slot.
        Already-running DNS calls unwind within their existing resolver lifetime
        (two seconds by default), after which this validator may be reused.
        """
        # One batch owns the cancellation event; concurrent calls on the same
        # instance must not reset it while an interrupted batch unwinds.
        with self._validation_lock:
            if self._owns_cancel:
                self._cancel.clear()
            executor = ThreadPoolExecutor(max_workers=self.workers)
            interrupted = False
            try:
                return list(executor.map(lambda value: self._validate(value, deadline), names))
            except BaseException:
                interrupted = True
                self._cancel.set()
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=interrupted)
                if interrupted or self._cancel.is_set():
                    with self._rate_lock:
                        # Cancelled workers never spent their reserved slots.
                        # Preserve spacing after the last actual query while
                        # discarding potentially minutes of queued delays.
                        self._next_query = max(
                            time.monotonic(),
                            self._last_query + 1 / self.rate if self._last_query is not None else 0,
                        )
                if self._owns_cancel:
                    self._cancel.clear()
