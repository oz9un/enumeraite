"""DNS validation for subdomain discovery."""

import math
import secrets
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import dns.exception
import dns.resolver

from .prompt_spec import SUBDOMAIN_SPEC
from .text_safety import safe_exception_text, safe_model_text

_MAX_VALIDATION_ITEMS = 10_000


@dataclass
class DNSResult:
    """Result of DNS validation."""

    subdomain: str
    exists: bool
    ip_addresses: List[str]
    response_time: float
    error: Optional[str] = None
    wildcard: bool = False


@dataclass
class WildcardDNS:
    """Wildcard DNS evidence collected from random labels."""

    apex: str
    detected: bool
    ip_addresses: List[str]
    error: Optional[str] = None


class DNSValidator:
    """DNS validator for checking subdomain existence."""

    def __init__(self, timeout: float = 5.0, max_concurrent: int = 50):
        """Initialize DNS validator.

        Args:
            timeout: DNS query timeout in seconds
            max_concurrent: Maximum number of concurrent DNS queries
        """
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("DNS timeout must be finite and positive")
        if (
            isinstance(max_concurrent, bool)
            or not isinstance(max_concurrent, int)
            or not 1 <= max_concurrent <= 1_000
        ):
            raise ValueError("DNS max_concurrent must be an integer between 1 and 1000")
        self.timeout = timeout
        self.max_concurrent = max_concurrent

    def validate_subdomains(
        self,
        subdomains: List[str],
        detect_wildcard: bool = False,
        apex: Optional[str] = None,
    ) -> List[DNSResult]:
        """Validate multiple subdomains using concurrent DNS lookups.

        Args:
            subdomains: List of subdomains to validate

        Returns:
            List of DNSResult objects with validation results
        """
        if isinstance(subdomains, (str, bytes)) or not isinstance(subdomains, list):
            raise ValueError("subdomains must be a list of DNS names")
        if not all(isinstance(value, str) for value in subdomains):
            raise ValueError("subdomains must contain only DNS-name strings")
        if len(subdomains) > _MAX_VALIDATION_ITEMS:
            raise ValueError("subdomains exceed the 10000-item validation limit")
        if not isinstance(detect_wildcard, bool):
            raise ValueError("detect_wildcard must be a boolean")
        results: List[DNSResult] = []

        # DNS names are case-insensitive. Normalize and reject malformed input
        # before it reaches the resolver while preserving stable output order.
        unique_subdomains = list(
            dict.fromkeys(value.strip().lower().rstrip(".") for value in subdomains if value)
        )
        valid_subdomains = [
            subdomain for subdomain in unique_subdomains if SUBDOMAIN_SPEC.validator(subdomain)
        ]
        results.extend(
            DNSResult(
                subdomain=subdomain,
                exists=False,
                ip_addresses=[],
                response_time=0.0,
                error="invalid DNS name",
            )
            for subdomain in unique_subdomains
            if subdomain not in valid_subdomains
        )

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all DNS lookup tasks
            future_to_subdomain = {
                executor.submit(self._validate_single_subdomain, subdomain): subdomain
                for subdomain in valid_subdomains
            }

            # Collect results as they complete
            for future in as_completed(future_to_subdomain):
                subdomain = future_to_subdomain[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Create error result for failed validation
                    results.append(
                        DNSResult(
                            subdomain=subdomain,
                            exists=False,
                            ip_addresses=[],
                            response_time=0.0,
                            error=safe_exception_text(e, 300),
                        )
                    )

        if detect_wildcard and valid_subdomains:
            if not apex:
                raise ValueError("apex is required when wildcard detection is enabled")
            if not isinstance(apex, str):
                raise ValueError("apex must be a DNS-name string")
            normalized_apex = apex.strip().lower().rstrip(".")
            if not SUBDOMAIN_SPEC.validator(normalized_apex):
                raise ValueError("apex must be a valid DNS name")
            wildcard = self.detect_wildcard(normalized_apex)
            wildcard_addresses = set(wildcard.ip_addresses)
            if wildcard.detected:
                for result in results:
                    if (
                        result.exists
                        and result.subdomain.endswith("." + normalized_apex)
                        and set(result.ip_addresses)
                        and set(result.ip_addresses).issubset(wildcard_addresses)
                    ):
                        result.wildcard = True
                        result.exists = False
            elif wildcard.error:
                for result in results:
                    if result.exists and result.subdomain.endswith("." + normalized_apex):
                        result.exists = False
                        result.error = wildcard.error

        # Sort results to match original order
        subdomain_to_result = {r.subdomain: r for r in results}
        return [
            subdomain_to_result[subdomain]
            for subdomain in unique_subdomains
            if subdomain in subdomain_to_result
        ]

    def detect_wildcard(self, apex: str, probes: int = 3) -> WildcardDNS:
        """Resolve random labels and report stable wildcard DNS addresses."""
        if not isinstance(apex, str):
            raise ValueError("apex must be a DNS-name string")
        apex = apex.strip().lower().rstrip(".")
        if not SUBDOMAIN_SPEC.validator(apex):
            return WildcardDNS(apex=apex, detected=False, ip_addresses=[])
        if isinstance(probes, bool) or not isinstance(probes, int) or not 2 <= probes <= 10:
            raise ValueError("wildcard probes must be an integer between 2 and 10")
        samples: List[Set[str]] = []
        errors: List[str] = []
        for _ in range(probes):
            hostname = f"enumeraite-{secrets.token_hex(8)}.{apex}"
            result = self._validate_single_subdomain(hostname)
            samples.append(set(result.ip_addresses) if result.exists else set())
            if result.error:
                errors.append(result.error)
        nonempty = [sample for sample in samples if sample]
        if len(nonempty) < 2:
            error = "wildcard DNS calibration inconclusive" if errors else None
            return WildcardDNS(
                apex=apex,
                detected=False,
                ip_addresses=[],
                error=error,
            )
        # Different random labels resolving at all is wildcard evidence even
        # when an anycast/load-balanced pool rotates addresses between probes.
        union = set().union(*nonempty)
        return WildcardDNS(apex=apex, detected=True, ip_addresses=sorted(union))

    def _validate_single_subdomain(self, subdomain: str) -> DNSResult:
        """Validate a single subdomain via DNS lookup.

        Args:
            subdomain: Subdomain to validate

        Returns:
            DNSResult with validation information
        """
        start_time = time.perf_counter()

        try:
            # A resolver instance supplies real per-query timeouts. The old
            # socket.setdefaulttimeout approach mutated process-global state
            # from worker threads and did not reliably bound getaddrinfo.
            resolver = dns.resolver.Resolver(configure=True)
            resolver.timeout = self.timeout
            resolver.lifetime = self.timeout
            ip_addresses: List[str] = []
            transient_errors: List[str] = []
            authoritative_nxdomain = False
            for record_type in ("A", "AAAA"):
                try:
                    answer = resolver.resolve(
                        subdomain,
                        record_type,
                        raise_on_no_answer=False,
                        lifetime=self.timeout,
                    )
                    ip_addresses.extend(str(item) for item in answer)
                except dns.resolver.NXDOMAIN:
                    # NXDOMAIN is authoritative for the name, not merely this
                    # record type, so a second query cannot change the result.
                    authoritative_nxdomain = True
                    transient_errors.clear()
                    break
                except dns.resolver.NoAnswer:
                    continue
                except dns.resolver.NoNameservers:
                    transient_errors.append(f"{record_type}: no nameservers available")
                except dns.exception.Timeout:
                    transient_errors.append(f"{record_type}: DNS query timed out")

            response_time = time.perf_counter() - start_time

            # Remove duplicates from IP addresses
            ip_addresses = list(dict.fromkeys(ip_addresses))
            error = (
                None
                if ip_addresses or authoritative_nxdomain or not transient_errors
                else safe_model_text("; ".join(transient_errors), 300)
            )

            return DNSResult(
                subdomain=subdomain,
                exists=len(ip_addresses) > 0,
                ip_addresses=ip_addresses,
                response_time=response_time,
                error=error,
            )

        except Exception as e:
            response_time = time.perf_counter() - start_time
            return DNSResult(
                subdomain=subdomain,
                exists=False,
                ip_addresses=[],
                response_time=response_time,
                error=safe_exception_text(e, 300),
            )

    def filter_existing_subdomains(self, subdomains: List[str]) -> List[str]:
        """Filter subdomains to only return those that exist in DNS.

        Args:
            subdomains: List of subdomains to check

        Returns:
            List of subdomains that exist in DNS
        """
        results = self.validate_subdomains(subdomains)
        return [result.subdomain for result in results if result.exists]

    def get_validation_stats(self, results: List[DNSResult]) -> Dict[str, Any]:
        """Get statistics from DNS validation results.

        Args:
            results: List of DNS validation results

        Returns:
            Dictionary with validation statistics
        """
        if not results:
            return {}

        existing_count = sum(1 for r in results if r.exists)
        error_count = sum(1 for r in results if r.error)
        avg_response_time = sum(r.response_time for r in results) / len(results)

        return {
            "total_checked": len(results),
            "existing": existing_count,
            "not_found": len(results) - existing_count - error_count,
            "errors": error_count,
            "success_rate": (existing_count / len(results)) * 100,
            "avg_response_time": avg_response_time,
        }
