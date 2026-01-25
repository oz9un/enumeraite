"""DNS validation for subdomain discovery."""
import asyncio
import socket
import time
from typing import List, Dict, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class DNSResult:
    """Result of DNS validation."""
    subdomain: str
    exists: bool
    ip_addresses: List[str]
    response_time: float
    error: Optional[str] = None


class DNSValidator:
    """DNS validator for checking subdomain existence."""

    def __init__(self, timeout: float = 5.0, max_concurrent: int = 50):
        """Initialize DNS validator.

        Args:
            timeout: DNS query timeout in seconds
            max_concurrent: Maximum number of concurrent DNS queries
        """
        self.timeout = timeout
        self.max_concurrent = max_concurrent

    def validate_subdomains(self, subdomains: List[str]) -> List[DNSResult]:
        """Validate multiple subdomains using concurrent DNS lookups.

        Args:
            subdomains: List of subdomains to validate

        Returns:
            List of DNSResult objects with validation results
        """
        results = []

        # Remove duplicates while preserving order
        unique_subdomains = list(dict.fromkeys(subdomains))

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all DNS lookup tasks
            future_to_subdomain = {
                executor.submit(self._validate_single_subdomain, subdomain): subdomain
                for subdomain in unique_subdomains
            }

            # Collect results as they complete
            for future in as_completed(future_to_subdomain):
                subdomain = future_to_subdomain[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Create error result for failed validation
                    results.append(DNSResult(
                        subdomain=subdomain,
                        exists=False,
                        ip_addresses=[],
                        response_time=0.0,
                        error=str(e)
                    ))

        # Sort results to match original order
        subdomain_to_result = {r.subdomain: r for r in results}
        return [subdomain_to_result[subdomain] for subdomain in unique_subdomains if subdomain in subdomain_to_result]

    def _validate_single_subdomain(self, subdomain: str) -> DNSResult:
        """Validate a single subdomain via DNS lookup.

        Args:
            subdomain: Subdomain to validate

        Returns:
            DNSResult with validation information
        """
        start_time = time.time()

        try:
            # Perform DNS lookup with timeout
            socket.setdefaulttimeout(self.timeout)
            ip_addresses = []

            # Try A record lookup
            try:
                result = socket.getaddrinfo(subdomain, None, socket.AF_INET)
                ip_addresses.extend([addr[4][0] for addr in result])
            except socket.gaierror:
                pass

            # Try AAAA record lookup (IPv6)
            try:
                result = socket.getaddrinfo(subdomain, None, socket.AF_INET6)
                ip_addresses.extend([addr[4][0] for addr in result])
            except socket.gaierror:
                pass

            response_time = time.time() - start_time

            # Remove duplicates from IP addresses
            ip_addresses = list(dict.fromkeys(ip_addresses))

            return DNSResult(
                subdomain=subdomain,
                exists=len(ip_addresses) > 0,
                ip_addresses=ip_addresses,
                response_time=response_time
            )

        except Exception as e:
            response_time = time.time() - start_time
            return DNSResult(
                subdomain=subdomain,
                exists=False,
                ip_addresses=[],
                response_time=response_time,
                error=str(e)
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

    def get_validation_stats(self, results: List[DNSResult]) -> Dict[str, any]:
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
            "avg_response_time": avg_response_time
        }