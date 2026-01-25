"""HTTP validation for subdomain discovery."""
import asyncio
import time
import requests
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class HTTPResult:
    """Result of HTTP validation."""
    subdomain: str
    url: str
    accessible: bool
    status_code: Optional[int]
    response_time: float
    protocol: str  # 'http' or 'https'
    title: Optional[str] = None
    server: Optional[str] = None
    error: Optional[str] = None


class HTTPValidator:
    """HTTP validator for checking subdomain web accessibility."""

    def __init__(self, timeout: float = 10.0, max_concurrent: int = 20,
                 user_agent: str = "Enumeraite/0.1.0", verify_ssl: bool = False):
        """Initialize HTTP validator.

        Args:
            timeout: HTTP request timeout in seconds
            max_concurrent: Maximum number of concurrent HTTP requests
            user_agent: User-Agent header to use
            verify_ssl: Whether to verify SSL certificates
        """
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.user_agent = user_agent
        self.verify_ssl = verify_ssl

    def validate_subdomains(self, subdomains: List[str], check_both_protocols: bool = True) -> List[HTTPResult]:
        """Validate multiple subdomains using concurrent HTTP requests.

        Args:
            subdomains: List of subdomains to validate
            check_both_protocols: Whether to check both HTTP and HTTPS

        Returns:
            List of HTTPResult objects with validation results
        """
        results = []

        # Remove duplicates while preserving order
        unique_subdomains = list(dict.fromkeys(subdomains))

        # Generate URLs to test
        urls_to_test = []
        for subdomain in unique_subdomains:
            if check_both_protocols:
                urls_to_test.append((f"https://{subdomain}", subdomain, "https"))
                urls_to_test.append((f"http://{subdomain}", subdomain, "http"))
            else:
                # Default to HTTPS first
                urls_to_test.append((f"https://{subdomain}", subdomain, "https"))

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all HTTP request tasks
            future_to_url = {
                executor.submit(self._validate_single_url, url, subdomain, protocol): (url, subdomain, protocol)
                for url, subdomain, protocol in urls_to_test
            }

            # Collect results as they complete
            for future in as_completed(future_to_url):
                url, subdomain, protocol = future_to_url[future]
                try:
                    result = future.result()
                    if result.accessible or not check_both_protocols:
                        results.append(result)
                except Exception as e:
                    # Create error result for failed validation
                    results.append(HTTPResult(
                        subdomain=subdomain,
                        url=url,
                        accessible=False,
                        status_code=None,
                        response_time=0.0,
                        protocol=protocol,
                        error=str(e)
                    ))

        # If checking both protocols, prefer HTTPS results
        if check_both_protocols:
            results = self._deduplicate_results(results)

        return results

    def _validate_single_url(self, url: str, subdomain: str, protocol: str) -> HTTPResult:
        """Validate a single URL via HTTP request.

        Args:
            url: Full URL to test
            subdomain: Original subdomain
            protocol: Protocol used (http/https)

        Returns:
            HTTPResult with validation information
        """
        start_time = time.time()

        try:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }

            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
                allow_redirects=True
            )

            response_time = time.time() - start_time

            # Extract page title if available
            title = None
            server = None
            try:
                if 'text/html' in response.headers.get('content-type', ''):
                    content = response.text.lower()
                    title_start = content.find('<title>')
                    if title_start != -1:
                        title_end = content.find('</title>', title_start)
                        if title_end != -1:
                            title = response.text[title_start + 7:title_end].strip()
                            if len(title) > 100:  # Limit title length
                                title = title[:97] + "..."

                server = response.headers.get('Server', None)
            except Exception:
                pass  # Title extraction failed, continue without it

            return HTTPResult(
                subdomain=subdomain,
                url=url,
                accessible=True,
                status_code=response.status_code,
                response_time=response_time,
                protocol=protocol,
                title=title,
                server=server
            )

        except Exception as e:
            response_time = time.time() - start_time
            return HTTPResult(
                subdomain=subdomain,
                url=url,
                accessible=False,
                status_code=None,
                response_time=response_time,
                protocol=protocol,
                error=str(e)
            )

    def _deduplicate_results(self, results: List[HTTPResult]) -> List[HTTPResult]:
        """Deduplicate results, preferring HTTPS over HTTP for the same subdomain.

        Args:
            results: List of HTTP results to deduplicate

        Returns:
            Deduplicated list preferring HTTPS results
        """
        subdomain_results = {}

        for result in results:
            subdomain = result.subdomain

            if subdomain not in subdomain_results:
                subdomain_results[subdomain] = result
            else:
                # Prefer accessible results over non-accessible
                existing = subdomain_results[subdomain]
                if not existing.accessible and result.accessible:
                    subdomain_results[subdomain] = result
                # Prefer HTTPS over HTTP if both are accessible
                elif existing.accessible == result.accessible and result.protocol == 'https':
                    subdomain_results[subdomain] = result

        return list(subdomain_results.values())

    def filter_accessible_subdomains(self, subdomains: List[str]) -> List[str]:
        """Filter subdomains to only return those that are HTTP accessible.

        Args:
            subdomains: List of subdomains to check

        Returns:
            List of subdomains that are accessible via HTTP/HTTPS
        """
        results = self.validate_subdomains(subdomains)
        return [result.subdomain for result in results if result.accessible]

    def get_validation_stats(self, results: List[HTTPResult]) -> Dict[str, any]:
        """Get statistics from HTTP validation results.

        Args:
            results: List of HTTP validation results

        Returns:
            Dictionary with validation statistics
        """
        if not results:
            return {}

        accessible_count = sum(1 for r in results if r.accessible)
        error_count = sum(1 for r in results if r.error)
        https_count = sum(1 for r in results if r.accessible and r.protocol == 'https')
        avg_response_time = sum(r.response_time for r in results if r.accessible) / max(accessible_count, 1)

        status_codes = {}
        for result in results:
            if result.accessible and result.status_code:
                code = result.status_code
                status_codes[code] = status_codes.get(code, 0) + 1

        return {
            "total_checked": len(results),
            "accessible": accessible_count,
            "not_accessible": len(results) - accessible_count - error_count,
            "errors": error_count,
            "https_available": https_count,
            "success_rate": (accessible_count / len(results)) * 100 if results else 0,
            "avg_response_time": avg_response_time,
            "status_codes": status_codes
        }