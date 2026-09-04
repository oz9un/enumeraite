"""HTTP validation for subdomain discovery."""

import math
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
from urllib.parse import urlunsplit

import requests

from .. import __version__
from .prompt_spec import PATH_SPEC, SUBDOMAIN_SPEC
from .text_safety import safe_exception_text, safe_model_text
from .url_safety import parse_http_url

DEFAULT_USER_AGENT = f"Enumeraite/{__version__}"
_MAX_VALIDATION_ITEMS = 10_000


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


@dataclass
class PathHTTPResult:
    """Result of HTTP endpoint validation with soft-404 classification."""

    path: str
    url: str
    status_code: Optional[int]
    exists: bool
    response_time: float
    soft_404_similarity: Optional[float] = None
    error: Optional[str] = None


@dataclass
class _PathResponse:
    status_code: int
    text: str
    location: str = ""


class PathHTTPValidator:
    """Validate paths against a target while suppressing soft-404 responses."""

    def __init__(
        self,
        timeout: float = 10.0,
        max_concurrent: int = 20,
        user_agent: str = DEFAULT_USER_AGENT,
        verify_ssl: bool = True,
        soft_404_threshold: float = 0.92,
        baseline_samples: int = 2,
        baseline_depth: int = 1,
        max_baseline_groups: int = 32,
        max_body_bytes: int = 20_000,
    ):
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("HTTP timeout must be finite and positive")
        if (
            isinstance(max_concurrent, bool)
            or not isinstance(max_concurrent, int)
            or not 1 <= max_concurrent <= 1_000
        ):
            raise ValueError("HTTP max_concurrent must be an integer between 1 and 1000")
        if (
            isinstance(soft_404_threshold, bool)
            or not isinstance(soft_404_threshold, (int, float))
            or not math.isfinite(soft_404_threshold)
            or not 0 <= soft_404_threshold <= 1
        ):
            raise ValueError("soft_404_threshold must be between 0 and 1")
        if not isinstance(verify_ssl, bool):
            raise ValueError("verify_ssl must be a boolean")
        if (
            isinstance(baseline_depth, bool)
            or not isinstance(baseline_depth, int)
            or not 0 <= baseline_depth <= 10
        ):
            raise ValueError("baseline_depth must be an integer between 0 and 10")
        if (
            isinstance(max_baseline_groups, bool)
            or not isinstance(max_baseline_groups, int)
            or not 1 <= max_baseline_groups <= 100
        ):
            raise ValueError("max_baseline_groups must be an integer between 1 and 100")
        if (
            isinstance(baseline_samples, bool)
            or not isinstance(baseline_samples, int)
            or not 1 <= baseline_samples <= 5
        ):
            raise ValueError("baseline_samples must be an integer between 1 and 5")
        if (
            isinstance(max_body_bytes, bool)
            or not isinstance(max_body_bytes, int)
            or not 1_024 <= max_body_bytes <= 10_000_000
        ):
            raise ValueError("max_body_bytes must be between 1024 and 10000000")
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.user_agent = user_agent
        self.verify_ssl = verify_ssl
        self.soft_404_threshold = soft_404_threshold
        self.baseline_samples = baseline_samples
        self.baseline_depth = baseline_depth
        self.max_baseline_groups = max_baseline_groups
        self.max_body_bytes = max_body_bytes

    def validate_paths(self, target: str, paths: List[str]) -> List[PathHTTPResult]:
        if isinstance(paths, (str, bytes)) or not isinstance(paths, list):
            raise ValueError("paths must be a list of path candidates")
        if not all(isinstance(path, str) for path in paths):
            raise ValueError("paths must contain only path strings")
        if len(paths) > _MAX_VALIDATION_ITEMS:
            raise ValueError("paths exceed the 10000-item validation limit")
        base = _normalize_base_url(target)
        unique_paths = list(dict.fromkeys(paths))
        valid_paths = [path for path in unique_paths if PATH_SPEC.validator(path)]
        valid_path_set = set(valid_paths)
        results: List[PathHTTPResult] = [
            PathHTTPResult(
                path,
                base,
                None,
                False,
                0.0,
                error="invalid or unsafe path candidate",
            )
            for path in unique_paths
            if path not in valid_path_set
        ]
        prefixes = list(
            dict.fromkeys(_baseline_prefix(path, self.baseline_depth) for path in valid_paths)
        )
        selected_prefixes = prefixes[: self.max_baseline_groups]
        if (
            any(prefix not in selected_prefixes for prefix in prefixes)
            and "" not in selected_prefixes
        ):
            # Reserve one of the configured group slots for a root fallback;
            # do not silently exceed the traffic cap by appending a 33rd group.
            selected_prefixes[-1] = ""
        baselines_by_prefix = self._fetch_baseline_groups(base, selected_prefixes)
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_path = {
                executor.submit(
                    self._validate_one,
                    base,
                    path,
                    baselines_by_prefix.get(
                        _baseline_prefix(path, self.baseline_depth),
                        baselines_by_prefix.get("", []),
                    ),
                ): path
                for path in valid_paths
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        PathHTTPResult(
                            path,
                            base + path,
                            None,
                            False,
                            0.0,
                            error=safe_exception_text(exc, 300),
                        )
                    )
        by_path = {result.path: result for result in results}
        return [by_path[path] for path in unique_paths if path in by_path]

    def _fetch_baseline_groups(self, base: str, prefixes: List[str]):
        results: dict[str, list[tuple[Any, float, Optional[str]]]] = {
            prefix: [] for prefix in prefixes
        }
        requests_to_make = [
            (
                prefix,
                base + prefix + f"/enumeraite-not-found-{secrets.token_hex(10)}",
            )
            for prefix in prefixes
            for _ in range(self.baseline_samples)
        ]
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_prefix = {
                executor.submit(self._fetch, url): prefix for prefix, url in requests_to_make
            }
            for future in as_completed(future_to_prefix):
                prefix = future_to_prefix[future]
                try:
                    results[prefix].append(future.result())
                except Exception as exc:
                    results[prefix].append((None, 0.0, safe_exception_text(exc, 300)))
        return results

    def _validate_one(self, base: str, path: str, baselines) -> PathHTTPResult:
        if not PATH_SPEC.validator(path):
            return PathHTTPResult(
                path,
                base,
                None,
                False,
                0.0,
                error="invalid or unsafe path candidate",
            )
        url = base + path
        response, elapsed, error = self._fetch(url)
        if error or response is None:
            return PathHTTPResult(path, url, None, False, elapsed, error=error)
        status = response.status_code
        similarity = None
        # Any response other than an authoritative 404 can be route evidence:
        # APIs commonly use 400/405/415/422 when a path exists but GET lacks
        # the required method, headers, or body. Calibration decides whether
        # that response is path-specific or merely a wildcard/global handler.
        exists = status != 404
        usable_baselines = [
            baseline_response
            for baseline_response, _, baseline_error in baselines
            if not baseline_error and baseline_response is not None
        ]
        if exists and not usable_baselines:
            return PathHTTPResult(
                path,
                url,
                status,
                False,
                elapsed,
                error="soft-404 calibration unavailable",
            )
        if exists:
            similarities = [
                (
                    _redirect_similarity(
                        getattr(response, "location", ""),
                        getattr(baseline_response, "location", ""),
                    )
                    if 300 <= status < 400
                    else _body_similarity(response.text, baseline_response.text)
                )
                for baseline_response, _, baseline_error in baselines
                if not baseline_error
                and baseline_response is not None
                and status == baseline_response.status_code
            ]
            if similarities:
                similarity = max(similarities)
                if similarity >= self.soft_404_threshold:
                    exists = False
        if status == 404:
            exists = False
        return PathHTTPResult(path, url, status, exists, elapsed, similarity)

    def _fetch(self, url: str):
        started = time.perf_counter()
        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "*/*",
                    "Accept-Encoding": "identity",
                },
                timeout=self.timeout,
                verify=self.verify_ssl,
                allow_redirects=False,
                stream=True,
            )
            try:
                body = _bounded_response_bytes(
                    response,
                    self.max_body_bytes,
                    deadline=started + self.timeout,
                )
                encoding = response.encoding or "utf-8"
                text = body.decode(encoding, errors="replace")
                result = _PathResponse(
                    response.status_code,
                    text,
                    str(response.headers.get("Location", "")),
                )
            finally:
                response.close()
            return result, time.perf_counter() - started, None
        except Exception as exc:
            return None, time.perf_counter() - started, safe_exception_text(exc, 300)


def _body_similarity(left: str, right: str) -> float:
    """Compare bounded, whitespace-normalized response bodies."""

    def normalize(value: str) -> str:
        text = (value or "")[:20_000]
        text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<uuid>", text, flags=re.IGNORECASE)
        text = re.sub(r"\b[0-9a-f]{16,}\b", "<hex>", text, flags=re.IGNORECASE)
        text = re.sub(r"\b\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+Z?\b", "<timestamp>", text)
        text = re.sub(r"\b\d+\b", "<number>", text)
        return " ".join(text.split())

    left_body, right_body = normalize(left), normalize(right)
    if not left_body and not right_body:
        return 1.0
    return SequenceMatcher(None, left_body, right_body, autojunk=True).ratio()


def _redirect_similarity(left: str, right: str) -> float:
    def normalize(value: str) -> str:
        text = value.strip()
        text = re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27,}", "<uuid>", text, flags=re.IGNORECASE)
        text = re.sub(r"[0-9a-f]{16,}", "<hex>", text, flags=re.IGNORECASE)
        return re.sub(r"\d+", "<number>", text)

    left_location, right_location = normalize(left), normalize(right)
    return 1.0 if left_location == right_location else 0.0


def _baseline_prefix(path: str, depth: int) -> str:
    """Return the bounded virtual-router parent used for miss calibration."""
    parts = [part for part in path.strip("/").split("/") if part]
    parent_depth = min(depth, max(0, len(parts) - 1))
    return "/" + "/".join(parts[:parent_depth]) if parent_depth else ""


def _normalize_base_url(target: str) -> str:
    if not isinstance(target, str):
        raise ValueError("target URL must be a valid HTTP(S) URL")
    value = target.strip()
    if not value:
        raise ValueError("target must not be empty")
    if "://" not in value:
        value = "https://" + value
    parsed = parse_http_url(value, label="target URL")
    if not PATH_SPEC.validator(parsed.path or "/"):
        raise ValueError("target base URL contains an invalid or unsafe path")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


class HTTPValidator:
    """HTTP validator for checking subdomain web accessibility."""

    def __init__(
        self,
        timeout: float = 10.0,
        max_concurrent: int = 20,
        user_agent: str = DEFAULT_USER_AGENT,
        verify_ssl: bool = True,
        max_body_bytes: int = 100_000,
    ):
        """Initialize HTTP validator.

        Args:
            timeout: HTTP request timeout in seconds
            max_concurrent: Maximum number of concurrent HTTP requests
            user_agent: User-Agent header to use
            verify_ssl: Whether to verify SSL certificates
            max_body_bytes: Maximum response bytes read for title extraction
        """
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("HTTP timeout must be finite and positive")
        if (
            isinstance(max_concurrent, bool)
            or not isinstance(max_concurrent, int)
            or not 1 <= max_concurrent <= 1_000
        ):
            raise ValueError("HTTP max_concurrent must be an integer between 1 and 1000")
        if (
            isinstance(max_body_bytes, bool)
            or not isinstance(max_body_bytes, int)
            or not 1_024 <= max_body_bytes <= 10_000_000
        ):
            raise ValueError("max_body_bytes must be between 1024 and 10000000")
        if not isinstance(verify_ssl, bool):
            raise ValueError("verify_ssl must be a boolean")
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.user_agent = user_agent
        self.verify_ssl = verify_ssl
        self.max_body_bytes = max_body_bytes

    def validate_subdomains(
        self, subdomains: List[str], check_both_protocols: bool = True
    ) -> List[HTTPResult]:
        """Validate multiple subdomains using concurrent HTTP requests.

        Args:
            subdomains: List of subdomains to validate
            check_both_protocols: Whether to check both HTTP and HTTPS

        Returns:
            List of HTTPResult objects with validation results
        """
        if isinstance(subdomains, (str, bytes)) or not isinstance(subdomains, list):
            raise ValueError("subdomains must be a list of DNS names")
        if not all(isinstance(value, str) for value in subdomains):
            raise ValueError("subdomains must contain only DNS-name strings")
        if len(subdomains) > _MAX_VALIDATION_ITEMS:
            raise ValueError("subdomains exceed the 10000-item validation limit")
        if not isinstance(check_both_protocols, bool):
            raise ValueError("check_both_protocols must be a boolean")
        results: List[HTTPResult] = []

        # Hostnames are case-insensitive. Invalid values become explicit error
        # results and never reach URL construction or the network.
        unique_subdomains = list(
            dict.fromkeys(value.strip().lower().rstrip(".") for value in subdomains if value)
        )
        valid_subdomains = [
            subdomain for subdomain in unique_subdomains if SUBDOMAIN_SPEC.validator(subdomain)
        ]
        results.extend(
            HTTPResult(
                subdomain=subdomain,
                url="",
                accessible=False,
                status_code=None,
                response_time=0.0,
                protocol="",
                error="invalid DNS name",
            )
            for subdomain in unique_subdomains
            if subdomain not in valid_subdomains
        )

        # Generate URLs to test
        urls_to_test = []
        for subdomain in valid_subdomains:
            if check_both_protocols:
                urls_to_test.append((f"https://{subdomain}", subdomain, "https"))
                urls_to_test.append((f"http://{subdomain}", subdomain, "http"))
            else:
                # Default to HTTPS first
                urls_to_test.append((f"https://{subdomain}", subdomain, "https"))

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all HTTP request tasks
            future_to_url = {
                executor.submit(self._validate_single_url, url, subdomain, protocol): (
                    url,
                    subdomain,
                    protocol,
                )
                for url, subdomain, protocol in urls_to_test
            }

            # Collect results as they complete
            for future in as_completed(future_to_url):
                url, subdomain, protocol = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Create error result for failed validation
                    results.append(
                        HTTPResult(
                            subdomain=subdomain,
                            url=url,
                            accessible=False,
                            status_code=None,
                            response_time=0.0,
                            protocol=protocol,
                            error=safe_exception_text(e, 300),
                        )
                    )

        # If checking both protocols, prefer HTTPS results
        if check_both_protocols:
            results = self._deduplicate_results(results)
        by_subdomain = {result.subdomain: result for result in results}
        return [
            by_subdomain[subdomain] for subdomain in unique_subdomains if subdomain in by_subdomain
        ]

    def _validate_single_url(self, url: str, subdomain: str, protocol: str) -> HTTPResult:
        """Validate a single URL via HTTP request.

        Args:
            url: Full URL to test
            subdomain: Original subdomain
            protocol: Protocol used (http/https)

        Returns:
            HTTPResult with validation information
        """
        start_time = time.perf_counter()

        try:
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                # Keep streamed byte/deadline limits meaningful and avoid
                # spending the validation budget on decompression.
                "Accept-Encoding": "identity",
                "Connection": "keep-alive",
            }

            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
                # A redirect response proves the submitted host is reachable;
                # following it can leave the authorized target scope.
                allow_redirects=False,
                stream=True,
            )

            response_time = time.perf_counter() - start_time

            # Extract page title if available
            title = None
            server = None
            try:
                if "text/html" in response.headers.get("content-type", ""):
                    response_text = _bounded_response_text(
                        response,
                        self.max_body_bytes,
                        deadline=start_time + self.timeout,
                    )
                    content = response_text.lower()
                    title_start = content.find("<title>")
                    if title_start != -1:
                        title_end = content.find("</title>", title_start)
                        if title_end != -1:
                            title = safe_model_text(response_text[title_start + 7 : title_end], 100)

                server = safe_model_text(response.headers.get("Server", ""), 200) or None
            except Exception:
                title = None  # Title extraction failed; accessibility is still valid.
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()

            return HTTPResult(
                subdomain=subdomain,
                url=url,
                accessible=True,
                status_code=response.status_code,
                response_time=response_time,
                protocol=protocol,
                title=title,
                server=server,
            )

        except Exception as e:
            response_time = time.perf_counter() - start_time
            return HTTPResult(
                subdomain=subdomain,
                url=url,
                accessible=False,
                status_code=None,
                response_time=response_time,
                protocol=protocol,
                error=safe_exception_text(e, 300),
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
                elif existing.accessible == result.accessible and result.protocol == "https":
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

    def get_validation_stats(self, results: List[HTTPResult]) -> Dict[str, Any]:
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
        https_count = sum(1 for r in results if r.accessible and r.protocol == "https")
        avg_response_time = sum(r.response_time for r in results if r.accessible) / max(
            accessible_count, 1
        )

        status_codes: Dict[int, int] = {}
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
            "status_codes": status_codes,
        }


def _bounded_response_text(
    response: Any,
    max_body_bytes: int,
    *,
    deadline: float | None = None,
) -> str:
    """Read a response body without allowing an unbounded reconnaissance download."""
    if not hasattr(response, "iter_content"):
        return (
            str(getattr(response, "text", ""))
            .encode("utf-8")[:max_body_bytes]
            .decode("utf-8", errors="replace")
        )
    try:
        body = _bounded_response_bytes(response, max_body_bytes, deadline=deadline)
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    encoding = getattr(response, "encoding", None) or "utf-8"
    return body.decode(encoding, errors="replace")


def _bounded_response_bytes(
    response: Any,
    max_body_bytes: int,
    *,
    deadline: float | None,
) -> bytes:
    """Read a byte-capped stream and enforce a monotonic whole-body deadline."""
    raw = getattr(response, "raw", None)
    read_one = getattr(raw, "read1", None)
    iterator = None if callable(read_one) else iter(response.iter_content(chunk_size=4_096))
    body = bytearray()
    while len(body) < max_body_bytes:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("response body deadline exceeded")
        try:
            chunk = (
                # The request asks for identity encoding. Reading raw bytes
                # keeps a non-compliant peer from turning decompression into
                # an unbounded operation inside one deadline check.
                read_one(4_096, decode_content=False)
                if callable(read_one)
                else next(iterator)  # type: ignore[arg-type]
            )
        except StopIteration:
            break
        if not chunk:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("response body deadline exceeded")
        if not isinstance(chunk, (bytes, bytearray)):
            raise ValueError("HTTP response returned a non-byte body chunk")
        remaining = max_body_bytes - len(body)
        body.extend(chunk[:remaining])
    return bytes(body)
