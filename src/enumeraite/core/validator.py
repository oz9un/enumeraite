"""HTTP path validator for testing generated endpoints."""
import asyncio
import aiohttp
import time
from typing import List, Optional
from .models import ValidationResult

class HTTPValidator:
    """Async HTTP validator for testing API path existence."""

    def __init__(self, timeout: int = 10, max_concurrent: int = 20):
        """Initialize validator with configuration.

        Args:
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent requests
        """
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def validate_paths(self, base_url: str, paths: List[str],
                           methods: List[str] = None) -> List[ValidationResult]:
        """Validate multiple paths via HTTP requests.

        Args:
            base_url: Base URL for the target
            paths: List of paths to validate
            methods: HTTP methods to test (default: ["GET"])

        Returns:
            List of ValidationResult objects for paths that appear to exist
        """
        if methods is None:
            methods = ["GET"]

        tasks = []
        for path in paths:
            for method in methods:
                full_url = f"{base_url.rstrip('/')}{path}"
                task = self.validate_path(full_url, method)
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results, group by path
        path_results = {}
        for i, result in enumerate(results):
            if isinstance(result, ValidationResult) and result.exists:
                path = paths[i % len(paths)]
                if path not in path_results or result.status_code < 400:
                    path_results[path] = result

        return list(path_results.values())

    async def validate_path(self, url: str, method: str = "GET") -> ValidationResult:
        """Validate a single path via HTTP request.

        Args:
            url: Full URL to test
            method: HTTP method to use

        Returns:
            ValidationResult with test outcome
        """
        async with self.semaphore:
            start_time = time.time()

            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.request(method, url) as response:
                        response_time = time.time() - start_time

                        return ValidationResult(
                            path=url,
                            status_code=response.status,
                            exists=self._is_success_status(response.status),
                            method=method,
                            response_time=response_time
                        )
            except Exception as e:
                return ValidationResult(
                    path=url,
                    status_code=None,
                    exists=False,
                    method=method,
                    response_time=time.time() - start_time
                )

    def _is_success_status(self, status_code: int) -> bool:
        """Determine if status code indicates path exists.

        Args:
            status_code: HTTP status code

        Returns:
            True if path appears to exist, False otherwise
        """
        # Consider anything not 404/403 as potentially existing
        return status_code not in [404, 403]