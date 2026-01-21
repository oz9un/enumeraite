"""Tests for HTTP path validator."""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from enumeraite.core.validator import HTTPValidator
from enumeraite.core.models import ValidationResult

def test_validator_init():
    """Test validator initialization."""
    validator = HTTPValidator(timeout=15, max_concurrent=10)
    assert validator.timeout == 15
    assert validator.max_concurrent == 10

@pytest.mark.asyncio
async def test_validate_path_exception_handling():
    """Test that exceptions are handled gracefully."""
    validator = HTTPValidator()

    # Mock aiohttp.ClientSession to raise an exception
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session_class.side_effect = Exception("Connection error")

        result = await validator.validate_path("https://example.com/api/test")

        assert result.exists == False
        assert result.status_code is None
        assert result.path == "https://example.com/api/test"

@pytest.mark.asyncio
async def test_validate_paths_basic():
    """Test basic validate_paths functionality without complex mocking."""
    validator = HTTPValidator()

    # Mock the individual validate_path method instead
    async def mock_validate_path(url, method="GET"):
        if "test1" in url:
            return ValidationResult(path=url, status_code=200, exists=True, method=method)
        else:
            return ValidationResult(path=url, status_code=404, exists=False, method=method)

    # Replace the method temporarily
    original_validate_path = validator.validate_path
    validator.validate_path = mock_validate_path

    try:
        results = await validator.validate_paths(
            "https://example.com",
            ["/api/test1", "/api/test2"],
            ["GET"]
        )

        # Should return only the successful one
        assert len(results) == 1
        assert results[0].path == "https://example.com/api/test1"

    finally:
        # Restore original method
        validator.validate_path = original_validate_path

def test_is_success_status():
    """Test status code evaluation logic."""
    validator = HTTPValidator()

    # Should consider these as existing
    assert validator._is_success_status(200) == True
    assert validator._is_success_status(301) == True
    assert validator._is_success_status(500) == True

    # Should consider these as not existing
    assert validator._is_success_status(404) == False
    assert validator._is_success_status(403) == False