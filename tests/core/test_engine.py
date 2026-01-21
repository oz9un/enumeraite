"""Tests for core generation engine."""
import pytest
from unittest.mock import Mock, AsyncMock
from enumeraite.core.engine import GenerationEngine
from enumeraite.core.models import GenerationResult, ValidationResult

def test_engine_initialization():
    """Test engine initialization with provider."""
    mock_provider = Mock()
    mock_validator = Mock()

    engine = GenerationEngine(mock_provider, mock_validator)

    assert engine.provider == mock_provider
    assert engine.validator == mock_validator
    assert len(engine._seen_paths) == 0

def test_generate_paths_basic():
    """Test basic path generation without validation."""
    mock_provider = Mock()
    mock_provider.generate_paths.return_value = GenerationResult(
        paths=["/api/user/settings", "/api/user/preferences"],
        confidence_scores=[0.8, 0.7],
        metadata={"provider": "test"}
    )

    engine = GenerationEngine(mock_provider)
    result = engine.generate_paths(["/api/user/profile"], "example.com", 10)

    assert len(result.paths) == 2
    assert "/api/user/settings" in result.paths
    mock_provider.generate_paths.assert_called_once_with(["/api/user/profile"], "example.com", 10)

def test_generate_paths_duplicate_filtering():
    """Test that duplicate paths are filtered out."""
    mock_provider = Mock()
    mock_provider.generate_paths.return_value = GenerationResult(
        paths=["/api/user/settings", "/api/user/settings", "/api/user/preferences"],
        confidence_scores=[0.8, 0.8, 0.7],
        metadata={"provider": "test"}
    )

    engine = GenerationEngine(mock_provider)
    result = engine.generate_paths(["/api/user/profile"], "example.com", 10)

    # Should deduplicate the duplicate "/api/user/settings"
    assert len(result.paths) == 2
    assert "/api/user/settings" in result.paths
    assert "/api/user/preferences" in result.paths

def test_generate_paths_filters_known_paths():
    """Test that known paths are filtered from results."""
    mock_provider = Mock()
    mock_provider.generate_paths.return_value = GenerationResult(
        paths=["/api/user/settings", "/api/user/profile", "/api/user/preferences"],
        confidence_scores=[0.8, 0.9, 0.7],
        metadata={"provider": "test"}
    )

    engine = GenerationEngine(mock_provider)
    known_paths = ["/api/user/profile"]
    result = engine.generate_paths(known_paths, "example.com", 10)

    # Should filter out the known path "/api/user/profile"
    assert len(result.paths) == 2
    assert "/api/user/profile" not in result.paths
    assert "/api/user/settings" in result.paths
    assert "/api/user/preferences" in result.paths

@pytest.mark.asyncio
async def test_generate_and_validate_paths():
    """Test combined generation and validation."""
    mock_provider = Mock()
    mock_provider.generate_paths.return_value = GenerationResult(
        paths=["/api/user/settings", "/api/user/preferences"],
        confidence_scores=[0.8, 0.7],
        metadata={"provider": "test"}
    )

    mock_validator = AsyncMock()
    mock_validator.validate_paths = AsyncMock(return_value=[
        ValidationResult(path="https://example.com/api/user/settings", status_code=200, exists=True, method="GET")
    ])

    engine = GenerationEngine(mock_provider, mock_validator)

    generation_result, validation_results = await engine.generate_and_validate_paths(
        ["/api/user/profile"], "example.com", 10, ["GET"]
    )

    assert len(generation_result.paths) == 2
    assert len(validation_results) == 1
    assert validation_results[0].exists == True
    mock_validator.validate_paths.assert_called_once()

@pytest.mark.asyncio
async def test_generate_and_validate_no_validator():
    """Test error when trying to validate without validator."""
    mock_provider = Mock()
    engine = GenerationEngine(mock_provider)  # No validator

    with pytest.raises(ValueError, match="No validator configured"):
        await engine.generate_and_validate_paths(["/api/user"], "example.com", 10)

def test_normalize_path():
    """Test path normalization."""
    mock_provider = Mock()
    engine = GenerationEngine(mock_provider)

    assert engine._normalize_path("/API/USER/") == "/api/user"
    assert engine._normalize_path("/api/user/profile") == "/api/user/profile"

def test_add_discovered_paths():
    """Test adding discovered paths to seen set."""
    mock_provider = Mock()
    engine = GenerationEngine(mock_provider)

    paths = ["/api/new/path1", "/api/new/path2"]
    engine.add_discovered_paths(paths)

    assert "/api/new/path1" in engine._seen_paths
    assert "/api/new/path2" in engine._seen_paths

def test_reset_seen_paths():
    """Test resetting the seen paths set."""
    mock_provider = Mock()
    engine = GenerationEngine(mock_provider)

    # Add some paths
    engine.add_discovered_paths(["/api/test"])
    assert len(engine._seen_paths) == 1

    # Reset
    engine.reset_seen_paths()
    assert len(engine._seen_paths) == 0