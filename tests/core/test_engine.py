"""Tests for core generation engine."""

from unittest.mock import ANY, AsyncMock, Mock

import pytest

from enumeraite.core.engine import GenerationEngine
from enumeraite.core.http_validator import PathHTTPResult, PathHTTPValidator
from enumeraite.core.models import (
    GenerationResult,
    PathFunctionAnalysis,
    PatternAnalysisResult,
    ValidationResult,
)
from enumeraite.core.strategies import SubdomainDiscoveryStrategy


def test_generation_result_clamps_non_finite_and_out_of_range_confidence():
    result = GenerationResult(
        paths=["/nan", "/inf", "/low", "/high"],
        confidence_scores=[float("nan"), float("inf"), -1.0, 2.0],
    )
    assert result.confidence_scores == [0.0, 0.0, 0.0, 1.0]
    assert GenerationResult(paths=["/bool"], confidence_scores=[True]).confidence_scores == [0.0]


def test_public_result_models_keep_confidence_json_safe():
    pattern = PatternAnalysisResult(
        original_subdomain="api.example.test",
        decomposition=[],
        reasoning="",
        pattern_template="",
        generated_variants=[],
        confidence_score=float("inf"),
    )
    function = PathFunctionAnalysis(
        original_path="/api",
        function_context="lookup",
        path_breakdown=[],
        function_analysis="",
        reasoning="",
        generated_paths=[],
        confidence_score=True,
    )
    assert pattern.confidence_score == 0.0
    assert function.confidence_score == 0.0


def test_subdomain_engine_rejects_invalid_scope_before_provider_call():
    provider = Mock()
    engine = GenerationEngine(provider)
    with pytest.raises(ValueError, match="valid DNS apex"):
        engine.generate_paths(
            ["api.example.com"],
            "https://example.com",
            10,
            strategy=SubdomainDiscoveryStrategy(),
        )
    provider.generate_paths.assert_not_called()


def test_engine_enforces_max_depth_for_third_party_providers():
    provider = Mock()
    provider.generate_paths.return_value = GenerationResult(
        paths=["/allowed", "/too/deep"],
        confidence_scores=[0.7, 0.9],
    )
    result = GenerationEngine(provider).generate_paths(["/seed"], "example.com", 2, max_depth=1)
    assert result.paths == ["/allowed"]
    assert result.metadata["engine_depth_rejected"] == 1


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
        metadata={"provider": "test"},
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
        metadata={"provider": "test"},
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
        metadata={"provider": "test"},
    )

    engine = GenerationEngine(mock_provider)
    known_paths = ["/api/user/profile"]
    result = engine.generate_paths(known_paths, "example.com", 10)

    # Should filter out the known path "/api/user/profile"
    assert len(result.paths) == 2
    assert "/api/user/profile" not in result.paths
    assert "/api/user/settings" in result.paths
    assert "/api/user/preferences" in result.paths


def test_engine_rejects_unsafe_and_cross_scope_third_party_output():
    mock_provider = Mock()
    mock_provider.generate_paths.return_value = GenerationResult(
        paths=[
            "API.Example.COM.",
            "outside.test",
            "bad..example.com",
            "new.example.com",
        ],
        confidence_scores=[0.9, 0.9, 0.9, 0.8],
        metadata={"provider": "third-party"},
    )
    result = GenerationEngine(mock_provider).generate_paths(
        ["seed.example.com"],
        "example.com",
        4,
        strategy=SubdomainDiscoveryStrategy(),
    )
    assert result.paths == ["api.example.com", "new.example.com"]
    assert result.metadata["engine_invalid_rejected"] == 1
    assert result.metadata["engine_scope_rejected"] == 1


def test_subdomain_engine_filters_cross_apex_seeds_before_provider_call():
    provider = Mock()
    provider.generate_paths.return_value = GenerationResult(
        paths=["new.example.com"], confidence_scores=[0.8]
    )
    result = GenerationEngine(provider).generate_paths(
        ["API.Example.COM.", "outside.test", "not a domain"],
        "example.com",
        1,
        strategy=SubdomainDiscoveryStrategy(),
    )
    provider.generate_paths.assert_called_once_with(
        ["api.example.com"], "example.com", 1, None, ANY
    )
    assert result.metadata["engine_seed_scope_rejected"] == 2


def test_engine_rejects_invalid_budgets():
    with pytest.raises(ValueError, match="count"):
        GenerationEngine(Mock()).generate_paths(["/seed"], "example.test", 0)
    with pytest.raises(ValueError, match="max_depth"):
        GenerationEngine(Mock()).generate_paths(["/seed"], "example.test", 1, max_depth=0)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("count", True, "count"),
        ("count", 2.5, "count"),
        ("max_depth", False, "max_depth"),
        ("max_depth", 1.5, "max_depth"),
        ("known_paths", "/seed", "known_paths"),
        ("target", None, "target"),
    ],
)
def test_engine_rejects_ambiguous_direct_api_inputs(argument, value, message):
    arguments = {
        "known_paths": ["/seed"],
        "target": "example.test",
        "count": 2,
        "max_depth": 2,
    }
    arguments[argument] = value
    with pytest.raises(ValueError, match=message):
        GenerationEngine(Mock()).generate_paths(**arguments)


@pytest.mark.asyncio
async def test_generate_and_validate_paths():
    """Test combined generation and validation."""
    mock_provider = Mock()
    mock_provider.generate_paths.return_value = GenerationResult(
        paths=["/api/user/settings", "/api/user/preferences"],
        confidence_scores=[0.8, 0.7],
        metadata={"provider": "test"},
    )

    mock_validator = AsyncMock()
    mock_validator.validate_paths = AsyncMock(
        return_value=[
            ValidationResult(
                path="https://example.com/api/user/settings",
                status_code=200,
                exists=True,
                method="GET",
            )
        ]
    )

    engine = GenerationEngine(mock_provider, mock_validator)

    generation_result, validation_results = await engine.generate_and_validate_paths(
        ["/api/user/profile"], "example.com", 10, ["GET"]
    )

    assert len(generation_result.paths) == 2
    assert len(validation_results) == 1
    assert validation_results[0].exists == True
    mock_validator.validate_paths.assert_awaited_once_with(
        "https://example.com",
        ["/api/user/settings", "/api/user/preferences"],
        ["GET"],
    )


@pytest.mark.asyncio
async def test_generate_and_validate_paths_uses_builtin_validator_without_blocking():
    mock_provider = Mock()
    mock_provider.generate_paths.return_value = GenerationResult(
        paths=["/api/status"],
        confidence_scores=[0.8],
        metadata={"provider": "test"},
    )
    validator = PathHTTPValidator()
    validator.validate_paths = Mock(
        return_value=[
            PathHTTPResult(
                path="/api/status",
                url="http://example.test/v1/api/status",
                status_code=200,
                exists=True,
                response_time=0.01,
            )
        ]
    )

    generated, validation = await GenerationEngine(
        mock_provider, validator
    ).generate_and_validate_paths([], "http://example.test/v1/", 1)

    assert generated.paths == ["/api/status"]
    assert validation == [
        ValidationResult(
            path="/api/status",
            status_code=200,
            exists=True,
            method="GET",
            response_time=0.01,
        )
    ]
    validator.validate_paths.assert_called_once_with("http://example.test/v1", ["/api/status"])


@pytest.mark.asyncio
async def test_generate_and_validate_paths_rejects_unsupported_builtin_method():
    provider = Mock()
    provider.generate_paths.return_value = GenerationResult(
        paths=["/status"], confidence_scores=[0.8], metadata={}
    )
    with pytest.raises(ValueError, match="GET method only"):
        await GenerationEngine(provider, PathHTTPValidator()).generate_and_validate_paths(
            [], "example.test", 1, methods=["POST"]
        )


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

    assert engine._normalize_path("/API/USER/") == "/API/USER"
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
