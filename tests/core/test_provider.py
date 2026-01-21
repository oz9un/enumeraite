"""Tests for base provider interface."""
import pytest
from enumeraite.core.provider import BaseProvider
from enumeraite.core.models import GenerationResult

def test_base_provider_interface():
    """Test that BaseProvider is abstract and raises NotImplementedError."""
    # Create a concrete implementation that doesn't override abstract methods
    class TestProvider(BaseProvider):
        pass

    # Should not be able to instantiate without implementing abstract methods
    with pytest.raises(TypeError):
        TestProvider({})

def test_generation_result_model():
    """Test that GenerationResult model works correctly."""
    result = GenerationResult(
        paths=["/api/user/settings", "/api/user/profile"],
        confidence_scores=[0.8, 0.9],
        metadata={"provider": "test", "method": "pattern"}
    )
    assert len(result.paths) == 2
    assert len(result.confidence_scores) == 2
    assert result.metadata["provider"] == "test"