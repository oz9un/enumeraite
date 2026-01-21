"""Tests for OpenAI provider implementation."""
import pytest
from unittest.mock import Mock, patch
from enumeraite.providers.openai_provider import OpenAIProvider

@patch('openai.OpenAI')
def test_openai_provider_generation(mock_openai):
    """Test OpenAI provider path generation with mocked API response."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = """
    Generated paths:
    /api/user/settings
    /api/user/preferences
    /api/user/notifications
    """
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    provider = OpenAIProvider({"api_key": "test-key"})
    result = provider.generate_paths(["/api/user/profile"], "example.com", 3)

    assert len(result.paths) == 3
    assert "/api/user/settings" in result.paths
    assert all(0.0 <= score <= 1.0 for score in result.confidence_scores)
    assert result.metadata["provider"] == "openai"
    assert result.metadata["target"] == "example.com"

def test_openai_provider_name():
    """Test that provider returns correct name."""
    provider = OpenAIProvider({"api_key": "test-key"})
    assert provider.get_provider_name() == "openai"