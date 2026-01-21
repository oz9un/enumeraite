"""Tests for Claude provider implementation."""
import pytest
from unittest.mock import Mock, patch
from enumeraite.providers.claude_provider import ClaudeProvider

@patch('anthropic.Anthropic')
def test_claude_provider_generation(mock_anthropic):
    """Test Claude provider path generation with mocked API response."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = """
    /api/user/settings
    /api/user/preferences
    /api/user/notifications
    """
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client

    provider = ClaudeProvider({"api_key": "test-key"})
    result = provider.generate_paths(["/api/user/profile"], "example.com", 3)

    assert len(result.paths) == 3
    assert "/api/user/settings" in result.paths
    assert all(0.0 <= score <= 1.0 for score in result.confidence_scores)
    assert result.metadata["provider"] == "claude"
    assert result.metadata["target"] == "example.com"

def test_claude_provider_name():
    """Test that provider returns correct name."""
    provider = ClaudeProvider({"api_key": "test-key"})
    assert provider.get_provider_name() == "claude"