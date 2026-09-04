from unittest.mock import Mock, patch

from enumeraite.providers.openai_compatible import OpenAICompatibleProvider


@patch("enumeraite.providers.openai_compatible.openai.OpenAI")
def test_minimax_uses_documented_openai_compatibility_parameters(mock_openai):
    client = Mock()
    response = Mock()
    choice = Mock()
    choice.message.content = '{"candidates":[]}'
    choice.finish_reason = "stop"
    choice.logprobs = None
    response.choices = [choice]
    response.model = "MiniMax-M3"
    response.usage.prompt_tokens = 10
    response.usage.prompt_tokens_details.cached_tokens = 4
    response.usage.completion_tokens = 20
    response.usage.completion_tokens_details.reasoning_tokens = 12
    client.chat.completions.create.return_value = response
    mock_openai.return_value = client

    provider = OpenAICompatibleProvider(
        {"preset": "minimax", "api_key": "test-secret", "confidence": "logprob"}
    )
    result = provider.chat("system", "user", schema={"type": "object"})

    args = client.chat.completions.create.call_args.kwargs
    assert args["model"] == "MiniMax-M3"
    assert args["max_completion_tokens"] == 8192
    assert args["extra_body"] == {"reasoning_split": True}
    assert "max_tokens" not in args
    assert "response_format" not in args
    assert "logprobs" not in args
    assert result.reasoning_tokens == 12
    assert result.cached_input_tokens == 4
    assert result.finish_reason == "stop"
    assert provider.batch_size == 20


@patch("enumeraite.providers.openai_compatible.openai.OpenAI")
def test_minimax_does_not_retry_identical_request_on_client_error(mock_openai):
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("bad request")
    mock_openai.return_value = client
    provider = OpenAICompatibleProvider({"preset": "minimax", "api_key": "test"})

    try:
        provider.chat("system", "user", schema={"type": "object"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected provider error")
    assert client.chat.completions.create.call_count == 1
