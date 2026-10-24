"""Unit tests for Groq LLM module."""
import pytest
from unittest.mock import MagicMock, patch
from groq import RateLimitError, APIStatusError
import httpx

from app.llm import generate, get_groq_client
import app.llm as llm_module


def test_missing_api_key_raises_value_error(monkeypatch):
    """Test that missing GROQ_API_KEY raises a clear ValueError."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr(llm_module, "GROQ_API_KEY", "")
    llm_module._client = None
    with pytest.raises(ValueError, match="GROQ_API_KEY not set"):
        get_groq_client(api_key=None)


def test_generate_success(monkeypatch):
    """Test standard successful generation using default main model."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Transformers rely entirely on attention [Introduction, p. 1]."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    monkeypatch.setattr(llm_module, "get_groq_client", lambda api_key=None: mock_client)

    result = generate(
        prompt="Explain attention mechanism",
        system="Answer strictly from context",
    )

    assert result == "Transformers rely entirely on attention [Introduction, p. 1]."
    mock_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["messages"][0] == {"role": "system", "content": "Answer strictly from context"}
    assert kwargs["messages"][1] == {"role": "user", "content": "Explain attention mechanism"}
    assert kwargs["model"] == llm_module.GROQ_MODEL_MAIN


def test_generate_retry_exponential_backoff(monkeypatch):
    """Test retry with exponential backoff on 429 RateLimitError."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Success after rate limit."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # Create dummy RateLimitError
    dummy_request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    dummy_response = httpx.Response(429, request=dummy_request)
    rate_limit_err = RateLimitError("Rate limit exceeded", response=dummy_response, body=None)

    # Fail twice with 429, succeed on third attempt
    mock_client.chat.completions.create.side_effect = [
        rate_limit_err,
        rate_limit_err,
        mock_response,
    ]

    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(llm_module, "get_groq_client", lambda api_key=None: mock_client)

    result = generate(
        prompt="Test retry prompt",
        initial_backoff=0.5,
        backoff_factor=2.0,
        max_retries=3,
    )

    assert result == "Success after rate limit."
    assert mock_client.chat.completions.create.call_count == 3
    # Check exponential backoff delays: 0.5s then 1.0s
    assert sleep_calls == [0.5, 1.0]


def test_generate_strips_reasoning_content(monkeypatch):
    """Test that generate() strips reasoning/thinking content and returns only final answer."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        "<think>\nLet's analyze the question and look at excerpt 1.\nThe answer is in section 6.\n</think>\n"
        "The model achieves a BLEU score of 28.4 [6 Results, pp. 8-9]."
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    monkeypatch.setattr(llm_module, "get_groq_client", lambda api_key=None: mock_client)

    result = generate(prompt="What was the BLEU score?")
    assert "<think>" not in result
    assert "</think>" not in result
    assert "Let's analyze" not in result
    assert result == "The model achieves a BLEU score of 28.4 [6 Results, pp. 8-9]."
