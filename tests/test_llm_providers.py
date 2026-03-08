"""Tests for new LLM providers: OpenAI, Anthropic, OpenRouter, LiteLLM, HuggingFace, LlamaCpp."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_index.llm import (
    PROVIDER_NAMES,
    AnthropicProvider,
    HuggingFaceProvider,
    LiteLLMProvider,
    LlamaCppProvider,
    OpenAIProvider,
    OpenRouterProvider,
    get_provider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_openai_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_anthropic_response(content: str, input_tokens: int = 10, output_tokens: int = 5):
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    content_block = MagicMock()
    content_block.text = content
    response = MagicMock()
    response.content = [content_block]
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# TestOpenAIProvider
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    def test_init_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIProvider(api_key=None)

    def test_init_reads_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        p = OpenAIProvider()
        assert p._api_key == "sk-test"

    @pytest.mark.asyncio
    async def test_generate_messages_format(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_response = _make_openai_response("hello")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        with patch.dict(sys.modules, {"openai": mock_openai}):
            p = OpenAIProvider(api_key="sk-test")
            result = await p.generate("hi", system="be helpful")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "be helpful"}
        assert messages[1] == {"role": "user", "content": "hi"}
        assert result.text == "hello"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    @pytest.mark.asyncio
    async def test_generate_no_system(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_response = _make_openai_response("world")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        with patch.dict(sys.modules, {"openai": mock_openai}):
            p = OpenAIProvider(api_key="sk-test")
            result = await p.generate("hello")

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_repr(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert "OpenAIProvider" in repr(OpenAIProvider())


# ---------------------------------------------------------------------------
# TestAnthropicProvider
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    def test_init_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicProvider(api_key=None)

    @pytest.mark.asyncio
    async def test_generate_system_as_kwarg(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        mock_response = _make_anthropic_response("ok")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            p = AnthropicProvider(api_key="sk-ant")
            result = await p.generate("hello", system="sys prompt")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        # system must be a top-level kwarg, NOT inside messages
        assert call_kwargs["system"] == "sys prompt"
        messages = call_kwargs["messages"]
        for msg in messages:
            assert msg["role"] != "system"
        assert result.text == "ok"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    @pytest.mark.asyncio
    async def test_generate_max_tokens_required(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        mock_response = _make_anthropic_response("ok")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            p = AnthropicProvider(api_key="sk-ant")
            await p.generate("hello", max_tokens=128)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 128

    def test_repr(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        assert "AnthropicProvider" in repr(AnthropicProvider())


# ---------------------------------------------------------------------------
# TestOpenRouterProvider
# ---------------------------------------------------------------------------

class TestOpenRouterProvider:
    def test_init_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterProvider(api_key=None)

    @pytest.mark.asyncio
    async def test_base_url_and_headers(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        mock_response = _make_openai_response("resp")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        with patch.dict(sys.modules, {"openai": mock_openai}):
            p = OpenRouterProvider(api_key="sk-or")
            await p.generate("hi")

        init_kwargs = mock_openai.AsyncOpenAI.call_args.kwargs
        assert init_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        headers = init_kwargs["default_headers"]
        assert "HTTP-Referer" in headers
        assert "X-Title" in headers

    def test_repr(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        assert "OpenRouterProvider" in repr(OpenRouterProvider())


# ---------------------------------------------------------------------------
# TestLiteLLMProvider
# ---------------------------------------------------------------------------

class TestLiteLLMProvider:
    @pytest.mark.asyncio
    async def test_generate_call_args(self):
        mock_response = _make_openai_response("out")
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            p = LiteLLMProvider(model="anthropic/claude-haiku-4-5-20251001")
            result = await p.generate("hello", system="sys", max_tokens=64, temperature=0.5)

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["model"] == "anthropic/claude-haiku-4-5-20251001"
        assert call_kwargs["max_tokens"] == 64
        assert call_kwargs["temperature"] == 0.5
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "sys"}
        assert messages[1] == {"role": "user", "content": "hello"}
        assert result.text == "out"

    def test_repr(self):
        assert "LiteLLMProvider" in repr(LiteLLMProvider())


# ---------------------------------------------------------------------------
# TestHuggingFaceProvider
# ---------------------------------------------------------------------------

class TestHuggingFaceProvider:
    def test_init_requires_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="HF_TOKEN"):
            HuggingFaceProvider(api_key=None)

    def test_init_fallback_to_huggingface_api_key(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf-key")
        p = HuggingFaceProvider()
        assert p._api_key == "hf-key"

    @pytest.mark.asyncio
    async def test_token_counts_fallback_to_zero(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf-test")

        mock_choice = MagicMock()
        mock_choice.message.content = "answer"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None  # no usage info

        mock_client = MagicMock()
        mock_client.chat_completion = AsyncMock(return_value=mock_response)
        mock_hf = MagicMock()
        mock_hf.AsyncInferenceClient.return_value = mock_client

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            p = HuggingFaceProvider(api_key="hf-test")
            result = await p.generate("hello")

        assert result.text == "answer"
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_repr(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf-test")
        assert "HuggingFaceProvider" in repr(HuggingFaceProvider())


# ---------------------------------------------------------------------------
# TestLlamaCppProvider
# ---------------------------------------------------------------------------

class TestLlamaCppProvider:
    @pytest.mark.asyncio
    async def test_endpoint_and_json_body(self, respx_mock=None):
        import httpx
        from unittest.mock import patch, AsyncMock

        response_data = {
            "choices": [{"message": {"content": "llama says hi"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = response_data

        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_instance.post = AsyncMock(return_value=mock_response)

        with patch("agentic_index.llm.httpx.AsyncClient", return_value=mock_client_instance):
            p = LlamaCppProvider(model="llama3", base_url="http://localhost:8080")
            result = await p.generate("hi", system="sys")

        call_args = mock_client_instance.post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
        assert url.endswith("/v1/chat/completions")
        body = call_args.kwargs["json"]
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        assert body["messages"][1] == {"role": "user", "content": "hi"}
        assert body["model"] == "llama3"
        assert result.text == "llama says hi"
        assert result.input_tokens == 7
        assert result.output_tokens == 3

    @pytest.mark.asyncio
    async def test_no_model_omits_model_field(self):
        response_data = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = response_data

        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_instance.post = AsyncMock(return_value=mock_response)

        with patch("agentic_index.llm.httpx.AsyncClient", return_value=mock_client_instance):
            p = LlamaCppProvider()  # no model
            await p.generate("hi")

        body = mock_client_instance.post.call_args.kwargs["json"]
        assert "model" not in body

    def test_repr(self):
        assert "LlamaCppProvider" in repr(LlamaCppProvider())


# ---------------------------------------------------------------------------
# TestGetProviderNames
# ---------------------------------------------------------------------------

class TestGetProviderNames:
    def test_provider_names_count(self):
        assert len(PROVIDER_NAMES) == 9

    def test_all_names_present(self):
        expected = {"auto", "ollama", "gemini", "openai", "anthropic",
                    "openrouter", "litellm", "huggingface", "llamacpp"}
        assert set(PROVIDER_NAMES) == expected

    def test_get_provider_ollama(self):
        p = get_provider("ollama")
        assert type(p).__name__ == "OllamaProvider"

    def test_get_provider_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        p = get_provider("openai")
        assert type(p).__name__ == "OpenAIProvider"

    def test_get_provider_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        p = get_provider("anthropic")
        assert type(p).__name__ == "AnthropicProvider"

    def test_get_provider_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        p = get_provider("openrouter")
        assert type(p).__name__ == "OpenRouterProvider"

    def test_get_provider_litellm(self):
        p = get_provider("litellm")
        assert type(p).__name__ == "LiteLLMProvider"

    def test_get_provider_huggingface(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf-tok")
        p = get_provider("huggingface")
        assert type(p).__name__ == "HuggingFaceProvider"

    def test_get_provider_llamacpp(self):
        p = get_provider("llamacpp")
        assert type(p).__name__ == "LlamaCppProvider"

    def test_get_provider_llamacpp_env_base_url(self, monkeypatch):
        monkeypatch.setenv("LLAMACPP_BASE_URL", "http://myhost:9090")
        p = get_provider("llamacpp")
        assert p.base_url == "http://myhost:9090"

    def test_get_provider_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown_provider")


# ---------------------------------------------------------------------------
# TestMissingPackageError
# ---------------------------------------------------------------------------

class TestMissingPackageError:
    @pytest.mark.asyncio
    async def test_openai_missing_package(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        p = OpenAIProvider(api_key="sk-test")
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(RuntimeError, match="openai package not installed"):
                await p.generate("hi")

    @pytest.mark.asyncio
    async def test_anthropic_missing_package(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        p = AnthropicProvider(api_key="sk-ant")
        with patch.dict(sys.modules, {"anthropic": None}):
            with pytest.raises(RuntimeError, match="anthropic package not installed"):
                await p.generate("hi")

    @pytest.mark.asyncio
    async def test_openrouter_missing_package(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        p = OpenRouterProvider(api_key="sk-or")
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(RuntimeError, match="openai package not installed"):
                await p.generate("hi")

    @pytest.mark.asyncio
    async def test_litellm_missing_package(self):
        p = LiteLLMProvider()
        with patch.dict(sys.modules, {"litellm": None}):
            with pytest.raises(RuntimeError, match="litellm package not installed"):
                await p.generate("hi")

    @pytest.mark.asyncio
    async def test_huggingface_missing_package(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf-test")
        p = HuggingFaceProvider(api_key="hf-test")
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            with pytest.raises(RuntimeError, match="huggingface_hub package not installed"):
                await p.generate("hi")
