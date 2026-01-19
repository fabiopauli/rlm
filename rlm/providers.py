"""
LLM Provider abstraction for multi-provider support.

Supports OpenAI, xAI Grok, and other providers with unified interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenUsageDetails:
    """Detailed token usage information."""

    # Main counts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Prompt details
    prompt_text_tokens: int = 0
    prompt_audio_tokens: int = 0
    prompt_image_tokens: int = 0
    prompt_cached_tokens: int = 0

    # Completion details
    completion_reasoning_tokens: int = 0
    completion_audio_tokens: int = 0
    completion_accepted_prediction_tokens: int = 0
    completion_rejected_prediction_tokens: int = 0

    # Additional metadata
    num_sources_used: int = 0


@dataclass
class LLMResponse:
    """Unified LLM response format."""

    content: str
    usage: TokenUsageDetails
    model: str
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Provides unified interface for different LLM APIs.
    """

    def __init__(
        self,
        api_key: str,
        default_model: str,
        context_window: int = 128_000,
        timeout: int = 600
    ):
        """
        Initialize provider.

        Args:
            api_key: API key for the provider
            default_model: Default model to use
            context_window: Maximum context window size (default 128k)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.default_model = default_model
        self.context_window = context_window
        self.timeout = timeout

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> LLMResponse:
        """
        Create a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to self.default_model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    def get_context_window(self, model: str) -> int:
        """Get context window size for a model."""
        pass

    @abstractmethod
    def get_pricing(self, model: str) -> Dict[str, float]:
        """
        Get pricing for a model.

        Returns:
            Dict with 'prompt' and 'completion' prices per 1K tokens
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    # Context windows for OpenAI models
    CONTEXT_WINDOWS = {
        "gpt-5-mini": 128_000,
        "gpt-5-nano": 128_000,
        "gpt-4.1": 128_000,
        "gpt-4.1-mini": 128_000,
        "gpt-4.1-nano": 128_000,
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "gpt-4-turbo": 128_000,
        "gpt-4": 8_192,
        "gpt-3.5-turbo": 16_385,
    }

    # Pricing (USD per 1K tokens)
    # Source: https://openai.com/api/pricing
    PRICING = {
        "gpt-5-mini": {"prompt": 0.00025, "completion": 0.002},
        "gpt-5-nano": {"prompt": 0.00005, "completion": 0.0004},
        "gpt-4.1": {"prompt": 0.002, "completion": 0.008},
        "gpt-4.1-mini": {"prompt": 0.0004, "completion": 0.0016},
        "gpt-4.1-nano": {"prompt": 0.0001, "completion": 0.0004},
        "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    }

    def __init__(
        self,
        api_key: str,
        default_model: str = "gpt-4o",
        context_window: Optional[int] = None,
        timeout: int = 600
    ):
        """Initialize OpenAI provider."""
        import openai

        # Auto-detect context window if not provided
        if context_window is None:
            context_window = self.CONTEXT_WINDOWS.get(default_model, 128_000)

        super().__init__(api_key, default_model, context_window, timeout)
        self.client = openai.OpenAI(api_key=api_key, timeout=timeout)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> LLMResponse:
        """Create OpenAI chat completion."""
        model = model or self.default_model

        try:
            # GPT-5 models use max_completion_tokens instead of max_tokens
            api_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }

            # Add max tokens parameter with correct name for model
            if max_tokens is not None:
                if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
                    api_kwargs["max_completion_tokens"] = max_tokens
                else:
                    api_kwargs["max_tokens"] = max_tokens

            # Add any other kwargs
            api_kwargs.update(kwargs)

            response = self.client.chat.completions.create(**api_kwargs)

            # Parse usage details
            usage_data = response.usage
            usage = TokenUsageDetails(
                prompt_tokens=usage_data.prompt_tokens,
                completion_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens,
            )

            # Add prompt details if available
            if hasattr(usage_data, 'prompt_tokens_details') and usage_data.prompt_tokens_details:
                details = usage_data.prompt_tokens_details
                usage.prompt_cached_tokens = getattr(details, 'cached_tokens', 0)

            # Add completion details if available
            if hasattr(usage_data, 'completion_tokens_details') and usage_data.completion_tokens_details:
                details = usage_data.completion_tokens_details
                usage.completion_reasoning_tokens = getattr(details, 'reasoning_tokens', 0)

            return LLMResponse(
                content=response.choices[0].message.content.strip(),
                usage=usage,
                model=model,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    def get_context_window(self, model: str) -> int:
        """Get context window for OpenAI model."""
        return self.CONTEXT_WINDOWS.get(model, self.context_window)

    def get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for OpenAI model."""
        return self.PRICING.get(model, self.PRICING["gpt-4o"])


class XAIProvider(LLMProvider):
    """xAI Grok API provider."""

    # Context windows for xAI models
    CONTEXT_WINDOWS = {
        "grok-4": 128_000,
        "grok-4-1-fast-reasoning": 128_000,
        "grok-4-1-fast-non-reasoning": 128_000,
        "grok-4-fast-reasoning": 128_000,
        "grok-4-fast-non-reasoning": 128_000,
        "grok-beta": 128_000,
    }

    # Pricing for xAI models (USD per 1K tokens)
    # Source: https://x.ai/api pricing page
    PRICING = {
        "grok-4": {"prompt": 0.002, "completion": 0.010},
        "grok-4-1-fast-reasoning": {"prompt": 0.0002, "completion": 0.0005},
        "grok-4-1-fast-non-reasoning": {"prompt": 0.0002, "completion": 0.0005},
        "grok-4-fast-reasoning": {"prompt": 0.002, "completion": 0.010},
        "grok-4-fast-non-reasoning": {"prompt": 0.002, "completion": 0.010},
        "grok-beta": {"prompt": 0.005, "completion": 0.015},
    }

    def __init__(
        self,
        api_key: str,
        default_model: str = "grok-4",
        context_window: Optional[int] = None,
        timeout: int = 3600  # Longer timeout for reasoning models
    ):
        """Initialize xAI provider."""
        try:
            from xai_sdk import Client
            from xai_sdk.chat import user, system, assistant
        except ImportError:
            raise ImportError(
                "xAI SDK not installed. Install with: pip install xai-sdk"
            )

        # Auto-detect context window if not provided
        if context_window is None:
            context_window = self.CONTEXT_WINDOWS.get(default_model, 128_000)

        super().__init__(api_key, default_model, context_window, timeout)
        self.client = Client(api_key=api_key, timeout=timeout)

        # Store message builder functions
        self._user = user
        self._system = system
        self._assistant = assistant

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> LLMResponse:
        """Create xAI chat completion."""
        model = model or self.default_model

        try:
            # Create chat session
            chat = self.client.chat.create(model=model)

            # Convert messages to xAI format
            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if role == "system":
                    chat.append(self._system(content))
                elif role == "user":
                    chat.append(self._user(content))
                elif role == "assistant":
                    chat.append(self._assistant(content))

            # Sample response
            # Note: xAI SDK's sample() doesn't support max_tokens parameter
            sample_kwargs = {}
            if temperature != 1.0:
                sample_kwargs['temperature'] = temperature
            # Add any other kwargs except max_tokens
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'max_tokens'}
            sample_kwargs.update(filtered_kwargs)

            response = chat.sample(**sample_kwargs)

            # Parse usage details
            usage_data = response.usage
            usage = TokenUsageDetails(
                prompt_tokens=getattr(usage_data, 'prompt_tokens', 0),
                completion_tokens=getattr(usage_data, 'completion_tokens', 0),
                total_tokens=getattr(usage_data, 'total_tokens', 0),
            )

            # Parse prompt details
            if hasattr(usage_data, 'prompt_tokens_details') and usage_data.prompt_tokens_details:
                details = usage_data.prompt_tokens_details
                usage.prompt_text_tokens = getattr(details, 'text_tokens', 0)
                usage.prompt_audio_tokens = getattr(details, 'audio_tokens', 0)
                usage.prompt_image_tokens = getattr(details, 'image_tokens', 0)
                usage.prompt_cached_tokens = getattr(details, 'cached_tokens', 0)

            # Parse completion details
            if hasattr(usage_data, 'completion_tokens_details') and usage_data.completion_tokens_details:
                details = usage_data.completion_tokens_details
                usage.completion_reasoning_tokens = getattr(details, 'reasoning_tokens', 0)
                usage.completion_audio_tokens = getattr(details, 'audio_tokens', 0)
                usage.completion_accepted_prediction_tokens = getattr(details, 'accepted_prediction_tokens', 0)
                usage.completion_rejected_prediction_tokens = getattr(details, 'rejected_prediction_tokens', 0)

            # Additional metadata
            usage.num_sources_used = getattr(usage_data, 'num_sources_used', 0)

            return LLMResponse(
                content=response.content,
                usage=usage,
                model=model,
                finish_reason=getattr(response, 'finish_reason', None),
                raw_response=response
            )

        except Exception as e:
            logger.error(f"xAI API error: {str(e)}")
            raise

    def get_context_window(self, model: str) -> int:
        """Get context window for xAI model."""
        return self.CONTEXT_WINDOWS.get(model, self.context_window)

    def get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for xAI model."""
        return self.PRICING.get(model, self.PRICING["grok-4"])


def create_provider(
    provider_name: str,
    api_key: str,
    default_model: Optional[str] = None,
    context_window: Optional[int] = None,
    timeout: Optional[int] = None
) -> LLMProvider:
    """
    Factory function to create LLM provider.

    Args:
        provider_name: Provider name ('openai' or 'xai')
        api_key: API key
        default_model: Default model to use
        context_window: Context window size
        timeout: Request timeout

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_name = provider_name.lower()

    kwargs = {"api_key": api_key}
    if default_model:
        kwargs["default_model"] = default_model
    if context_window:
        kwargs["context_window"] = context_window
    if timeout:
        kwargs["timeout"] = timeout

    if provider_name == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_name in ["xai", "grok"]:
        return XAIProvider(**kwargs)
    else:
        raise ValueError(
            f"Unsupported provider: {provider_name}. "
            f"Supported providers: 'openai', 'xai'"
        )
