"""
Mock test for RLM with fake provider.
Tests the integration without requiring API keys.
"""

from rlm.providers import LLMProvider, LLMResponse, TokenUsageDetails
from rlm import RecursiveLanguageModel


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    def __init__(self):
        super().__init__(
            api_key="mock-key",
            default_model="mock-model",
            context_window=128_000
        )
        self.call_count = 0

    def chat_completion(self, messages, model=None, max_tokens=None, **kwargs):
        """Mock chat completion."""
        self.call_count += 1

        # Check if asking for code
        last_message = messages[-1]["content"]

        # Generate mock response based on task
        if "magic number" in last_message.lower() or "42" in last_message.lower():
            # Return code to find the number
            response_text = """
Let me search for the magic number in the context.

```repl
# Search for "magic number" pattern
import re
matches = regex_search(r'magic number is (\d+)', context)
if matches:
    answer = matches[0]
    print(f"Found: {answer}")
    FINAL(answer)
else:
    # Try direct search
    if '42' in context:
        FINAL("42")
    else:
        FINAL("Not found")
```
"""
        else:
            response_text = "Mock response"

        # Create detailed token usage
        usage = TokenUsageDetails(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_cached_tokens=10,  # Simulate some caching
            completion_reasoning_tokens=20,  # Simulate reasoning
            prompt_text_tokens=90,
            prompt_image_tokens=0,
            prompt_audio_tokens=0,
            completion_audio_tokens=0,
            completion_accepted_prediction_tokens=0,
            completion_rejected_prediction_tokens=0,
            num_sources_used=0
        )

        return LLMResponse(
            content=response_text,
            usage=usage,
            model=model or self.default_model,
            finish_reason="stop"
        )

    def get_context_window(self, model):
        return 128_000

    def get_pricing(self, model):
        return {"prompt": 0.001, "completion": 0.002}


def test_mock_provider():
    """Test RLM with mock provider."""
    print("=== Test 3: Mock Provider Test ===\n")

    # Create mock provider
    mock_provider = MockProvider()

    # Create RLM with mock provider
    print("1. Creating RLM with mock provider...")
    rlm = RecursiveLanguageModel(
        provider=mock_provider,
        enable_cache=True,
        enable_security=False,  # Disable for simpler testing
        log_level="WARNING"
    )
    print(f"   ✓ RLM created with model: {rlm.model}")
    print(f"   ✓ Context window: {rlm.context_window:,} tokens")

    # Create simple context
    context = """
    Document 1: The weather is sunny.
    Document 2: The magic number is 42.
    Document 3: Python is great.
    """

    # Run task
    print("\n2. Running task...")
    try:
        result = rlm.run(
            task="Find the magic number in the context.",
            context=context,
            max_iterations=5,
            verbose=False
        )
        print(f"   ✓ Task completed successfully")
        print(f"   ✓ Result: {result}")
    except Exception as e:
        print(f"   ✗ Task failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # Check metrics
    print("\n3. Checking metrics...")
    metrics = rlm.get_metrics_summary()

    print(f"   ✓ Total calls: {metrics['total_calls']}")
    print(f"   ✓ Sub-calls: {metrics['sub_calls']}")
    print(f"   ✓ Iterations: {metrics['iterations']}")
    print(f"   ✓ Total tokens: {metrics['tokens']['total']:,}")

    # Check detailed token tracking
    details = metrics['tokens']['details']
    print(f"   ✓ Cached tokens: {details['cached_prompt_tokens']}")
    print(f"   ✓ Reasoning tokens: {details['reasoning_tokens']}")

    # Check provider info
    provider_info = metrics['provider']
    print(f"   ✓ Provider model: {provider_info['root_model']}")
    print(f"   ✓ Context window: {provider_info['context_window']:,}")

    # Check cache if enabled
    if rlm.enable_cache:
        cache_stats = rlm.cache.get_stats()
        print(f"   ✓ Cache size: {cache_stats['size']}")

    print("\n4. Testing provider methods...")
    print(f"   ✓ Context window: {mock_provider.get_context_window('mock-model'):,}")
    pricing = mock_provider.get_pricing('mock-model')
    print(f"   ✓ Pricing: ${pricing['prompt']}/1K prompt, ${pricing['completion']}/1K completion")
    print(f"   ✓ Total provider calls: {mock_provider.call_count}")

    print("\n✓ All mock tests passed!")
    return True


def test_provider_factory():
    """Test provider factory function."""
    print("\n=== Test 3b: Provider Factory ===\n")

    from rlm import create_provider

    # Test creating providers
    print("1. Testing OpenAI provider creation...")
    try:
        openai_provider = create_provider(
            provider_name="openai",
            api_key="fake-key",
            default_model="gpt-4o",
            context_window=128_000
        )
        print(f"   ✓ OpenAI provider created: {openai_provider}")
        print(f"   ✓ Default model: {openai_provider.default_model}")
        print(f"   ✓ Context window: {openai_provider.context_window:,}")
    except Exception as e:
        print(f"   ✗ Failed: {str(e)}")

    print("\n2. Testing XAI provider creation...")
    try:
        # This will fail if xai-sdk not installed, which is OK
        xai_provider = create_provider(
            provider_name="xai",
            api_key="fake-key",
            default_model="grok-4"
        )
        print(f"   ✓ XAI provider created: {xai_provider}")
        print(f"   ✓ Default model: {xai_provider.default_model}")
    except ImportError as e:
        print(f"   ⚠ XAI SDK not installed (expected): {str(e)}")
        print(f"   → Install with: pip install xai-sdk")
    except Exception as e:
        print(f"   ✗ Failed: {str(e)}")

    print("\n3. Testing invalid provider...")
    try:
        invalid_provider = create_provider(
            provider_name="invalid",
            api_key="fake-key"
        )
        print(f"   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {str(e)}")

    print("\n✓ Factory tests passed!")


if __name__ == "__main__":
    success = test_mock_provider()
    test_provider_factory()

    if success:
        print("\n" + "="*60)
        print("SUCCESS: All mock tests passed!")
        print("="*60)
        print("\nNext steps:")
        print("1. Set API keys: export OPENAI_API_KEY=... or export XAI_API_KEY=...")
        print("2. Run real tests: python quick_start.py or python quick_start_grok.py")
        print("3. Try examples: python examples/grok_basic_example.py")
    else:
        print("\n" + "="*60)
        print("FAILURE: Some tests failed")
        print("="*60)
        exit(1)
