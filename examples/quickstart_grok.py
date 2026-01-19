"""
Quick Start Example with xAI Grok - Simplest possible usage

Run this after setting your XAI_API_KEY environment variable:
    export XAI_API_KEY="your-key-here"
    python quick_start_grok.py
"""

import os
from dotenv import load_dotenv
from rlm import RecursiveLanguageModel


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get API key
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("ERROR: Please set XAI_API_KEY environment variable")
        print("Example: export XAI_API_KEY='your-key-here'")
        return

    # Simple context
    context = """
    Document 1: The weather today is sunny.
    Document 2: Python is a programming language.
    Document 3: The magic number is 42.
    Document 4: Coffee is delicious.
    Document 5: Grok is an AI assistant by xAI.
    """

    # Create RLM with Grok (minimal configuration)
    rlm = RecursiveLanguageModel(
        api_key=api_key,
        model="grok-4-1-fast-reasoning",
        provider="xai"  # Specify xAI provider
    )

    # Run task
    try:
        result = rlm.run(
            task="What is the magic number?",
            context=context
        )

        # Print result
        print("\n" + "="*60)
        print("RESULT:", result)
        print("="*60)

        # Show metrics
        print("\nMetrics:")
        print(f"  Model: {rlm.model}")
        print(f"  Cost: ${rlm.metrics.total_cost:.4f}")
        print(f"  Tokens: {rlm.metrics.total_tokens}")
        print(f"  Reasoning tokens: {rlm.metrics.total_completion_reasoning_tokens}")
        print(f"  Sub-calls: {rlm.metrics.sub_calls}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        rlm.print_metrics()


if __name__ == "__main__":
    main()
