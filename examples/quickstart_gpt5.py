"""
Quick Start with GPT-5 mini - Test new model

Run this with:
    export OPENAI_API_KEY="your-key"
    python quick_start_gpt5.py
"""

import os
from dotenv import load_dotenv
from rlm import RecursiveLanguageModel


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        return

    # Simple context with magic number
    context = """
    Document 1: The weather today is sunny.
    Document 2: Python is a programming language.
    Document 3: The magic number is 42.
    Document 4: Coffee is delicious.
    """

    # Create RLM with GPT-5 mini
    print("Testing with GPT-5 mini...")
    rlm = RecursiveLanguageModel(
        api_key=api_key,
        model="gpt-5-mini",
        provider="openai"
    )

    # Run task
    try:
        result = rlm.run(
            task="What is the magic number?",
            context=context,
            verbose=True,
            max_iterations=10
        )

        # Print result
        print("\n" + "="*60)
        print("RESULT:", result)
        print("="*60)

        # Show metrics
        print("\nMetrics:")
        print(f"  Model: {rlm.model}")
        print(f"  Cost: ${rlm.metrics.total_cost:.4f}")
        print(f"  Tokens: {rlm.metrics.total_tokens:,}")
        print(f"  Sub-calls: {rlm.metrics.sub_calls}")
        print(f"  Iterations: {rlm.metrics.iterations}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        rlm.print_metrics()


if __name__ == "__main__":
    main()
