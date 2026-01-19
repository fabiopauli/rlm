"""
Basic usage example with xAI Grok.

Demonstrates using Grok models for recursive language processing.
"""

import os
from dotenv import load_dotenv
from rlm import RecursiveLanguageModel


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("Please set XAI_API_KEY environment variable")

    # Create a long context with a hidden "needle"
    context = """
    This is a very long document with lots of information.
    """ + "\n".join([f"This is line {i} with some random content about topic {i % 10}."
                     for i in range(1000)]) + """

    Here is the important information: The secret code is GROK2026.

    """ + "\n".join([f"More content on line {i} discussing various subjects."
                     for i in range(1000, 2000)])

    # Initialize RLM with Grok
    rlm = RecursiveLanguageModel(
        api_key=api_key,
        model="grok-4",
        provider="xai",  # Explicitly specify xAI provider
        sub_model="grok-4",  # Could use grok-4-1-fast-reasoning for faster sub-calls
        enable_cache=True,
        max_cost=2.0  # Budget limit: $2
    )

    # Run task
    task = "Find the secret code mentioned in the context."

    print("="*60)
    print("Grok Example: Needle in Haystack")
    print("="*60)
    print(f"\nTask: {task}")
    print(f"Model: {rlm.model}")
    print(f"Context length: {len(context):,} characters")
    print(f"Context window: {rlm.context_window:,} tokens\n")

    try:
        result = rlm.run(task=task, context=context, verbose=True)

        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"\n{result}\n")

        # Print metrics
        rlm.print_metrics()

        # Export metrics to file
        rlm.export_metrics("grok_basic_metrics.json")
        print("\nMetrics exported to grok_basic_metrics.json")

    except Exception as e:
        print(f"\nError: {str(e)}")
        rlm.print_metrics()


if __name__ == "__main__":
    main()
