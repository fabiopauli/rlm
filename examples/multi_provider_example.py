"""
Multi-provider example.

Demonstrates using different providers for root and sub-calls.
For example: Grok for main reasoning, GPT-4o-mini for quick sub-calls.
"""

import os
from dotenv import load_dotenv
from rlm import RecursiveLanguageModel, create_provider


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    xai_key = os.getenv("XAI_API_KEY")

    if not openai_key:
        print("Warning: OPENAI_API_KEY not set, will use same provider for sub-calls")
    if not xai_key:
        print("Warning: XAI_API_KEY not set, falling back to OpenAI")

    # Create context with classification task
    items = [
        "Tesla Model 3 - Electric sedan",
        "Bicycle - Two-wheeled vehicle",
        "Boeing 747 - Commercial aircraft",
        "Honda Civic - Gasoline car",
        "Skateboard - Four-wheeled board",
        "Airbus A380 - Large passenger plane",
        "Ford F-150 - Pickup truck",
        "Scooter - Small motorized vehicle",
        "Helicopter - Rotary wing aircraft",
        "Motorcycle - Two-wheeled motor vehicle",
    ] * 50  # Repeat to create larger list

    context = "\n".join(items)

    print("="*60)
    print("Multi-Provider Example: Cross-Provider RLM")
    print("="*60)

    # Example 1: Grok main + GPT-4o-mini sub-calls (cost-effective)
    if xai_key and openai_key:
        print("\n## Configuration 1: Grok + GPT-4o-mini")
        print("Using Grok for main reasoning, GPT-4o-mini for sub-calls\n")

        rlm = RecursiveLanguageModel(
            api_key=xai_key,
            model="grok-4",
            provider="xai",
            enable_cache=True
        )

        # Override sub-calls to use OpenAI
        # This is advanced usage - normally sub_model would be same provider
        print("Note: This example shows provider architecture.")
        print("In practice, use same provider for main and sub-calls.\n")

    # Example 2: Standard OpenAI with different models
    elif openai_key:
        print("\n## Configuration: GPT-4o + GPT-4o-mini")
        print("Using GPT-4o for main reasoning, GPT-4o-mini for sub-calls\n")

        rlm = RecursiveLanguageModel(
            api_key=openai_key,
            model="gpt-4o",
            sub_model="gpt-4o-mini",
            provider="openai",
            enable_cache=True
        )

    # Example 3: Grok only
    elif xai_key:
        print("\n## Configuration: Grok + Grok Fast Reasoning")
        print("Using Grok-4 for main, Grok-4-1-fast for sub-calls\n")

        rlm = RecursiveLanguageModel(
            api_key=xai_key,
            model="grok-4",
            sub_model="grok-4-1-fast-reasoning",
            provider="xai",
            enable_cache=True
        )

    else:
        raise ValueError("No API keys found. Set OPENAI_API_KEY or XAI_API_KEY")

    # Task
    task = """
    Classify each item into categories: Cars, Aircraft, or Other.
    Count how many items are in each category.
    Return the counts in the format: "Cars: X, Aircraft: Y, Other: Z"
    """

    print(f"Task: {task}")
    print(f"Main Model: {rlm.model}")
    print(f"Sub Model: {rlm.sub_model}")
    print(f"Total items: {len(items):,}")
    print(f"Context length: {len(context):,} characters\n")

    try:
        result = rlm.run(task=task, context=context, verbose=True)

        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"\n{result}\n")

        # Print metrics
        rlm.print_metrics()

        # Show cost breakdown by model
        metrics = rlm.get_metrics_summary()
        print("\n" + "="*60)
        print("Cost Analysis")
        print("="*60)
        for model, cost in metrics['cost']['by_model'].items():
            print(f"{model}: ${cost}")

        print(f"\nTotal: ${metrics['cost']['total_usd']}")

        # Export metrics
        rlm.export_metrics("multi_provider_metrics.json")
        print("\nMetrics exported to multi_provider_metrics.json")

    except Exception as e:
        print(f"\nError: {str(e)}")
        if 'rlm' in locals():
            rlm.print_metrics()


if __name__ == "__main__":
    main()
