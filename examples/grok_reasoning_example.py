"""
Grok Fast Reasoning example.

Demonstrates using Grok's fast reasoning model for complex analysis tasks.
Shows detailed reasoning token tracking.
"""

import os
from dotenv import load_dotenv
from rlm import RecursiveLanguageModel


def main():
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("Please set XAI_API_KEY environment variable")

    # Create context with a complex problem requiring multi-step reasoning
    context = """
    # Complex Dataset Analysis

    ## Sales Data for Q1 2024

    January:
    - Product A: 150 units @ $25/unit = $3,750
    - Product B: 200 units @ $15/unit = $3,000
    - Product C: 100 units @ $50/unit = $5,000
    Total January Revenue: $11,750

    February:
    - Product A: 180 units @ $25/unit = $4,500
    - Product B: 220 units @ $15/unit = $3,300
    - Product C: 90 units @ $50/unit = $4,500
    Total February Revenue: $12,300

    March:
    - Product A: 200 units @ $25/unit = $5,000
    - Product B: 250 units @ $15/unit = $3,750
    - Product C: 110 units @ $50/unit = $5,500
    Total March Revenue: $14,250

    ## Costs (Monthly)
    - Fixed Costs: $5,000/month
    - Variable Cost per Product A: $10
    - Variable Cost per Product B: $6
    - Variable Cost per Product C: $20

    ## Additional Information
    - Target Profit Margin: 40%
    - Customer Acquisition Cost: $50 per new customer
    - New Customers in Q1: 45
    - Customer Retention Rate: 85%

    """ * 10  # Repeat to make it longer

    # Initialize RLM with Grok reasoning model
    rlm = RecursiveLanguageModel(
        api_key=api_key,
        model="grok-4",  # Main model
        sub_model="grok-4-1-fast-reasoning",  # Fast reasoning for sub-calls
        provider="xai",
        enable_cache=True,
        context_window=128000
    )

    # Complex task requiring multi-step reasoning
    task = """
    Analyze the sales data and calculate:
    1. Total profit for Q1 2024
    2. Profit margin for each product
    3. Which product is most profitable and why
    4. Whether the company met its 40% target profit margin
    5. ROI on customer acquisition for Q1

    Use multi-step reasoning to break down the problem.
    """

    print("="*60)
    print("Grok Fast Reasoning Example: Financial Analysis")
    print("="*60)
    print(f"\nTask: Multi-step financial analysis")
    print(f"Main Model: {rlm.model}")
    print(f"Sub Model: {rlm.sub_model}")
    print(f"Context window: {rlm.context_window:,} tokens\n")

    try:
        result = rlm.run(task=task, context=context, verbose=True, max_iterations=30)

        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"\n{result}\n")

        # Print detailed metrics
        rlm.print_metrics()

        # Get metrics summary
        metrics = rlm.get_metrics_summary()

        # Show reasoning token usage
        print("\n" + "="*60)
        print("Reasoning Tokens Analysis")
        print("="*60)
        reasoning_tokens = metrics['tokens']['details']['reasoning_tokens']
        total_tokens = metrics['tokens']['total']
        print(f"Reasoning tokens: {reasoning_tokens:,}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Reasoning percentage: {(reasoning_tokens/total_tokens*100):.2f}%")

        # Show cached tokens (if any)
        cached_tokens = metrics['tokens']['details']['cached_prompt_tokens']
        if cached_tokens > 0:
            print(f"\nCached tokens: {cached_tokens:,}")
            print(f"Cache efficiency: {metrics['efficiency']['cache_efficiency']:.2f}%")

        # Export metrics
        rlm.export_metrics("grok_reasoning_metrics.json")
        print("\nMetrics exported to grok_reasoning_metrics.json")

    except Exception as e:
        print(f"\nError: {str(e)}")
        rlm.print_metrics()


if __name__ == "__main__":
    main()
