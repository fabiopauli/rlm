"""
Basic usage example for Recursive Language Model.

Demonstrates simple needle-in-haystack task.
"""

import os
from dotenv import load_dotenv
from rlm import RecursiveLanguageModel


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Create a long context with a hidden "needle"
    context = """
    This is a very long document with lots of information.
    """ + "\n".join([f"This is line {i} with some random content about topic {i % 10}."
                     for i in range(1000)]) + """

    Here is the important information: The magic number is 42.

    """ + "\n".join([f"More content on line {i} discussing various subjects."
                     for i in range(1000, 2000)])

    # Initialize RLM
    rlm = RecursiveLanguageModel(
        api_key=api_key,
        model="gpt-4o",
        sub_model="gpt-4o-mini",  # Use cheaper model for sub-calls
        enable_cache=True,
        max_cost=1.0  # Budget limit: $1
    )

    # Run task
    task = "Find the magic number mentioned in the context."

    print("="*60)
    print("Basic RLM Example: Needle in Haystack")
    print("="*60)
    print(f"\nTask: {task}")
    print(f"Context length: {len(context):,} characters\n")

    try:
        result = rlm.run(task=task, context=context, verbose=True)

        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"\n{result}\n")

        # Print metrics
        rlm.print_metrics()

        # Export metrics to file
        rlm.export_metrics("basic_usage_metrics.json")
        print("Metrics exported to basic_usage_metrics.json")

    except Exception as e:
        print(f"\nError: {str(e)}")
        rlm.print_metrics()


if __name__ == "__main__":
    main()
