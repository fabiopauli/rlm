"""
Classification and aggregation example.

Demonstrates recursive chunking and classification pattern.
"""

import os
from dotenv import load_dotenv
from rlm import RecursiveLanguageModel


def main():
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Create context with items to classify
    items = [
        "Apple - a red fruit",
        "Carrot - an orange vegetable",
        "Banana - a yellow fruit",
        "Broccoli - a green vegetable",
        "Orange - a citrus fruit",
        "Spinach - a leafy green vegetable",
        "Grape - a small purple fruit",
        "Tomato - a red fruit (botanically)",
        "Cucumber - a green vegetable",
        "Strawberry - a red berry fruit",
        "Lettuce - a leafy vegetable",
        "Mango - a tropical fruit",
        "Pepper - a colorful vegetable",
        "Pineapple - a tropical fruit",
        "Zucchini - a green vegetable",
    ] * 100  # Repeat to create larger list

    context = "\n".join(items)

    # Initialize RLM
    rlm = RecursiveLanguageModel(
        api_key=api_key,
        model="gpt-4o",
        sub_model="gpt-4o-mini",
        enable_cache=True,  # Cache will help with repeated items
    )

    # Task: Count fruits vs vegetables
    task = """
    Analyze the list and count how many items are fruits and how many are vegetables.
    Return the counts in the format: "Fruits: X, Vegetables: Y"
    """

    print("="*60)
    print("Classification Example: Fruits vs Vegetables")
    print("="*60)
    print(f"\nTask: {task}")
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

        # Export metrics
        rlm.export_metrics("classification_metrics.json")
        print("Metrics exported to classification_metrics.json")

    except Exception as e:
        print(f"\nError: {str(e)}")
        rlm.print_metrics()


if __name__ == "__main__":
    main()
