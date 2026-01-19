"""
Advanced patterns example: Map-Reduce and Recursive Split.

Demonstrates sophisticated decomposition strategies.
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

    # Create context with structured data
    context = """
    # Customer Reviews Dataset

    Review 1: "This product is amazing! Best purchase I've made this year. Highly recommended. 5/5"
    Review 2: "Terrible quality. Broke after one use. Very disappointed. 1/5"
    Review 3: "Good value for money. Works as expected. Happy with my purchase. 4/5"
    Review 4: "Not worth the price. Poor customer service. Would not buy again. 2/5"
    Review 5: "Excellent product! Exceeded my expectations. Will buy again. 5/5"
    Review 6: "Average product. Nothing special but does the job. 3/5"
    Review 7: "Fantastic! Love everything about it. Best in class. 5/5"
    Review 8: "Mediocre at best. Expected better quality. 2/5"
    Review 9: "Great value! Works perfectly. Very satisfied. 4/5"
    Review 10: "Outstanding quality. Professional grade. Worth every penny. 5/5"
    """ * 50  # Repeat to create larger dataset

    # Initialize RLM
    rlm = RecursiveLanguageModel(
        api_key=api_key,
        model="gpt-4o",
        sub_model="gpt-4o-mini",
        enable_cache=True,  # Cache will help with repeated reviews
    )

    # Task: Sentiment analysis with detailed breakdown
    task = """
    Perform sentiment analysis on the customer reviews using a map-reduce pattern:

    1. Split the context into individual reviews
    2. For each review, use llm_query to classify sentiment as: Positive, Negative, or Neutral
    3. Extract the rating (X/5)
    4. Aggregate results to get:
       - Total count of each sentiment
       - Average rating
       - Sentiment distribution

    Use the map_reduce() helper function for efficient processing.
    """

    print("="*60)
    print("Advanced Patterns: Map-Reduce Sentiment Analysis")
    print("="*60)
    print(f"\nTask: {task}\n")

    try:
        result = rlm.run(task=task, context=context, verbose=True, max_iterations=30)

        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"\n{result}\n")

        # Print metrics
        rlm.print_metrics()

        # Export metrics
        rlm.export_metrics("advanced_patterns_metrics.json")

        # Check cache effectiveness
        print("\nCache was particularly useful here due to repeated reviews!")

    except Exception as e:
        print(f"\nError: {str(e)}")
        rlm.print_metrics()


if __name__ == "__main__":
    main()
