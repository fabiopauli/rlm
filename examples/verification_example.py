"""
Verification and consensus checking example.

Demonstrates self-verification pattern.
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

    # Create context with facts to extract and verify
    context = """
    Historical Document: The Discovery of Penicillin

    In 1928, Alexander Fleming, a Scottish bacteriologist, made one of the most
    important medical discoveries in history. While working at St. Mary's Hospital
    in London, he accidentally discovered penicillin.

    The discovery happened when Fleming noticed that a mold (later identified as
    Penicillium notatum) had contaminated one of his bacterial culture plates.
    Surprisingly, the bacteria around the mold had been killed.

    Fleming published his findings in 1929, but it took over a decade before
    penicillin was developed into a practical antibiotic. During World War II,
    Howard Florey and Ernst Boris Chain successfully mass-produced penicillin,
    saving countless lives.

    Fleming, Florey, and Chain were awarded the Nobel Prize in Physiology or
    Medicine in 1945 for their work on penicillin.

    Some incorrect information for testing verification:
    - Penicillin was discovered in 1930 (WRONG - it was 1928)
    - Fleming discovered penicillin in Paris (WRONG - it was London)
    - Fleming won the Nobel Prize in 1950 (WRONG - it was 1945)
    """

    # Initialize RLM
    rlm = RecursiveLanguageModel(
        api_key=api_key,
        model="gpt-4o",
        sub_model="gpt-4o-mini",
        enable_cache=True,
    )

    # Task: Extract and verify key facts
    task = """
    Extract the following facts from the context and verify they are correct:
    1. Year penicillin was discovered
    2. Name of the discoverer
    3. Location where it was discovered
    4. Year the Nobel Prize was awarded

    Use the verify_answer() helper to cross-check each fact.
    Return verified facts only.
    """

    print("="*60)
    print("Verification Example: Historical Facts")
    print("="*60)
    print(f"\nTask: {task}\n")

    try:
        result = rlm.run(task=task, context=context, verbose=True)

        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"\n{result}\n")

        # Print metrics
        rlm.print_metrics()

        # Export metrics
        rlm.export_metrics("verification_metrics.json")

    except Exception as e:
        print(f"\nError: {str(e)}")
        rlm.print_metrics()


if __name__ == "__main__":
    main()
