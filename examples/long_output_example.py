"""
Long output generation example.

Demonstrates building extended responses by aggregating sub-call outputs.
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

    # Create context with multiple topics
    context = """
    # Topics for Summary

    ## Artificial Intelligence
    Artificial Intelligence (AI) is the simulation of human intelligence processes by machines,
    especially computer systems. These processes include learning, reasoning, and self-correction.
    AI applications include expert systems, natural language processing, speech recognition, and
    machine vision. The field was founded on the assumption that human intelligence can be so
    precisely described that a machine can be made to simulate it.

    ## Quantum Computing
    Quantum computing is a type of computation that harnesses the collective properties of quantum
    states, such as superposition, interference, and entanglement, to perform calculations. Quantum
    computers have the potential to solve certain computational problems much faster than classical
    computers. They are particularly suited for problems involving optimization, cryptography, and
    simulation of quantum systems.

    ## Renewable Energy
    Renewable energy is energy from sources that are naturally replenishing but flow-limited.
    Renewable resources are virtually inexhaustible in duration but limited in the amount of energy
    that is available per unit of time. The major types include solar, wind, hydroelectric, biomass,
    and geothermal. Renewable energy is increasingly important in addressing climate change and
    reducing dependence on fossil fuels.

    ## Biotechnology
    Biotechnology is the use of living systems and organisms to develop or make products. It has
    applications in areas such as medicine, agriculture, and industrial processes. Modern
    biotechnology includes genetic engineering, which allows scientists to modify DNA and create
    organisms with desired traits. CRISPR technology has revolutionized gene editing capabilities.

    ## Space Exploration
    Space exploration is the use of astronomy and space technology to explore outer space. It is
    carried out mainly by robotic spacecraft and human spaceflight. Major milestones include the
    Moon landing, Mars rovers, the International Space Station, and the James Webb Space Telescope.
    Future goals include returning humans to the Moon and sending crewed missions to Mars.
    """

    # Initialize RLM
    rlm = RecursiveLanguageModel(
        api_key=api_key,
        model="gpt-4o",
        sub_model="gpt-4o-mini",
        enable_cache=True,
    )

    # Task: Generate comprehensive summary
    task = """
    Generate a comprehensive summary document covering all topics in the context.
    For each topic, create a detailed paragraph (3-5 sentences) using llm_query.
    Combine all paragraphs into a well-structured final document.

    Use the find_sections() helper to identify topics, then process each with llm_query.
    """

    print("="*60)
    print("Long Output Example: Multi-Topic Summary")
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

        # Save result to file
        with open("long_output_result.txt", "w") as f:
            f.write(result)
        print("Result saved to long_output_result.txt")

        # Export metrics
        rlm.export_metrics("long_output_metrics.json")

    except Exception as e:
        print(f"\nError: {str(e)}")
        rlm.print_metrics()


if __name__ == "__main__":
    main()
