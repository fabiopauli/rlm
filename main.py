#!/usr/bin/env python3
"""
Rlm - Recursive Language Model

Main entry point for running RLM tasks against long contexts.
Supports multiple LLM providers (OpenAI, xAI Grok).

Usage:
    python main.py run --task "Find the secret code" --context-file data.txt
    python main.py run --task "Summarize" --context "Your text here" --provider xai
    python main.py test --quick
    python main.py test --suite comprehensive
    python main.py info

Environment Variables:
    XAI_API_KEY: API key for xAI Grok
    OPENAI_API_KEY: API key for OpenAI
"""

import os
import sys
import json
import argparse
from typing import Optional

from dotenv import load_dotenv


def detect_provider_from_model(model: str) -> str:
    """
    Auto-detect the provider based on model name.

    Args:
        model: Model name (e.g., 'gpt-5-mini', 'grok-4-1-fast-reasoning')

    Returns:
        Provider name: 'openai' or 'xai'
    """
    model_lower = model.lower()

    # OpenAI models
    if any(model_lower.startswith(prefix) for prefix in ['gpt-', 'o1', 'o3', 'o4']):
        return "openai"

    # xAI Grok models
    if model_lower.startswith('grok'):
        return "xai"

    # Default to OpenAI for unknown models
    return "openai"


def get_api_key_for_provider(provider: str) -> Optional[str]:
    """
    Get the API key for the given provider from environment.

    Args:
        provider: Provider name ('openai' or 'xai')

    Returns:
        API key or None if not set
    """
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif provider == "xai":
        return os.getenv("XAI_API_KEY")
    return None


def cmd_run(args):
    """Run an RLM task."""
    from rlm import RecursiveLanguageModel

    # Get context
    if args.context_file:
        with open(args.context_file, 'r') as f:
            context = f.read()
    elif args.context:
        context = args.context
    else:
        print("ERROR: Must provide --context or --context-file")
        sys.exit(1)

    # Get API key
    if args.provider == "xai":
        api_key = args.api_key or os.getenv("XAI_API_KEY")
        default_model = "grok-4-1-fast-reasoning"
    else:
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        default_model = "gpt-4o"

    if not api_key:
        env_var = "XAI_API_KEY" if args.provider == "xai" else "OPENAI_API_KEY"
        print(f"ERROR: Please set {env_var} environment variable or use --api-key")
        sys.exit(1)

    model = args.model or default_model

    print(f"Rlm Task Runner")
    print(f"================")
    print(f"Provider: {args.provider}")
    print(f"Model: {model}")
    print(f"Context length: {len(context):,} characters (~{len(context)//4:,} tokens)")
    print(f"Task: {args.task}")
    print()

    # Create RLM
    rlm = RecursiveLanguageModel(
        api_key=api_key,
        model=model,
        provider=args.provider,
        max_cost=args.max_cost,
        enable_cache=not args.no_cache
    )

    # Run task
    try:
        result = rlm.run(
            task=args.task,
            context=context,
            max_iterations=args.max_iterations,
            verbose=not args.quiet
        )

        print("\n" + "="*60)
        print("RESULT:")
        print("="*60)
        print(result)
        print("="*60)

        # Print metrics
        print("\nMetrics:")
        print(f"  Total cost: ${rlm.metrics.total_cost:.4f}")
        print(f"  Total tokens: {rlm.metrics.total_tokens:,}")
        print(f"  Reasoning tokens: {rlm.metrics.total_completion_reasoning_tokens:,}")
        print(f"  Sub-calls: {rlm.metrics.sub_calls}")
        print(f"  Iterations: {rlm.metrics.iterations}")

        # Save result if requested
        if args.output:
            output_data = {
                "task": args.task,
                "result": result,
                "metrics": {
                    "total_cost": rlm.metrics.total_cost,
                    "total_tokens": rlm.metrics.total_tokens,
                    "sub_calls": rlm.metrics.sub_calls,
                    "iterations": rlm.metrics.iterations
                }
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


def cmd_test(args):
    """Run tests."""
    # Determine model and provider
    default_model = "gpt-5-mini"
    model = args.model or default_model
    provider = detect_provider_from_model(model)

    # Get API key for the detected provider
    api_key = get_api_key_for_provider(provider)
    if not api_key:
        env_var = "OPENAI_API_KEY" if provider == "openai" else "XAI_API_KEY"
        print(f"ERROR: Please set {env_var} environment variable for model '{model}'")
        sys.exit(1)

    if args.quick:
        # Run quick sanity check
        from tests.test_rlm_comprehensive import ComprehensiveTest

        test_suite = ComprehensiveTest(
            api_key=api_key,
            model=model,
            provider=provider,
            verbose=not args.quiet
        )
        result = test_suite.run_quick_test()
        print("\nQuick Test Result:")
        print(json.dumps(result, indent=2))

    elif args.suite == "comprehensive":
        # Run full comprehensive test suite
        from tests.test_rlm_comprehensive import ComprehensiveTest

        test_suite = ComprehensiveTest(
            api_key=api_key,
            model=model,
            provider=provider,
            verbose=not args.quiet,
            max_cost=args.max_cost or 10.0
        )

        summary = test_suite.run_all_tests(include_256k=not args.no_256k)

        if args.output:
            test_suite.save_results(args.output)

    elif args.suite == "unit":
        # Run unit tests with pytest
        import subprocess
        cmd = ["python", "-m", "pytest", "tests/", "-v"]
        if args.quiet:
            cmd.append("-q")
        subprocess.run(cmd)

    else:
        print("ERROR: Must specify --quick, --suite comprehensive, or --suite unit")
        sys.exit(1)


def cmd_info(args):
    """Show information about Rlm."""
    from rlm import RecursiveLanguageModel
    from rlm.providers import OpenAIProvider, XAIProvider

    print("Rlm - Recursive Language Model")
    print("================================")
    print()
    print("Version: 0.1.0")
    print()
    print("Supported Providers:")
    print("  - OpenAI (gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4o)")
    print("  - xAI Grok (grok-4, grok-4-1-fast-reasoning, grok-4-fast-reasoning)")
    print()
    print("OpenAI Models:")
    for model, window in OpenAIProvider.CONTEXT_WINDOWS.items():
        pricing = OpenAIProvider.PRICING.get(model, {})
        print(f"  - {model}: {window:,} tokens")
        if pricing:
            print(f"    Pricing: ${pricing.get('prompt', 0):.4f}/${pricing.get('completion', 0):.4f} per 1K tokens")
    print()
    print("xAI Grok Models:")
    for model, window in XAIProvider.CONTEXT_WINDOWS.items():
        pricing = XAIProvider.PRICING.get(model, {})
        print(f"  - {model}: {window:,} tokens")
        if pricing:
            print(f"    Pricing: ${pricing.get('prompt', 0):.4f}/${pricing.get('completion', 0):.4f} per 1K tokens")
    print()
    print("Environment Variables:")
    print(f"  XAI_API_KEY: {'Set' if os.getenv('XAI_API_KEY') else 'Not set'}")
    print(f"  OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print()
    print("Usage Examples:")
    print("  python main.py run --task 'Find the secret' --context-file doc.txt")
    print("  python main.py run --task 'Summarize' --context 'text' --provider openai")
    print("  python main.py test --quick")
    print("  python main.py test --suite comprehensive --no-256k")


def cmd_generate_test(args):
    """Generate test data files."""
    from tests.test_data_generator import (
        generate_needle_haystack_prompt,
        generate_multi_needle_prompt,
        generate_reasoning_test_prompt,
        save_test_prompt
    )

    output_dir = args.output_dir or "tests/data"
    os.makedirs(output_dir, exist_ok=True)

    if args.type == "needle" or args.type == "all":
        print(f"Generating {args.tokens}k needle-in-haystack test...")
        context, needle, question = generate_needle_haystack_prompt(
            target_tokens=args.tokens * 1000,
            needle_position=args.position
        )
        filepath = os.path.join(output_dir, f"needle_{args.tokens}k_{args.position}.json")
        save_test_prompt(filepath, context, question, {"needle": needle, "position": args.position})
        print(f"  Saved to: {filepath}")

    if args.type == "multi" or args.type == "all":
        print("Generating multi-needle test...")
        context, needles, question = generate_multi_needle_prompt(
            target_tokens=100_000,
            num_needles=args.num_needles
        )
        filepath = os.path.join(output_dir, f"multi_needle_{args.num_needles}.json")
        save_test_prompt(filepath, context, question, {"needles": needles})
        print(f"  Saved to: {filepath}")

    if args.type == "reasoning" or args.type == "all":
        print("Generating reasoning test...")
        context, expected, question = generate_reasoning_test_prompt(
            target_tokens=50_000,
            complexity=args.complexity
        )
        filepath = os.path.join(output_dir, f"reasoning_{args.complexity}.json")
        save_test_prompt(filepath, context, question, {"expected_answer": expected})
        print(f"  Saved to: {filepath}")

    print("\nDone!")


def main():
    """Main entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Rlm - Recursive Language Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run --task "Find the secret code" --context-file data.txt
  python main.py run --task "Summarize" --context "Your text" --provider xai
  python main.py test --quick
  python main.py test --suite comprehensive --no-256k
  python main.py info
  python main.py generate --type needle --tokens 256
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an RLM task")
    run_parser.add_argument("--task", "-t", required=True, help="Task description/question")
    run_parser.add_argument("--context", "-c", help="Context string")
    run_parser.add_argument("--context-file", "-f", help="Path to context file")
    run_parser.add_argument("--provider", "-p", default="xai", choices=["xai", "openai"],
                           help="LLM provider (default: xai)")
    run_parser.add_argument("--model", "-m", help="Model to use (default: auto)")
    run_parser.add_argument("--api-key", help="API key (or use environment variable)")
    run_parser.add_argument("--max-cost", type=float, default=5.0,
                           help="Maximum cost in USD (default: 5.0)")
    run_parser.add_argument("--max-iterations", type=int, default=50,
                           help="Maximum iterations (default: 50)")
    run_parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    run_parser.add_argument("--output", "-o", help="Save results to JSON file")
    run_parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output")
    run_parser.set_defaults(func=cmd_run)

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--quick", action="store_true", help="Run quick sanity check")
    test_parser.add_argument("--suite", choices=["comprehensive", "unit"],
                            help="Test suite to run")
    test_parser.add_argument("--no-256k", action="store_true",
                            help="Skip 256k token test (saves cost)")
    test_parser.add_argument("--model", "-m", help="Model to use for tests")
    test_parser.add_argument("--max-cost", type=float, help="Max cost per test")
    test_parser.add_argument("--output", "-o", help="Save results to JSON file")
    test_parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output")
    test_parser.set_defaults(func=cmd_test)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show Rlm information")
    info_parser.set_defaults(func=cmd_info)

    # Generate test data command
    gen_parser = subparsers.add_parser("generate", help="Generate test data")
    gen_parser.add_argument("--type", choices=["needle", "multi", "reasoning", "all"],
                           default="all", help="Type of test data to generate")
    gen_parser.add_argument("--tokens", type=int, default=256,
                           help="Target tokens in thousands (default: 256)")
    gen_parser.add_argument("--position", choices=["start", "middle", "end", "random"],
                           default="middle", help="Needle position (default: middle)")
    gen_parser.add_argument("--num-needles", type=int, default=5,
                           help="Number of needles for multi-needle test")
    gen_parser.add_argument("--complexity", choices=["simple", "medium", "complex"],
                           default="medium", help="Reasoning test complexity")
    gen_parser.add_argument("--output-dir", help="Output directory (default: tests/data)")
    gen_parser.set_defaults(func=cmd_generate_test)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
