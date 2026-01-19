"""
Comprehensive Integration Tests for Rlm.

Supports multiple providers: OpenAI (gpt-5-mini, gpt-4o, etc.) and xAI (grok-4, etc.)

This module provides extensive tests including:
- Needle in the haystack (256k tokens)
- Multi-needle retrieval
- Long context reasoning
- Edge case handling
- Performance benchmarking

Run with: python -m tests.test_rlm_comprehensive
Or: python tests/test_rlm_comprehensive.py
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from rlm import RecursiveLanguageModel
from tests.test_data_generator import (
    generate_needle_haystack_prompt,
    generate_multi_needle_prompt,
    generate_reasoning_test_prompt,
    generate_edge_case_prompts,
)


class TestResult:
    """Container for test results."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.result = None
        self.expected = None
        self.error = None
        self.duration = 0.0
        self.metrics = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "result": str(self.result)[:500] if self.result else None,
            "expected": str(self.expected)[:500] if self.expected else None,
            "error": str(self.error) if self.error else None,
            "duration_seconds": round(self.duration, 2),
            "metrics": self.metrics
        }


class ComprehensiveTest:
    """
    Comprehensive test suite for Rlm.

    Supports both OpenAI and xAI providers.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "grok-4-1-fast-reasoning",
        provider: str = "xai",
        verbose: bool = True,
        max_cost: float = 10.0
    ):
        """
        Initialize test suite.

        Args:
            api_key: API key for the provider
            model: Model to use (e.g., 'gpt-5-mini', 'grok-4-1-fast-reasoning')
            provider: Provider name ('openai' or 'xai')
            verbose: Print detailed output
            max_cost: Maximum cost per test in USD
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.verbose = verbose
        self.max_cost = max_cost
        self.results: list[TestResult] = []

    def _create_rlm(self) -> RecursiveLanguageModel:
        """Create a fresh RLM instance for testing."""
        return RecursiveLanguageModel(
            api_key=self.api_key,
            model=self.model,
            provider=self.provider,
            max_cost=self.max_cost,
            enable_cache=True
        )

    def _log(self, message: str):
        """Log a message if verbose mode is on."""
        if self.verbose:
            print(message)

    def test_needle_in_haystack_256k(self, position: str = "middle") -> TestResult:
        """
        Test: Find a needle in 256k tokens of haystack.

        This is the canonical long-context retrieval test.
        """
        test_name = f"needle_in_haystack_256k_{position}"
        result = TestResult(test_name)

        self._log(f"\n{'='*60}")
        self._log(f"TEST: {test_name}")
        self._log(f"{'='*60}")

        try:
            # Generate test data
            self._log("Generating 256k context...")
            context, needle, question = generate_needle_haystack_prompt(
                target_tokens=256_000,
                needle_position=position
            )

            result.expected = "ALPHA-7749-OMEGA"

            self._log(f"Context size: {len(context):,} chars (~{len(context)//4:,} tokens)")
            self._log(f"Question: {question}")

            # Create RLM and run
            rlm = self._create_rlm()

            start_time = time.time()
            answer = rlm.run(
                task=question,
                context=context,
                max_iterations=30,
                verbose=self.verbose
            )
            result.duration = time.time() - start_time

            result.result = answer
            result.metrics = {
                "total_cost": rlm.metrics.total_cost,
                "total_tokens": rlm.metrics.total_tokens,
                "sub_calls": rlm.metrics.sub_calls,
                "iterations": rlm.metrics.iterations,
                "reasoning_tokens": rlm.metrics.total_completion_reasoning_tokens
            }

            # Check if answer contains the code
            if result.expected.lower() in str(answer).lower():
                result.passed = True
                self._log(f"\n[PASS] Found the needle: {result.expected}")
            else:
                self._log(f"\n[FAIL] Expected '{result.expected}' in answer")
                self._log(f"Got: {answer}")

        except Exception as e:
            result.error = str(e)
            self._log(f"\n[ERROR] {e}")

        self.results.append(result)
        return result

    def test_multi_needle_retrieval(self, num_needles: int = 3) -> TestResult:
        """
        Test: Find multiple needles in a large context.
        """
        test_name = f"multi_needle_{num_needles}"
        result = TestResult(test_name)

        self._log(f"\n{'='*60}")
        self._log(f"TEST: {test_name}")
        self._log(f"{'='*60}")

        try:
            # Generate test data
            self._log(f"Generating context with {num_needles} needles...")
            context, needles_info, question = generate_multi_needle_prompt(
                target_tokens=100_000,  # Smaller for multi-needle
                num_needles=num_needles
            )

            result.expected = needles_info

            self._log(f"Context size: {len(context):,} chars (~{len(context)//4:,} tokens)")

            # Create RLM and run
            rlm = self._create_rlm()

            start_time = time.time()
            answer = rlm.run(
                task=question,
                context=context,
                max_iterations=40,
                verbose=self.verbose
            )
            result.duration = time.time() - start_time

            result.result = answer
            result.metrics = {
                "total_cost": rlm.metrics.total_cost,
                "total_tokens": rlm.metrics.total_tokens,
                "sub_calls": rlm.metrics.sub_calls,
                "iterations": rlm.metrics.iterations,
                "reasoning_tokens": rlm.metrics.total_completion_reasoning_tokens
            }

            # Check how many needles were found
            found_count = 0
            for needle in needles_info:
                if needle["value"].lower() in str(answer).lower() or \
                   needle["code"].lower() in str(answer).lower():
                    found_count += 1

            result.passed = found_count >= (num_needles * 0.8)  # 80% threshold
            self._log(f"\n[{'PASS' if result.passed else 'FAIL'}] Found {found_count}/{num_needles} needles")

        except Exception as e:
            result.error = str(e)
            self._log(f"\n[ERROR] {e}")

        self.results.append(result)
        return result

    def test_reasoning_over_context(self, complexity: str = "medium") -> TestResult:
        """
        Test: Answer questions requiring reasoning over distributed facts.
        """
        test_name = f"reasoning_{complexity}"
        result = TestResult(test_name)

        self._log(f"\n{'='*60}")
        self._log(f"TEST: {test_name}")
        self._log(f"{'='*60}")

        try:
            # Generate test data
            self._log(f"Generating {complexity} reasoning test...")
            context, expected_answer, question = generate_reasoning_test_prompt(
                target_tokens=50_000,
                complexity=complexity
            )

            result.expected = expected_answer

            self._log(f"Context size: {len(context):,} chars (~{len(context)//4:,} tokens)")
            self._log(f"Question: {question}")

            # Create RLM and run
            rlm = self._create_rlm()

            start_time = time.time()
            answer = rlm.run(
                task=question,
                context=context,
                max_iterations=25,
                verbose=self.verbose
            )
            result.duration = time.time() - start_time

            result.result = answer
            result.metrics = {
                "total_cost": rlm.metrics.total_cost,
                "total_tokens": rlm.metrics.total_tokens,
                "sub_calls": rlm.metrics.sub_calls,
                "iterations": rlm.metrics.iterations,
                "reasoning_tokens": rlm.metrics.total_completion_reasoning_tokens
            }

            # For reasoning tests, check key terms are present
            answer_lower = str(answer).lower()

            if complexity == "simple":
                result.passed = "bob" in answer_lower and ("cto" in answer_lower or "engineering" in answer_lower)
            elif complexity == "medium":
                result.passed = "bob" in answer_lower and "carol" in answer_lower
            else:  # complex
                result.passed = "carol" in answer_lower and ("approve" in answer_lower or "veto" in answer_lower)

            self._log(f"\n[{'PASS' if result.passed else 'FAIL'}] Reasoning test")
            if not result.passed:
                self._log(f"Expected key elements from: {expected_answer[:200]}")
                self._log(f"Got: {answer}")

        except Exception as e:
            result.error = str(e)
            self._log(f"\n[ERROR] {e}")

        self.results.append(result)
        return result

    def test_edge_case(self, case_name: str) -> TestResult:
        """
        Test: Handle various edge cases.
        """
        test_name = f"edge_case_{case_name}"
        result = TestResult(test_name)

        self._log(f"\n{'='*60}")
        self._log(f"TEST: {test_name}")
        self._log(f"{'='*60}")

        try:
            # Find the test case
            edge_cases = generate_edge_case_prompts()
            test_case = next((tc for tc in edge_cases if tc["name"] == case_name), None)

            if not test_case:
                raise ValueError(f"Unknown edge case: {case_name}")

            context = test_case["context"]
            question = test_case["question"]
            result.expected = test_case["expected"]

            self._log(f"Description: {test_case['description']}")
            self._log(f"Context size: {len(context):,} chars")

            # Create RLM and run
            rlm = self._create_rlm()

            start_time = time.time()
            answer = rlm.run(
                task=question,
                context=context,
                max_iterations=15,
                verbose=self.verbose
            )
            result.duration = time.time() - start_time

            result.result = answer
            result.metrics = {
                "total_cost": rlm.metrics.total_cost,
                "total_tokens": rlm.metrics.total_tokens,
                "sub_calls": rlm.metrics.sub_calls,
                "iterations": rlm.metrics.iterations
            }

            # Check result based on expected type
            answer_lower = str(answer).lower()

            if result.expected == "no_answer":
                result.passed = "no" in answer_lower or "not found" in answer_lower or "doesn't" in answer_lower
            else:
                result.passed = str(result.expected).lower() in answer_lower

            self._log(f"\n[{'PASS' if result.passed else 'FAIL'}] Edge case: {case_name}")

        except Exception as e:
            result.error = str(e)
            self._log(f"\n[ERROR] {e}")

        self.results.append(result)
        return result

    def run_quick_test(self) -> Dict[str, Any]:
        """
        Run a quick sanity check test (small context).
        """
        test_name = "quick_sanity_check"
        result = TestResult(test_name)

        self._log(f"\n{'='*60}")
        self._log(f"TEST: {test_name}")
        self._log(f"{'='*60}")

        try:
            context = """
            Document 1: The company was founded in 2010.
            Document 2: Our main product is called "DataSync Pro".
            Document 3: The secret project code is PHOENIX-42.
            Document 4: We have offices in 15 countries.
            Document 5: The CEO's name is John Smith.
            """

            question = "What is the secret project code?"
            result.expected = "PHOENIX-42"

            rlm = self._create_rlm()

            start_time = time.time()
            answer = rlm.run(
                task=question,
                context=context,
                max_iterations=10,
                verbose=self.verbose
            )
            result.duration = time.time() - start_time

            result.result = answer
            result.metrics = {
                "total_cost": rlm.metrics.total_cost,
                "total_tokens": rlm.metrics.total_tokens,
                "sub_calls": rlm.metrics.sub_calls
            }

            result.passed = "PHOENIX-42" in str(answer).upper()
            self._log(f"\n[{'PASS' if result.passed else 'FAIL'}] Quick test")
            self._log(f"Answer: {answer}")

        except Exception as e:
            result.error = str(e)
            self._log(f"\n[ERROR] {e}")

        self.results.append(result)
        return result.to_dict()

    def run_all_tests(self, include_256k: bool = True) -> Dict[str, Any]:
        """
        Run the complete test suite.

        Args:
            include_256k: Include the 256k token test (expensive)

        Returns:
            Summary of all test results
        """
        provider_name = self.provider.upper()
        self._log("\n" + "="*60)
        self._log(f"Rlm COMPREHENSIVE TEST SUITE - {provider_name}")
        self._log("="*60)
        self._log(f"Provider: {self.provider}")
        self._log(f"Model: {self.model}")
        self._log(f"Max cost per test: ${self.max_cost}")
        self._log(f"Start time: {datetime.now().isoformat()}")
        self._log("="*60)

        # Run tests
        self.run_quick_test()

        if include_256k:
            self.test_needle_in_haystack_256k(position="middle")

        self.test_multi_needle_retrieval(num_needles=3)
        self.test_reasoning_over_context(complexity="medium")

        # Run edge cases
        for case in ["repeated_info", "contradictory_info"]:
            self.test_edge_case(case)

        # Generate summary
        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all test results."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed and not r.error)
        errors = sum(1 for r in self.results if r.error)

        total_cost = sum(r.metrics.get("total_cost", 0) for r in self.results)
        total_duration = sum(r.duration for r in self.results)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "total_tests": len(self.results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": f"{(passed/len(self.results)*100):.1f}%" if self.results else "N/A",
            "total_cost_usd": round(total_cost, 4),
            "total_duration_seconds": round(total_duration, 2),
            "results": [r.to_dict() for r in self.results]
        }

        self._log("\n" + "="*60)
        self._log("TEST SUMMARY")
        self._log("="*60)
        self._log(f"Total: {len(self.results)} | Passed: {passed} | Failed: {failed} | Errors: {errors}")
        self._log(f"Pass Rate: {summary['pass_rate']}")
        self._log(f"Total Cost: ${total_cost:.4f}")
        self._log(f"Total Duration: {total_duration:.1f}s")
        self._log("="*60)

        return summary

    def save_results(self, filepath: str):
        """Save test results to JSON file."""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        self._log(f"\nResults saved to: {filepath}")


# Backward compatibility alias
GrokComprehensiveTest = ComprehensiveTest


def main():
    """Main entry point for running tests."""
    parser = argparse.ArgumentParser(description="Rlm Comprehensive Test Suite")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the quick sanity check test"
    )
    parser.add_argument(
        "--no-256k",
        action="store_true",
        help="Skip the 256k token test (saves cost)"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["needle", "multi", "reasoning", "edge"],
        help="Run a specific test type"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model to use (e.g., gpt-5-mini, grok-4-1-fast-reasoning)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "xai"],
        help="Provider to use (auto-detected from model if not specified)"
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=10.0,
        help="Maximum cost per test in USD"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Auto-detect provider from model if not specified
    provider = args.provider
    if not provider:
        if args.model.startswith("grok"):
            provider = "xai"
        else:
            provider = "openai"

    # Get appropriate API key based on provider
    if provider == "xai":
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            print("ERROR: Please set XAI_API_KEY environment variable")
            print("Example: export XAI_API_KEY='your-key-here'")
            sys.exit(1)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: Please set OPENAI_API_KEY environment variable")
            print("Example: export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)

    # Create test suite
    test_suite = ComprehensiveTest(
        api_key=api_key,
        model=args.model,
        provider=provider,
        verbose=not args.quiet,
        max_cost=args.max_cost
    )

    # Run appropriate tests
    if args.quick:
        result = test_suite.run_quick_test()
        print(json.dumps(result, indent=2))
    elif args.test == "needle":
        test_suite.test_needle_in_haystack_256k()
        summary = test_suite.get_summary()
    elif args.test == "multi":
        test_suite.test_multi_needle_retrieval()
        summary = test_suite.get_summary()
    elif args.test == "reasoning":
        test_suite.test_reasoning_over_context(complexity="medium")
        summary = test_suite.get_summary()
    elif args.test == "edge":
        for case in generate_edge_case_prompts():
            test_suite.test_edge_case(case["name"])
        summary = test_suite.get_summary()
    else:
        summary = test_suite.run_all_tests(include_256k=not args.no_256k)

    # Save results if requested
    if args.output:
        test_suite.save_results(args.output)


if __name__ == "__main__":
    main()
