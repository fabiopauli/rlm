"""
Unit tests for RLM metrics system.

Run with: pytest tests/test_metrics.py
"""

import pytest
from rlm.metrics import RLMMetrics


class TestRLMMetrics:
    """Test RLMMetrics functionality."""

    def test_record_call(self):
        metrics = RLMMetrics()

        metrics.record_call(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            is_sub_call=False,
            depth=0
        )

        assert metrics.total_prompt_tokens == 100
        assert metrics.total_completion_tokens == 50
        assert metrics.total_tokens == 150
        assert metrics.total_calls == 1
        assert metrics.sub_calls == 0

    def test_record_sub_call(self):
        metrics = RLMMetrics()

        metrics.record_call(
            model="gpt-4o-mini",
            prompt_tokens=50,
            completion_tokens=25,
            is_sub_call=True,
            depth=1
        )

        assert metrics.sub_calls == 1
        assert metrics.max_recursion_depth == 1

    def test_cost_calculation(self):
        metrics = RLMMetrics()

        # GPT-4o pricing: $0.0025 per 1K prompt, $0.01 per 1K completion
        metrics.record_call(
            model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=1000,
            is_sub_call=False,
            depth=0
        )

        expected_cost = (1000 / 1000 * 0.0025) + (1000 / 1000 * 0.01)
        assert metrics.total_cost == pytest.approx(expected_cost, rel=1e-5)

    def test_cost_by_model(self):
        metrics = RLMMetrics()

        metrics.record_call(
            model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=1000,
            is_sub_call=False,
            depth=0
        )

        metrics.record_call(
            model="gpt-4o-mini",
            prompt_tokens=1000,
            completion_tokens=1000,
            is_sub_call=True,
            depth=1
        )

        assert "gpt-4o" in metrics.cost_by_model
        assert "gpt-4o-mini" in metrics.cost_by_model
        assert metrics.cost_by_model["gpt-4o"] > metrics.cost_by_model["gpt-4o-mini"]

    def test_increment_iteration(self):
        metrics = RLMMetrics()

        metrics.increment_iteration()
        metrics.increment_iteration()
        metrics.increment_iteration()

        assert metrics.iterations == 3

    def test_max_recursion_depth(self):
        metrics = RLMMetrics()

        metrics.record_call("gpt-4o", 100, 50, False, depth=0)
        metrics.record_call("gpt-4o-mini", 100, 50, True, depth=1)
        metrics.record_call("gpt-4o-mini", 100, 50, True, depth=2)
        metrics.record_call("gpt-4o-mini", 100, 50, True, depth=1)

        assert metrics.max_recursion_depth == 2

    def test_get_summary(self):
        metrics = RLMMetrics()

        metrics.record_call("gpt-4o", 100, 50, False, depth=0)
        metrics.record_call("gpt-4o-mini", 100, 50, True, depth=1)
        metrics.increment_iteration()

        summary = metrics.get_summary()

        assert "duration_seconds" in summary
        assert "iterations" in summary
        assert summary["iterations"] == 1
        assert summary["total_calls"] == 2
        assert summary["sub_calls"] == 1
        assert summary["max_recursion_depth"] == 1
        assert "tokens" in summary
        assert "cost" in summary
        assert "efficiency" in summary

    def test_check_budget_exceeded_cost(self):
        metrics = RLMMetrics()

        # Add calls to exceed budget
        for _ in range(10):
            metrics.record_call("gpt-4o", 10000, 10000, False, depth=0)

        # Should exceed $1 budget
        assert metrics.check_budget_exceeded(max_cost=1.0)

    def test_check_budget_exceeded_tokens(self):
        metrics = RLMMetrics()

        metrics.record_call("gpt-4o", 100000, 50000, False, depth=0)

        # Should exceed 100K tokens
        assert metrics.check_budget_exceeded(max_tokens=100000)

    def test_call_history(self):
        metrics = RLMMetrics()

        metrics.record_call(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            is_sub_call=False,
            depth=0,
            call_id="call1"
        )

        metrics.record_call(
            model="gpt-4o-mini",
            prompt_tokens=50,
            completion_tokens=25,
            is_sub_call=True,
            depth=1,
            call_id="call2",
            parent_id="call1"
        )

        assert len(metrics.call_history) == 2
        assert metrics.call_history[0].call_id == "call1"
        assert metrics.call_history[1].call_id == "call2"
        assert metrics.call_history[1].parent_id == "call1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
