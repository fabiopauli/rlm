"""
Metrics tracking for RLM operations.

Tracks tokens, costs, recursion depth, sub-calls, and execution time.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


@dataclass
class CallMetrics:
    """Metrics for a single LLM call with detailed token breakdown."""

    timestamp: float
    model: str

    # Main token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Detailed prompt tokens
    prompt_text_tokens: int = 0
    prompt_audio_tokens: int = 0
    prompt_image_tokens: int = 0
    prompt_cached_tokens: int = 0

    # Detailed completion tokens
    completion_reasoning_tokens: int = 0
    completion_audio_tokens: int = 0
    completion_accepted_prediction_tokens: int = 0
    completion_rejected_prediction_tokens: int = 0

    # Metadata
    num_sources_used: int = 0
    cost: float = 0.0
    duration: float = 0.0
    is_sub_call: bool = False
    depth: int = 0
    parent_id: Optional[str] = None
    call_id: str = ""


@dataclass
class RLMMetrics:
    """
    Comprehensive metrics tracking for RLM operations.

    Tracks:
    - Total tokens used (prompt + completion)
    - Detailed token breakdown (cached, reasoning, etc.)
    - Estimated costs based on model pricing
    - Number and depth of sub-calls
    - Execution time
    - Recursion patterns
    """

    # Pricing provider (callable that returns pricing dict for a model)
    pricing_provider: Optional[callable] = None

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Token counts (main)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0

    # Detailed token counts
    total_prompt_cached_tokens: int = 0
    total_completion_reasoning_tokens: int = 0
    total_prompt_image_tokens: int = 0
    total_prompt_audio_tokens: int = 0

    # Cost tracking
    total_cost: float = 0.0
    cost_by_model: Dict[str, float] = field(default_factory=dict)

    # Sub-call tracking
    total_calls: int = 0
    sub_calls: int = 0
    max_recursion_depth: int = 0

    # Call history
    call_history: List[CallMetrics] = field(default_factory=list)

    # Iteration tracking
    iterations: int = 0

    def record_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        is_sub_call: bool = False,
        depth: int = 0,
        duration: float = 0.0,
        call_id: str = "",
        parent_id: Optional[str] = None,
        # Detailed token counts
        prompt_cached_tokens: int = 0,
        completion_reasoning_tokens: int = 0,
        prompt_text_tokens: int = 0,
        prompt_image_tokens: int = 0,
        prompt_audio_tokens: int = 0,
        completion_audio_tokens: int = 0,
        completion_accepted_prediction_tokens: int = 0,
        completion_rejected_prediction_tokens: int = 0,
        num_sources_used: int = 0
    ):
        """Record metrics for a single LLM call with detailed token breakdown."""
        total = prompt_tokens + completion_tokens
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        # Update main totals
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total
        self.total_cost += cost
        self.total_calls += 1

        # Update detailed totals
        self.total_prompt_cached_tokens += prompt_cached_tokens
        self.total_completion_reasoning_tokens += completion_reasoning_tokens
        self.total_prompt_image_tokens += prompt_image_tokens
        self.total_prompt_audio_tokens += prompt_audio_tokens

        if is_sub_call:
            self.sub_calls += 1

        # Update max recursion depth
        self.max_recursion_depth = max(self.max_recursion_depth, depth)

        # Update cost by model
        if model not in self.cost_by_model:
            self.cost_by_model[model] = 0.0
        self.cost_by_model[model] += cost

        # Record call details
        call_metrics = CallMetrics(
            timestamp=time.time(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            prompt_text_tokens=prompt_text_tokens,
            prompt_audio_tokens=prompt_audio_tokens,
            prompt_image_tokens=prompt_image_tokens,
            prompt_cached_tokens=prompt_cached_tokens,
            completion_reasoning_tokens=completion_reasoning_tokens,
            completion_audio_tokens=completion_audio_tokens,
            completion_accepted_prediction_tokens=completion_accepted_prediction_tokens,
            completion_rejected_prediction_tokens=completion_rejected_prediction_tokens,
            num_sources_used=num_sources_used,
            cost=cost,
            duration=duration,
            is_sub_call=is_sub_call,
            depth=depth,
            call_id=call_id,
            parent_id=parent_id
        )
        self.call_history.append(call_metrics)

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for a call based on model pricing."""
        # Use pricing provider if available
        if self.pricing_provider:
            pricing = self.pricing_provider(model)
        else:
            # Fallback to default pricing (USD per 1K tokens)
            DEFAULT_PRICING = {
                "gpt-5-mini": {"prompt": 0.00025, "completion": 0.002},
                "gpt-5-nano": {"prompt": 0.00005, "completion": 0.0004},
                "gpt-4.1": {"prompt": 0.002, "completion": 0.008},
                "gpt-4.1-mini": {"prompt": 0.0004, "completion": 0.0016},
                "gpt-4.1-nano": {"prompt": 0.0001, "completion": 0.0004},
                "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
                "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
                "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
                "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
                "grok-4": {"prompt": 0.002, "completion": 0.010},
                "grok-4-1-fast-reasoning": {"prompt": 0.0002, "completion": 0.0005},
                "grok-4-1-fast-non-reasoning": {"prompt": 0.0002, "completion": 0.0005},
                "grok-4-fast-reasoning": {"prompt": 0.002, "completion": 0.010},
                "grok-4-fast-non-reasoning": {"prompt": 0.002, "completion": 0.010},
            }
            pricing = DEFAULT_PRICING.get(model, DEFAULT_PRICING["gpt-4o"])

        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]

        return prompt_cost + completion_cost

    def increment_iteration(self):
        """Increment the iteration counter."""
        self.iterations += 1

    def finalize(self):
        """Mark the end of RLM execution."""
        self.end_time = time.time()

    def get_duration(self) -> float:
        """Get total execution duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    def get_summary(self) -> Dict:
        """Get a summary of all metrics including detailed token breakdown."""
        return {
            "duration_seconds": round(self.get_duration(), 2),
            "iterations": self.iterations,
            "total_calls": self.total_calls,
            "sub_calls": self.sub_calls,
            "max_recursion_depth": self.max_recursion_depth,
            "tokens": {
                "prompt": self.total_prompt_tokens,
                "completion": self.total_completion_tokens,
                "total": self.total_tokens,
                "details": {
                    "cached_prompt_tokens": self.total_prompt_cached_tokens,
                    "reasoning_tokens": self.total_completion_reasoning_tokens,
                    "image_tokens": self.total_prompt_image_tokens,
                    "audio_tokens": self.total_prompt_audio_tokens,
                }
            },
            "cost": {
                "total_usd": round(self.total_cost, 4),
                "by_model": {k: round(v, 4) for k, v in self.cost_by_model.items()},
            },
            "efficiency": {
                "tokens_per_call": round(self.total_tokens / self.total_calls, 2) if self.total_calls > 0 else 0,
                "cost_per_call": round(self.total_cost / self.total_calls, 4) if self.total_calls > 0 else 0,
                "sub_call_ratio": round(self.sub_calls / self.total_calls, 2) if self.total_calls > 0 else 0,
                "cache_efficiency": round(self.total_prompt_cached_tokens / self.total_prompt_tokens * 100, 2) if self.total_prompt_tokens > 0 else 0,
            }
        }

    def print_summary(self):
        """Print a formatted summary of metrics."""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("RLM Execution Metrics Summary")
        print("="*60)

        print(f"\nExecution:")
        print(f"  Duration: {summary['duration_seconds']}s")
        print(f"  Iterations: {summary['iterations']}")

        print(f"\nCalls:")
        print(f"  Total: {summary['total_calls']}")
        print(f"  Sub-calls: {summary['sub_calls']}")
        print(f"  Max recursion depth: {summary['max_recursion_depth']}")

        print(f"\nTokens:")
        print(f"  Prompt: {summary['tokens']['prompt']:,}")
        print(f"  Completion: {summary['tokens']['completion']:,}")
        print(f"  Total: {summary['tokens']['total']:,}")

        print(f"\nCost:")
        print(f"  Total: ${summary['cost']['total_usd']}")
        for model, cost in summary['cost']['by_model'].items():
            print(f"    {model}: ${cost}")

        print(f"\nEfficiency:")
        print(f"  Tokens/call: {summary['efficiency']['tokens_per_call']}")
        print(f"  Cost/call: ${summary['efficiency']['cost_per_call']}")
        print(f"  Sub-call ratio: {summary['efficiency']['sub_call_ratio']}")

        print("="*60 + "\n")

    def export_to_json(self, filepath: str):
        """Export metrics to a JSON file."""
        data = {
            "summary": self.get_summary(),
            "call_history": [
                {
                    "timestamp": call.timestamp,
                    "model": call.model,
                    "prompt_tokens": call.prompt_tokens,
                    "completion_tokens": call.completion_tokens,
                    "total_tokens": call.total_tokens,
                    "cost": call.cost,
                    "duration": call.duration,
                    "is_sub_call": call.is_sub_call,
                    "depth": call.depth,
                    "call_id": call.call_id,
                    "parent_id": call.parent_id,
                }
                for call in self.call_history
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def check_budget_exceeded(self, max_cost: Optional[float] = None, max_tokens: Optional[int] = None) -> bool:
        """Check if budget limits have been exceeded."""
        if max_cost is not None and self.total_cost > max_cost:
            return True
        if max_tokens is not None and self.total_tokens > max_tokens:
            return True
        return False
