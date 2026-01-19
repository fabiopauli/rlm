"""
Enhanced Recursive Language Model (RLM) implementation.

Processes extremely long contexts through recursive decomposition and sub-calls.
"""

import openai
import re
import time
import logging
from typing import Optional, Dict, Any, Callable
from uuid import uuid4

from .metrics import RLMMetrics
from .cache import RLMCache
from .security import RestrictedGlobals, ExecutionMonitor
from .helpers import (
    TextProcessor,
    SearchHelper,
    AggregationHelper,
    VerificationHelper,
    RecursionHelper
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class RecursiveLanguageModel:
    """
    Enhanced Recursive Language Model with:
    - Metrics tracking
    - Sub-call caching
    - Security sandboxing
    - Advanced helpers
    - Budget controls
    - Improved error handling
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        sub_model: Optional[str] = None,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: Optional[float] = 3600,
        max_cost: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_security: bool = True,
        log_level: str = "INFO"
    ):
        """
        Initialize the Enhanced RLM.

        Args:
            api_key: Your OpenAI API key
            model: Main model for root LLM (e.g., 'gpt-4o')
            sub_model: Optional cheaper model for sub-calls (e.g., 'gpt-4o-mini')
            enable_cache: Enable caching of sub-call results
            cache_size: Maximum number of cached entries
            cache_ttl: Cache time-to-live in seconds (None = no expiration)
            max_cost: Maximum total cost in USD (None = unlimited)
            max_tokens: Maximum total tokens (None = unlimited)
            enable_security: Enable security sandboxing for code execution
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        # API setup
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.sub_model = sub_model or model

        # Metrics tracking
        self.metrics = RLMMetrics()
        self.max_cost = max_cost
        self.max_tokens = max_tokens

        # Caching
        self.enable_cache = enable_cache
        self.cache = RLMCache(max_size=cache_size, ttl=cache_ttl) if enable_cache else None

        # Security
        self.enable_security = enable_security
        self.execution_monitor = ExecutionMonitor() if enable_security else None

        # REPL state
        self.repl_globals: Dict[str, Any] = {}
        self.final_answer = None
        self.history = []

        # Recursion tracking
        self.recursion_depth = 0
        self.current_call_id = None

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Helper instances
        self.text_processor = TextProcessor()
        self.search_helper = SearchHelper()
        self.aggregation_helper = AggregationHelper()
        self.verification_helper = VerificationHelper()
        self.recursion_helper = RecursionHelper()

    def _extract_code(self, text: str) -> list[str]:
        """
        Extract all code blocks labeled as 'repl' from the LLM response.

        Args:
            text: LLM response text

        Returns:
            List of code blocks
        """
        # Find all code blocks (not just first one)
        matches = re.findall(r'```repl\n(.*?)\n```', text, re.DOTALL)
        return [match.strip() for match in matches] if matches else []

    def _llm_query(self, prompt: str, model: Optional[str] = None, parent_call_id: Optional[str] = None) -> str:
        """
        Execute a sub-LLM query with caching, metrics, and recursion tracking.

        Args:
            prompt: Prompt for the sub-call
            model: Model to use (defaults to self.sub_model)
            parent_call_id: ID of parent call for tracking

        Returns:
            LLM response
        """
        model = model or self.sub_model
        call_id = str(uuid4())

        # Check budget limits
        if self.metrics.check_budget_exceeded(self.max_cost, self.max_tokens):
            raise RuntimeError(
                f"Budget exceeded! Cost: ${self.metrics.total_cost:.4f}, "
                f"Tokens: {self.metrics.total_tokens}"
            )

        # Check cache
        if self.enable_cache:
            cached_response = self.cache.get(prompt, model)
            if cached_response is not None:
                self.logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return cached_response

        # Make API call
        self.logger.debug(f"Sub-call (depth={self.recursion_depth}): {prompt[:100]}...")

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )

            result = response.choices[0].message.content.strip()
            duration = time.time() - start_time

            # Record metrics
            self.metrics.record_call(
                model=model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                is_sub_call=True,
                depth=self.recursion_depth,
                duration=duration,
                call_id=call_id,
                parent_id=parent_call_id or self.current_call_id
            )

            # Update cache
            if self.enable_cache:
                self.cache.put(prompt, model, result)

            return result

        except Exception as e:
            self.logger.error(f"Sub-call failed: {str(e)}")
            raise

    def _final(self, answer: Any):
        """Set the final answer."""
        self.final_answer = answer
        self.logger.info(f"Final answer set: {str(answer)[:100]}...")

    def _final_var(self, var_name: str):
        """Set the final answer from a REPL variable."""
        if var_name in self.repl_globals:
            self.final_answer = self.repl_globals[var_name]
            self.logger.info(f"Final answer set from variable '{var_name}'")
        else:
            raise ValueError(f"Variable '{var_name}' not found in REPL.")

    def _setup_repl_globals(self, context: str) -> Dict[str, Any]:
        """
        Setup REPL globals with context, helpers, and API functions.

        Args:
            context: The long input context

        Returns:
            Dictionary of REPL globals
        """
        # Core variables and functions
        globals_dict = {
            'context': context,
            'llm_query': self._llm_query,
            'FINAL': self._final,
            'FINAL_VAR': self._final_var,
        }

        # Add helper functions
        globals_dict.update({
            # Text processing
            'chunk_text': self.text_processor.chunk_text,
            'chunk_by_tokens': self.text_processor.chunk_by_tokens,
            'smart_truncate': self.text_processor.smart_truncate,

            # Search
            'regex_search': self.search_helper.regex_search,
            'find_sections': self.search_helper.find_sections,
            'keyword_filter': self.search_helper.keyword_filter,

            # Aggregation
            'aggregate_results': self.aggregation_helper.aggregate_results,
            'count_frequencies': self.aggregation_helper.count_frequencies,
            'merge_dicts': self.aggregation_helper.merge_dicts,

            # Verification
            'verify_answer': lambda answer, prompt: self.verification_helper.verify_answer(
                answer, prompt, self._llm_query
            ),
            'consensus_check': lambda question, n=3: self.verification_helper.consensus_check(
                question, self._llm_query, n
            ),

            # Recursion patterns
            'recursive_split': self.recursion_helper.recursive_split,
            'map_reduce': self.recursion_helper.map_reduce,
        })

        # Apply security restrictions if enabled
        if self.enable_security:
            globals_dict = RestrictedGlobals.create_safe_globals(globals_dict)

        return globals_dict

    def _get_system_prompt(self) -> str:
        """
        Generate comprehensive system prompt for the RLM.

        Returns:
            System prompt string
        """
        return """
You are a Recursive Language Model (RLM). You process long contexts by writing Python code in a REPL environment.
The full context is available as the variable 'context' (a string, potentially very long—do NOT print or load it all at once).

## Core Guidelines:
- Write code inside ```repl ... ``` blocks
- Use code for syntactic tasks (slicing, regex, counting)
- Use llm_query(prompt) for semantic sub-tasks on small snippets
- Decompose complex tasks recursively
- Print intermediate results for observation
- Call FINAL(answer) or FINAL_VAR(var_name) to output final answer
- If code fails, revise in the next iteration

## Available Helper Functions:

### Text Processing:
- chunk_text(text, chunk_size=2000, overlap=200, preserve_paragraphs=False)
- chunk_by_tokens(text, max_tokens=1000, overlap_tokens=100)
- smart_truncate(text, max_length, suffix="...")

### Search & Filtering:
- regex_search(pattern, text, max_matches=None, return_positions=False)
- find_sections(text, section_pattern=r'^#+\s+(.+)$', include_content=True)
- keyword_filter(text, keywords, context_chars=200, case_sensitive=False)

### Aggregation:
- aggregate_results(results, method='join', separator='\n', filter_empty=True)
  Methods: 'join', 'sum', 'count', 'list', 'dict'
- count_frequencies(items)
- merge_dicts(dicts, merge_strategy='sum')
  Strategies: 'sum', 'last', 'first', 'list'

### Verification:
- verify_answer(answer, verification_prompt)
- consensus_check(question, num_attempts=3)

### Recursion Patterns:
- recursive_split(text, condition_fn, split_fn, max_depth=10)
- map_reduce(items, map_fn, reduce_fn, parallel=False)

## Emergent Behavior Patterns to Emulate:

1. **Filtering + Probing**: Use regex_search/keyword_filter to find candidates, then llm_query to verify
   Example: Needle-in-haystack tasks

2. **Recursive Chunking**: Use chunk_text or recursive_split, process each chunk with llm_query
   Example: Long-context QA, classification over lists

3. **Classification + Aggregation**: Chunk, classify each with llm_query, aggregate with count_frequencies
   Example: Counting items with specific properties

4. **Self-Verification**: Extract answer, use verify_answer to cross-check
   Example: Ensure accuracy on critical tasks

5. **Map-Reduce Pattern**: Split into items, map llm_query over each, reduce results
   Example: Pairwise reasoning, batch processing

6. **Consensus Checking**: Use consensus_check for higher confidence
   Example: Ambiguous or high-stakes decisions

## Best Practices:
- Never load entire context into llm_query—always slice/filter first
- Use chunk_by_tokens when approaching context limits
- Prefer built-in helpers over reimplementing (e.g., use chunk_text not manual slicing)
- For dense tasks, embrace recursion; for sparse tasks, use targeted filtering
- Monitor intermediate outputs with print() to debug
- Use verify_answer or consensus_check for critical results

## Security Notes:
- Dangerous operations (file I/O, network, subprocess) are restricted
- Focus on text processing and LLM calls within the sandbox
"""

    def run(
        self,
        task: str,
        context: str,
        max_iterations: int = 50,
        verbose: bool = True
    ) -> Any:
        """
        Run the RLM on a task with long context.

        Args:
            task: The query or task description
            context: The long input string (can be millions of tokens)
            max_iterations: Safety limit to prevent infinite loops
            verbose: If True, print progress updates

        Returns:
            Final answer

        Raises:
            RuntimeError: If max iterations reached or budget exceeded
        """
        self.logger.info(f"Starting RLM task: {task}")
        self.logger.info(f"Context length: {len(context):,} characters")

        # Reset state
        self.final_answer = None
        self.recursion_depth = 0
        self.current_call_id = str(uuid4())

        # Setup REPL globals
        self.repl_globals = self._setup_repl_globals(context)

        # Initialize conversation
        system_prompt = self._get_system_prompt()
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}"}
        ]

        iteration = 0

        # Main iteration loop
        while self.final_answer is None and iteration < max_iterations:
            self.metrics.increment_iteration()
            iteration += 1

            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration}/{max_iterations}")
                print(f"{'='*60}")

            # Check budget
            if self.metrics.check_budget_exceeded(self.max_cost, self.max_tokens):
                raise RuntimeError(
                    f"Budget exceeded at iteration {iteration}! "
                    f"Cost: ${self.metrics.total_cost:.4f}, Tokens: {self.metrics.total_tokens}"
                )

            # Call root LLM
            start_time = time.time()

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.history,
                    max_tokens=2048,
                )

                assistant_content = response.choices[0].message.content
                duration = time.time() - start_time

                # Record metrics
                self.metrics.record_call(
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    is_sub_call=False,
                    depth=0,
                    duration=duration,
                    call_id=self.current_call_id
                )

            except Exception as e:
                self.logger.error(f"Root LLM call failed: {str(e)}")
                raise

            self.history.append({"role": "assistant", "content": assistant_content})

            if verbose:
                print(f"\nAssistant response preview:")
                print(assistant_content[:500] + ("..." if len(assistant_content) > 500 else ""))

            # Extract and execute code blocks
            code_blocks = self._extract_code(assistant_content)

            if not code_blocks:
                feedback = "No valid code block found. Please provide code in ```repl ... ```."
                self.history.append({"role": "system", "content": feedback})
                self.logger.warning("No code blocks found in response")
                continue

            # Execute all code blocks
            all_outputs = []

            for i, code in enumerate(code_blocks):
                if verbose:
                    print(f"\nExecuting code block {i+1}/{len(code_blocks)}:")
                    print(code[:300] + ("..." if len(code) > 300 else ""))

                # Execute code
                if self.enable_security:
                    success, output, error = self.execution_monitor.capture_execution(
                        code, self.repl_globals
                    )
                else:
                    # Simple execution without monitoring
                    import io
                    import sys

                    output_capture = io.StringIO()
                    original_stdout = sys.stdout
                    sys.stdout = output_capture

                    try:
                        exec(code, self.repl_globals)
                        output = output_capture.getvalue().strip()
                        error = ""
                        success = True
                    except Exception as e:
                        output = ""
                        error = str(e)
                        success = False
                    finally:
                        sys.stdout = original_stdout
                        output_capture.close()

                # Format feedback
                if success:
                    if output:
                        feedback = f"Code block {i+1} executed successfully. Output:\n{output}"
                    else:
                        feedback = f"Code block {i+1} executed successfully (no output)."
                    all_outputs.append(output)

                    if verbose:
                        print(f"✓ Success")
                        if output:
                            print(f"Output:\n{output[:200]}" + ("..." if len(output) > 200 else ""))
                else:
                    feedback = f"Code block {i+1} execution failed: {error}"
                    if verbose:
                        print(f"✗ Error: {error}")

                self.logger.debug(feedback)

            # Send aggregated feedback
            combined_feedback = "\n\n".join(
                [f"Code block {i+1}: {out}" for i, out in enumerate(all_outputs) if out]
            ) or "All code blocks executed successfully (no output)."

            self.history.append({"role": "system", "content": combined_feedback})

            # Check if final answer was set
            if self.final_answer is not None:
                if verbose:
                    print(f"\n{'='*60}")
                    print("✓ Final answer produced!")
                    print(f"{'='*60}")
                break

        # Finalize metrics
        self.metrics.finalize()

        # Check completion
        if self.final_answer is None:
            raise RuntimeError(f"Max iterations ({max_iterations}) reached without final answer.")

        self.logger.info("RLM task completed successfully")

        return self.final_answer

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get execution metrics summary.

        Returns:
            Dictionary of metrics
        """
        summary = self.metrics.get_summary()

        if self.enable_cache:
            summary['cache'] = self.cache.get_stats()

        return summary

    def print_metrics(self):
        """Print execution metrics."""
        self.metrics.print_summary()

        if self.enable_cache:
            self.cache.print_stats()

    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        self.metrics.export_to_json(filepath)

    def reset(self):
        """Reset RLM state for a new task."""
        self.final_answer = None
        self.repl_globals = {}
        self.history = []
        self.recursion_depth = 0
        self.current_call_id = None
        self.metrics = RLMMetrics()

        if self.enable_cache:
            self.cache.clear()

        self.logger.info("RLM state reset")
