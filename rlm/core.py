"""
Enhanced Recursive Language Model (RLM) implementation with multi-provider support.

Processes extremely long contexts through recursive decomposition and sub-calls.
Supports OpenAI, xAI Grok, and other LLM providers.
"""

import re
import time
import logging
from typing import Optional, Dict, Any, Callable, Union
from uuid import uuid4

from .metrics import RLMMetrics
from .cache import RLMCache
from .security import RestrictedGlobals, ExecutionMonitor
from .providers import LLMProvider, create_provider
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
    - Multi-provider support (OpenAI, xAI Grok, etc.)
    - Context window awareness
    - Detailed token tracking (cached, reasoning, etc.)
    - Metrics tracking
    - Sub-call caching
    - Security sandboxing
    - Advanced helpers
    - Budget controls
    - Improved error handling
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        sub_model: Optional[str] = None,
        provider: Union[str, LLMProvider, None] = None,
        context_window: Optional[int] = None,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: Optional[float] = 3600,
        max_cost: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_security: bool = True,
        log_level: str = "INFO",
        timeout: Optional[int] = None
    ):
        """
        Initialize the Enhanced RLM.

        Args:
            api_key: API key for the provider
            model: Main model for root LLM (e.g., 'gpt-4o', 'grok-4')
            sub_model: Optional cheaper model for sub-calls
            provider: Provider name ('openai', 'xai') or LLMProvider instance
            context_window: Maximum context window (default: 128k)
            enable_cache: Enable caching of sub-call results
            cache_size: Maximum number of cached entries
            cache_ttl: Cache time-to-live in seconds (None = no expiration)
            max_cost: Maximum total cost in USD (None = unlimited)
            max_tokens: Maximum total tokens (None = unlimited)
            enable_security: Enable security sandboxing for code execution
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            timeout: API request timeout in seconds
        """
        # Setup provider
        if isinstance(provider, LLMProvider):
            self.root_provider = provider
            self.model = model or provider.default_model
        else:
            # Auto-detect provider from model name or use provided provider name
            if provider is None:
                if model.startswith('grok'):
                    provider = 'xai'
                else:
                    provider = 'openai'

            # Create provider
            self.root_provider = create_provider(
                provider_name=provider,
                api_key=api_key,
                default_model=model,
                context_window=context_window,
                timeout=timeout
            )
            self.model = model

        # Sub-model provider (can be different)
        self.sub_model = sub_model or model
        if sub_model and sub_model != model:
            # Determine sub-provider
            sub_provider_name = 'xai' if sub_model.startswith('grok') else 'openai'
            self.sub_provider = create_provider(
                provider_name=sub_provider_name,
                api_key=api_key,
                default_model=sub_model,
                context_window=context_window,
                timeout=timeout
            )
        else:
            self.sub_provider = self.root_provider

        # Context window tracking
        self.context_window = context_window or self.root_provider.context_window

        # Metrics tracking with pricing provider
        self.metrics = RLMMetrics(pricing_provider=self.root_provider.get_pricing)
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
        Extract all code blocks from the LLM response.
        Accepts ```repl, ```python, or plain ``` code blocks.

        Args:
            text: LLM response text

        Returns:
            List of code blocks
        """
        matches = re.findall(r'```(?:repl|python)?\n(.*?)\n```', text, re.DOTALL)
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

        # Make API call using provider
        self.logger.debug(f"Sub-call (depth={self.recursion_depth}): {prompt[:100]}...")

        start_time = time.time()

        try:
            provider = self.sub_provider if model == self.sub_model else self.root_provider
            response = provider.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=1024
            )

            duration = time.time() - start_time

            # Record detailed metrics
            usage = response.usage
            self.metrics.record_call(
                model=model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                is_sub_call=True,
                depth=self.recursion_depth,
                duration=duration,
                call_id=call_id,
                parent_id=parent_call_id or self.current_call_id,
                # Detailed token counts
                prompt_cached_tokens=usage.prompt_cached_tokens,
                completion_reasoning_tokens=usage.completion_reasoning_tokens,
                prompt_text_tokens=usage.prompt_text_tokens,
                prompt_image_tokens=usage.prompt_image_tokens,
                prompt_audio_tokens=usage.prompt_audio_tokens,
                completion_audio_tokens=usage.completion_audio_tokens,
                completion_accepted_prediction_tokens=usage.completion_accepted_prediction_tokens,
                completion_rejected_prediction_tokens=usage.completion_rejected_prediction_tokens,
                num_sources_used=usage.num_sources_used
            )

            # Update cache
            if self.enable_cache:
                self.cache.put(prompt, model, response.content)

            return response.content

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
            'find_tags': self.search_helper.find_tags,
            'extract_between_markers': self.search_helper.extract_between_markers,
            'count_occurrences': self.search_helper.count_occurrences,

            # Aggregation
            'aggregate_results': self.aggregation_helper.aggregate_results,
            'count_frequencies': self.aggregation_helper.count_frequencies,
            'merge_dicts': self.aggregation_helper.merge_dicts,

            # Verification (returns tuple: (is_valid, explanation))
            'verify_answer': lambda answer, prompt="Verify if the following answer is correct based on the context": self.verification_helper.verify_answer(
                answer, prompt, self._llm_query
            ),
            # Consensus check (returns tuple: (answer, confidence))
            'consensus_check': lambda question, num_attempts=3: self.verification_helper.consensus_check(
                question, self._llm_query, num_attempts
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
            System prompt string with context window info
        """
        return f"""
You are a Recursive Language Model (RLM). You process long contexts by writing Python code in a REPL environment.
The full context is available as the variable 'context' (a string, potentially very long—do NOT print or load it all at once).

## System Information:
- Model: {self.model}
- Context Window: {self.context_window:,} tokens (~{self.context_window * 4:,} characters)
- Sub-model for queries: {self.sub_model}

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
- find_tags(text, tag_pattern, extract_content=True, content_delimiter=None, max_results=None)
  Fast string-based tag finder (NOT regex). Use this for finding markers like "[KEY POINT]".
  Returns: List of dicts with keys 'tag', 'content', 'start', 'end'.
  Example: find_tags(text, "[KEY POINT]") finds all [KEY POINT] tags
  More reliable than regex for simple tag matching. PREFER THIS over regex_search for tags.
- extract_between_markers(text, start_marker, end_marker, include_markers=False, max_results=None)
  Extract content between paired markers (e.g., "[DOC START]" and "[DOC END]").
  Returns: List of dicts with keys 'content', 'start', 'end'.
- count_occurrences(text, pattern, overlapping=False)
  Fast string-based counting (NOT regex). Much faster than regex for simple patterns.
  Returns: Integer count of occurrences.
- regex_search(pattern, text, max_matches=None, return_positions=False, flags=0)
  Regex-based search with optional flags (re.MULTILINE, re.DOTALL, re.IGNORECASE).
  When return_positions=True: Returns List of dicts with keys 'match', 'start', 'end'.
  Example: {{'match': 'found text', 'start': 100, 'end': 110}}
  When return_positions=False: Returns List of matched strings.
  For simple tag matching, PREFER find_tags() instead for better reliability.
- find_sections(text, section_pattern=r'^#+\s+(.+)$', include_content=True)
- keyword_filter(text, keywords, context_chars=200, case_sensitive=False)
  Returns: List of (snippet, position) tuples. Access snippet with item[0], position with item[1].

### Aggregation:
- aggregate_results(results, method='join', separator='\\n', filter_empty=True)
  Methods: 'join', 'sum', 'count', 'list', 'dict'
- count_frequencies(items)
- merge_dicts(dicts, merge_strategy='sum')
  Strategies: 'sum', 'last', 'first', 'list'

### Verification:
- verify_answer(answer, verification_prompt) -> Returns: (is_valid: bool, explanation: str)
- consensus_check(question, num_attempts=3) -> Returns: (consensus_answer: str, confidence: float)

⚠️ CRITICAL WARNING about verify_answer and consensus_check:
These functions have NO ACCESS to the original context variable!
They query the LLM with ONLY the prompt/question you provide.
NEVER use them to find or discover information - they will hallucinate.
ONLY use them to verify data you have ALREADY extracted from context.
If you already found data via regex/search, trust that data - don't second-guess it with consensus_check.

### Recursion Patterns:
- recursive_split(text, condition_fn, split_fn, max_depth=10)
- map_reduce(items, map_fn, reduce_fn, parallel=False)

## Emergent Behavior Patterns to Emulate:

1. **Tag-Based Extraction**: Use find_tags() to find structured markers (e.g., "[KEY POINT]"), then process content
2. **Filtering + Probing**: Use find_tags/keyword_filter to find candidates, then llm_query to verify
3. **Recursive Chunking**: Use chunk_text or recursive_split, process each chunk with llm_query
4. **Classification + Aggregation**: Chunk, classify each with llm_query, aggregate with count_frequencies
5. **Self-Verification**: Extract answer, use verify_answer to cross-check
6. **Map-Reduce Pattern**: Split into items, map llm_query over each, reduce results
7. **Consensus Checking**: Use consensus_check for higher confidence
8. **Marker Extraction**: Use extract_between_markers() to extract sections between paired delimiters

## Best Practices:
- Never load entire context into llm_query—always slice/filter first
- Use chunk_by_tokens when approaching context limits
- For dense tasks, embrace recursion; for sparse tasks, use targeted filtering
- Monitor intermediate outputs with print() to debug
- Use verify_answer or consensus_check for critical results

## CRITICAL - Anti-Hallucination Rules:
- ALWAYS use data you actually extracted from the context. NEVER fabricate or invent data.
- If you found data via regex/search, use THAT EXACT DATA in your final answer.
- Store extracted values in variables immediately: `extracted_value = match['match']`
- Before calling FINAL(), verify you're returning actual extracted data, not placeholder/example values.
- If a search found results, those results are your answer—don't make up different ones.
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
        self.logger.info(f"Context length: {len(context):,} characters (~{len(context)//4:,} tokens)")
        self.logger.info(f"Context window: {self.context_window:,} tokens")

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
        consecutive_errors = 0
        max_consecutive_errors = 5  # Prevent infinite token burning on repeated errors

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

            # Call root LLM using provider
            start_time = time.time()

            try:
                response = self.root_provider.chat_completion(
                    messages=self.history,
                    model=self.model,
                    max_tokens=2048
                )

                duration = time.time() - start_time

                assistant_content = response.content

                # Record detailed metrics
                usage = response.usage
                self.metrics.record_call(
                    model=self.model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    is_sub_call=False,
                    depth=0,
                    duration=duration,
                    call_id=self.current_call_id,
                    # Detailed token counts
                    prompt_cached_tokens=usage.prompt_cached_tokens,
                    completion_reasoning_tokens=usage.completion_reasoning_tokens,
                    prompt_text_tokens=usage.prompt_text_tokens,
                    prompt_image_tokens=usage.prompt_image_tokens,
                    prompt_audio_tokens=usage.prompt_audio_tokens,
                    completion_audio_tokens=usage.completion_audio_tokens,
                    completion_accepted_prediction_tokens=usage.completion_accepted_prediction_tokens,
                    completion_rejected_prediction_tokens=usage.completion_rejected_prediction_tokens,
                    num_sources_used=usage.num_sources_used
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
            iteration_had_error = False

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
                    iteration_had_error = True
                    if verbose:
                        print(f"✗ Error: {error}")

                self.logger.debug(feedback)

            # Send aggregated feedback
            combined_feedback = "\n\n".join(
                [f"Code block {i+1}: {out}" for i, out in enumerate(all_outputs) if out]
            ) or "All code blocks executed successfully (no output)."

            self.history.append({"role": "system", "content": combined_feedback})

            # Track consecutive errors
            if iteration_had_error:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"Terminated after {consecutive_errors} consecutive REPL errors. "
                        f"The model appears stuck in an error loop. "
                        f"Cost so far: ${self.metrics.total_cost:.4f}, Tokens: {self.metrics.total_tokens}"
                    )
            else:
                consecutive_errors = 0  # Reset on successful execution

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
        Get execution metrics summary including context window info.

        Returns:
            Dictionary of metrics
        """
        summary = self.metrics.get_summary()

        # Add provider and context window info
        summary['provider'] = {
            'root_model': self.model,
            'sub_model': self.sub_model,
            'context_window': self.context_window,
        }

        if self.enable_cache:
            summary['cache'] = self.cache.get_stats()

        return summary

    def print_metrics(self):
        """Print execution metrics."""
        self.metrics.print_summary()

        # Print provider info
        print(f"Provider Info:")
        print(f"  Root model: {self.model}")
        print(f"  Sub model: {self.sub_model}")
        print(f"  Context window: {self.context_window:,} tokens")

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
        self.metrics = RLMMetrics(pricing_provider=self.root_provider.get_pricing)

        if self.enable_cache:
            self.cache.clear()

        self.logger.info("RLM state reset")
