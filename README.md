# Recursive Language Model (RLM) Framework

A Python framework for processing extremely long contexts (10M+ tokens) using recursive decomposition and sub-calls to language models.

## Overview

Recursive Language Models (RLMs) enable LLMs to handle inputs far beyond their context window limits by:

- **Breaking down** complex tasks into smaller, manageable sub-problems
- **Invoking sub-LLMs** recursively on targeted snippets
- **Aggregating results** back up through a tree-like call structure
- **Operating in a REPL environment** where context is accessed programmatically, not loaded into neural context

This implementation supports multiple LLM providers including OpenAI (GPT-4o, GPT-5) and xAI (Grok), with an extensible provider architecture.

## Key Features

### 🚀 Core Capabilities

- **Process massive inputs**: Handle 10M+ tokens, 100x beyond typical context windows
- **Recursive sub-calls**: Automatic decomposition with nested LLM invocations
- **REPL-based execution**: Generate and execute Python code in a persistent environment
- **Model flexibility**: Use different models for root and sub-calls (e.g., GPT-4o + GPT-4o-mini)
- **Comprehensive testing**: Built-in test suite for needle-in-haystack, reasoning, and summarization tasks

### 📊 Metrics & Monitoring

- **Comprehensive tracking**: Tokens, costs, recursion depth, call counts
- **Real-time budget controls**: Set max cost/token limits
- **Detailed analytics**: Export metrics to JSON for analysis
- **Cache statistics**: Monitor cache hit rates and efficiency

### ⚡ Performance Optimization

- **LRU caching**: Avoid redundant sub-calls (configurable size & TTL)
- **Model tiering**: Use cheaper models for sub-calls
- **Smart chunking**: Token-aware and paragraph-preserving strategies
- **Parallel processing**: Map-reduce patterns with optional parallelization

### 🔒 Security

- **Sandboxed execution**: Restricted globals prevent dangerous operations
- **Code safety checks**: Block forbidden patterns (file I/O, network, subprocess)
- **Resource limits**: Execution timeouts and output size constraints

### 🛠️ Advanced Helpers

- **Text processing**: Smart chunking, token-based splitting, truncation
- **Search & filtering**: Regex, keyword search, section extraction
- **Aggregation**: Multiple strategies (sum, join, count, dict)
- **Verification**: Answer validation, consensus checking
- **Recursion patterns**: Recursive split, map-reduce

## Usage at a Glance

**CLI (easiest):**
```bash
uv run python main.py run --task "Your question" --context-file data.txt
```

**Python API:**
```python
from rlm import RecursiveLanguageModel
rlm = RecursiveLanguageModel(api_key="...", model="grok-4-1-fast-reasoning")
result = rlm.run(task="Your question", context="Your long text...")
```

**See [CLI Usage](#cli-usage) for detailed examples with different models and options.**

## Repository Structure

```
Rlm/
├── rlm/                          # Main package
│   ├── core.py                   # RecursiveLanguageModel implementation
│   ├── providers.py              # Multi-provider support (OpenAI, xAI)
│   ├── metrics.py                # Token and cost tracking
│   ├── helpers.py                # Advanced utility functions
│   ├── security.py               # Sandboxed execution
│   ├── cache.py                  # LRU caching system
│   └── __init__.py               # Package exports
├── examples/                     # Usage examples
│   ├── quickstart_grok.py        # Minimal Grok example
│   ├── quickstart_gpt5.py        # Minimal GPT-5 example
│   ├── basic_usage.py            # Needle-in-haystack pattern
│   ├── classification_example.py # Classification and aggregation
│   ├── verification_example.py   # Verification pattern
│   ├── long_output_example.py    # Long output generation
│   ├── advanced_patterns.py      # Map-reduce pattern
│   ├── grok_basic_example.py     # Grok integration
│   ├── grok_reasoning_example.py # Grok reasoning metrics
│   └── multi_provider_example.py # Cross-provider comparison
├── tests/                        # Comprehensive test suite
│   ├── test_helpers.py           # Helper function tests
│   ├── test_cache.py             # Caching tests
│   ├── test_metrics.py           # Metrics tests
│   ├── test_mock.py              # Mock/stub tests
│   ├── test_rlm_comprehensive.py # Long-context integration tests (256k needle, multi-needle, reasoning, summarization)
│   └── test_data_generator.py    # Test data generation (technical reports, edge cases)
├── main.py                       # CLI entry point
├── Makefile                      # Development commands (uv-based)
├── pyproject.toml                # Package metadata
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment configuration template
├── PROJECT_STRUCTURE.md          # Detailed structure documentation
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer (recommended)
- OpenAI API key or xAI API key

### Install with uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (recommended - uses uv.lock for reproducibility)
uv sync

# Or install in editable mode
uv pip install -e .
```

### Or install with pip

```bash
# Install dependencies and package in editable mode
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
pip install -e .
```

### Setup Environment

Create a `.env` file or set environment variables:

```bash
# For OpenAI models
export OPENAI_API_KEY="your-openai-key"

# For xAI Grok models
export XAI_API_KEY="your-xai-key"
```

## Quick Start

```python
from rlm import RecursiveLanguageModel

# Initialize RLM
rlm = RecursiveLanguageModel(
    api_key="your-api-key",
    model="gpt-4o",           # Root model
    sub_model="gpt-4o-mini",  # Cheaper model for sub-calls
    enable_cache=True,        # Cache sub-call results
    max_cost=1.0              # Budget limit: $1
)

# Create a long context
context = "..." # Your long document (can be millions of characters)

# Define task
task = "Find the magic number mentioned in the context."

# Run RLM
result = rlm.run(task=task, context=context, verbose=True)
print(f"Result: {result}")

# View metrics
rlm.print_metrics()
```

## CLI Usage

The `main.py` CLI provides an easy way to run RLM tasks without writing Python code.

### Basic Commands

```bash
# Show available models and configuration
uv run python main.py info

# Run a task with a text file
uv run python main.py run --task "Your question" --context-file document.txt

# Run a task with direct text
uv run python main.py run --task "Your question" --context "Your text here"

# Run tests
uv run python main.py test --quick
```

### Using xAI Grok Models (Default)

```bash
# Basic usage with Grok (uses grok-4-1-fast-reasoning by default)
uv run python main.py run \
  --task "Find the secret code mentioned in the document" \
  --context-file data.txt

# Specify a different Grok model
uv run python main.py run \
  --task "Summarize the key findings" \
  --context-file research_paper.txt \
  --model grok-4 \
  --provider xai

# With custom settings
uv run python main.py run \
  --task "Extract all dates and events" \
  --context-file historical_records.txt \
  --model grok-4-1-fast-reasoning \
  --max-cost 2.0 \
  --max-iterations 30 \
  --output results.json
```

### Using OpenAI Models

```bash
# Using GPT-4o
uv run python main.py run \
  --task "Analyze sentiment across all reviews" \
  --context-file reviews.txt \
  --model gpt-4o \
  --provider openai

# Using GPT-5 mini (fast and cheap)
uv run python main.py run \
  --task "Find references to project Apollo" \
  --context-file transcripts.txt \
  --model gpt-5-mini \
  --provider openai \
  --max-cost 1.0
```

### Direct Text Input

```bash
# Short context via command line
uv run python main.py run \
  --task "What is the main topic?" \
  --context "Artificial intelligence is transforming how we process information..."

# Multi-line text (using quotes)
uv run python main.py run \
  --task "Count how many people are mentioned" \
  --context "John works in marketing. Sarah leads engineering. Bob manages operations."
```

### Advanced Options

```bash
# Save results to JSON file
uv run python main.py run \
  --task "Summarize findings" \
  --context-file report.txt \
  --output summary.json

# Quiet mode (less verbose output)
uv run python main.py run \
  --task "Find the error code" \
  --context-file logs.txt \
  --quiet

# Disable caching
uv run python main.py run \
  --task "Process each item uniquely" \
  --context-file items.txt \
  --no-cache

# Set custom API key (instead of env variable)
uv run python main.py run \
  --task "Your task" \
  --context-file data.txt \
  --api-key "your-api-key-here"
```

### Available Models

#### xAI Grok Models
- `grok-4` - Standard Grok 4 model (128k context)
- `grok-4-1-fast-reasoning` - Fast reasoning variant (recommended, cheaper)
- `grok-4-1-fast-non-reasoning` - Non-reasoning variant
- `grok-4-fast-reasoning` - Fast reasoning
- `grok-4-fast-non-reasoning` - Fast non-reasoning
- `grok-beta` - Beta version

#### OpenAI Models
- `gpt-5-mini` - GPT-5 mini (fast and cheap, recommended for testing)
- `gpt-5-nano` - GPT-5 nano (ultra-fast)
- `gpt-4.1` - GPT-4.1
- `gpt-4o` - GPT-4 optimized

### CLI Options Reference

**Run Command Options:**
```
--task, -t          Task description/question (required)
--context, -c       Direct text input
--context-file, -f  Path to text file
--provider, -p      LLM provider: 'xai' or 'openai' (default: xai)
--model, -m         Model name (default: auto-detected)
--api-key           API key (or use environment variable)
--max-cost          Maximum cost in USD (default: 5.0)
--max-iterations    Maximum iterations (default: 50)
--no-cache          Disable caching
--output, -o        Save results to JSON file
--quiet, -q         Reduce output verbosity
```

### Test Command

```bash
# Quick sanity check
uv run python main.py test --quick --model grok-4-1-fast-reasoning

# Run comprehensive test suite
uv run python main.py test \
  --suite comprehensive \
  --model grok-4-1-fast-reasoning \
  --no-256k

# Run unit tests
uv run python main.py test --suite unit

# Save test results
uv run python main.py test \
  --suite comprehensive \
  --model grok-4-1-fast-reasoning \
  --output test_results.json \
  --max-cost 2.0
```

### Generate Test Data

```bash
# Generate all test types
uv run python main.py generate --type all

# Generate needle-in-haystack test
uv run python main.py generate \
  --type needle \
  --tokens 256 \
  --position middle \
  --output-dir tests/data

# Generate multi-needle test
uv run python main.py generate \
  --type multi \
  --num-needles 5

# Generate reasoning test
uv run python main.py generate \
  --type reasoning \
  --complexity medium
```

### Real-World Examples

**Example 1: Process a large log file**
```bash
uv run python main.py run \
  --task "Find all ERROR entries and summarize the most common issues" \
  --context-file application.log \
  --model grok-4-1-fast-reasoning \
  --max-cost 3.0 \
  --output error_summary.json
```

**Example 2: Analyze research papers**
```bash
uv run python main.py run \
  --task "Extract methodology, key findings, and limitations" \
  --context-file paper.txt \
  --model gpt-4o \
  --provider openai \
  --max-iterations 40
```

**Example 3: Process customer feedback**
```bash
uv run python main.py run \
  --task "Categorize feedback by topic and sentiment, then count frequencies" \
  --context-file customer_reviews.txt \
  --model grok-4-1-fast-reasoning \
  --max-cost 2.0
```

**Example 4: Code review**
```bash
uv run python main.py run \
  --task "Find potential security issues and performance bottlenecks" \
  --context-file codebase_dump.txt \
  --model gpt-4o \
  --provider openai
```

## Architecture

### How It Works

1. **Initialization**: Context is loaded into REPL as a variable (not into neural context)
2. **Code Generation**: Root LLM generates Python code to process the task
3. **Execution**: Code runs in REPL, can inspect/slice context programmatically
4. **Sub-calls**: Code invokes `llm_query()` on small snippets for semantic tasks
5. **Recursion**: Sub-calls can make their own sub-calls (tree structure)
6. **Aggregation**: Results bubble up and are combined
7. **Iteration**: Process repeats until `FINAL()` is called

```
Root LLM → Generate Code → Execute in REPL
              ↓
          llm_query(snippet1) → Sub-LLM → Result1
          llm_query(snippet2) → Sub-LLM → Result2
              ↓
          Aggregate Results → Next Iteration
              ↓
          FINAL(answer)
```

### Components

#### 1. Core (`rlm/core.py`)

Main RLM implementation with:
- REPL management
- LLM API calls
- Iteration loop
- Code execution
- Final answer handling

#### 2. Metrics (`rlm/metrics.py`)

Tracks:
- Token usage (prompt + completion)
- Costs by model
- Recursion depth
- Call counts
- Execution time
- Per-call details

#### 3. Cache (`rlm/cache.py`)

LRU cache for sub-calls:
- Hash-based lookup (prompt + model)
- Configurable size and TTL
- Hit/miss tracking
- Export capabilities

#### 4. Security (`rlm/security.py`)

Sandboxing for code execution:
- Restricted builtins (no `eval`, `open`, `__import__`, etc.)
- Blocked dangerous modules (os, sys, subprocess, etc.)
- Code pattern checking
- Execution monitoring

#### 5. Helpers (`rlm/helpers.py`)

Advanced utilities:
- **TextProcessor**: Chunking, token-based splitting
- **SearchHelper**: Regex, keyword filtering, section extraction
- **AggregationHelper**: Multiple aggregation strategies
- **VerificationHelper**: Answer validation, consensus
- **RecursionHelper**: Recursive patterns, map-reduce

## Usage Patterns

The RLM naturally develops these emergent behaviors:

### 1. Filtering + Probing

**Use case**: Needle-in-haystack tasks

```python
task = "Find the magic number in the context."

# RLM will:
# 1. Use regex_search() to find candidates
# 2. Use llm_query() on each to verify
# 3. Return the verified answer
```

**Example**: `examples/basic_usage.py`

### 2. Recursive Chunking + Classification

**Use case**: Long lists, classification tasks

```python
task = "Count how many items are fruits vs vegetables."

# RLM will:
# 1. Split context into lines/chunks
# 2. Call llm_query() on each chunk for classification
# 3. Use count_frequencies() to aggregate
```

**Example**: `examples/classification_example.py`

### 3. Self-Verification

**Use case**: Critical facts extraction

```python
task = "Extract key facts and verify them."

# RLM will:
# 1. Extract facts from chunks
# 2. Use verify_answer() to cross-check
# 3. Return only verified facts
```

**Example**: `examples/verification_example.py`

### 4. Long Output Generation

**Use case**: Comprehensive summaries, reports

```python
task = "Generate detailed summary of all topics."

# RLM will:
# 1. Use find_sections() to identify topics
# 2. Call llm_query() for each topic's summary
# 3. Aggregate into final document
```

**Example**: `examples/long_output_example.py`

### 5. Map-Reduce Pattern

**Use case**: Batch processing, sentiment analysis

```python
task = "Analyze sentiment of all reviews."

# RLM will:
# 1. Use map_reduce() to process reviews
# 2. Map: llm_query() classifies each review
# 3. Reduce: Aggregate results
```

**Example**: `examples/advanced_patterns.py`

## Available Helper Functions

When generating code, the RLM has access to these helpers:

### Text Processing

```python
# Chunk text with overlap
chunks = chunk_text(text, chunk_size=2000, overlap=200, preserve_paragraphs=False)

# Token-based chunking (more accurate for LLM limits)
chunks = chunk_by_tokens(text, max_tokens=1000, overlap_tokens=100)

# Smart truncation at word boundaries
truncated = smart_truncate(text, max_length=100, suffix="...")
```

### Search & Filtering

```python
# Regex search with limits
matches = regex_search(pattern, text, max_matches=10, return_positions=False)

# Find markdown sections
sections = find_sections(text, section_pattern=r'^#+\s+(.+)$', include_content=True)

# Keyword filtering with context
snippets = keyword_filter(text, keywords=['important', 'critical'], context_chars=200)
```

### Aggregation

```python
# Aggregate results
result = aggregate_results(results, method='join', separator='\n', filter_empty=True)
# Methods: 'join', 'sum', 'count', 'list', 'dict'

# Count frequencies
freq = count_frequencies(['apple', 'banana', 'apple'])  # {'apple': 2, 'banana': 1}

# Merge dictionaries
merged = merge_dicts([dict1, dict2], merge_strategy='sum')
# Strategies: 'sum', 'last', 'first', 'list'
```

### Verification

```python
# Verify an answer
is_valid, explanation = verify_answer(answer, verification_prompt)

# Consensus check (multiple attempts)
consensus_answer, confidence = consensus_check(question, num_attempts=3)
```

### Recursion Patterns

```python
# Recursive split until condition met
chunks = recursive_split(
    text,
    condition=lambda t: len(t) < 1000,  # Stop when small enough
    split_fn=lambda t: chunk_text(t, chunk_size=5000),
    max_depth=10
)

# Map-reduce pattern
result = map_reduce(
    items,
    map_fn=lambda item: llm_query(f"Process: {item}"),
    reduce_fn=lambda results: aggregate_results(results, method='join'),
    parallel=False
)
```

## Configuration Options

### RLM Initialization

```python
rlm = RecursiveLanguageModel(
    api_key="...",              # OpenAI API key
    model="gpt-4o",             # Root model
    sub_model="gpt-4o-mini",    # Sub-call model (defaults to root model)
    enable_cache=True,          # Enable caching
    cache_size=1000,            # Max cached entries
    cache_ttl=3600,             # Cache TTL in seconds (None = no expiration)
    max_cost=None,              # Max total cost in USD (None = unlimited)
    max_tokens=None,            # Max total tokens (None = unlimited)
    enable_security=True,       # Enable sandboxing
    log_level="INFO"            # Logging level
)
```

### Run Options

```python
result = rlm.run(
    task="...",                 # Task description
    context="...",              # Long input context
    max_iterations=50,          # Safety limit
    verbose=True                # Print progress
)
```

## Metrics & Export

### View Metrics

```python
# Print summary
rlm.print_metrics()

# Get metrics dict
metrics = rlm.get_metrics_summary()
print(metrics['cost']['total_usd'])
print(metrics['tokens']['total'])
print(metrics['cache']['hit_rate_percent'])
```

### Export Metrics

```python
# Export to JSON
rlm.export_metrics("metrics.json")

# Exported data includes:
# - Summary: duration, iterations, calls, tokens, cost, efficiency
# - Call history: per-call details, timestamps, recursion depth
# - Cache stats (if enabled)
```

## Best Practices

### 1. Choose Appropriate Models

- Use powerful model (GPT-4o) for root LLM (complex reasoning)
- Use cheaper model (GPT-4o-mini) for sub-calls (simple tasks)

### 2. Set Budget Limits

```python
rlm = RecursiveLanguageModel(
    ...,
    max_cost=5.0,        # Stop if cost exceeds $5
    max_tokens=1_000_000 # Stop if tokens exceed 1M
)
```

### 3. Enable Caching

Essential for repeated sub-calls (e.g., classification of duplicate items):

```python
rlm = RecursiveLanguageModel(
    ...,
    enable_cache=True,
    cache_size=1000,
    cache_ttl=3600  # 1 hour
)
```

### 4. Use Token-Based Chunking

More accurate than character-based:

```python
# In generated code:
chunks = chunk_by_tokens(context, max_tokens=1000)
```

### 5. Leverage Verification for Critical Tasks

```python
# In generated code:
answer = llm_query("Extract the fact")
is_valid, explanation = verify_answer(answer, "Cross-check this fact")
if is_valid:
    FINAL(answer)
```

### 6. Monitor Metrics

Always check metrics after runs to optimize:

```python
rlm.print_metrics()
# Look for:
# - High sub-call ratio (good for dense tasks)
# - Cache hit rate (should be high for repeated items)
# - Cost per call (optimize model choices)
```

## Limitations

1. **Over-Recursion Risk**: Some models may make excessive sub-calls, inflating costs
   - **Mitigation**: Set `max_cost` and `max_tokens` limits

2. **Code Generation Dependency**: Relies on LLM's coding ability
   - **Mitigation**: Use stronger root model (GPT-4o)

3. **Execution Time**: Many sub-calls can be slow
   - **Mitigation**: Use caching, cheaper sub-models, parallel processing

4. **Security**: Code execution has inherent risks
   - **Mitigation**: Sandboxing is enabled by default

## Examples

### CLI Examples

For the easiest way to get started without writing code, see the [CLI Usage](#cli-usage) section above. The CLI allows you to run tasks directly from the command line:

```bash
# Quick example with CLI
uv run python main.py run \
  --task "Find all mentions of 'quantum computing'" \
  --context-file research.txt \
  --model grok-4-1-fast-reasoning
```

### Quickstart Examples

For the simplest possible Python usage, start with the quickstart examples:

```bash
# Grok quickstart (minimal example)
uv run python examples/quickstart_grok.py

# GPT-5 quickstart (minimal example)
uv run python examples/quickstart_gpt5.py
```

### Full Examples

Run the comprehensive examples to see different patterns:

```bash
# Basic needle-in-haystack
uv run python examples/basic_usage.py

# Classification and aggregation
uv run python examples/classification_example.py

# Verification pattern
uv run python examples/verification_example.py

# Long output generation
uv run python examples/long_output_example.py

# Advanced map-reduce
uv run python examples/advanced_patterns.py

# Grok-specific examples
uv run python examples/grok_basic_example.py
uv run python examples/grok_reasoning_example.py

# Multi-provider comparison
uv run python examples/multi_provider_example.py
```

### Using Make

You can also use the Makefile for convenience:

```bash
# Run all examples
make examples

# Run quickstart examples only
make quickstart

# Run unit tests
make test

# Run with coverage
make test-coverage

# Run comprehensive integration tests
uv run python tests/test_rlm_comprehensive.py
```

## Testing

The RLM framework includes a comprehensive test suite to evaluate long-context capabilities across multiple dimensions.

### Test Suite Overview

Located in `tests/test_rlm_comprehensive.py`, the test suite includes:

#### 1. **Needle in Haystack (256k tokens)**
- Tests retrieval of specific information buried in massive contexts
- Evaluates precision at different positions (start, middle, end)
- **Use case**: Finding specific facts in huge documents

#### 2. **Multi-Needle Retrieval**
- Tests extraction of multiple distributed facts
- Evaluates recall and completeness
- **Use case**: Extracting all key data points from long reports

#### 3. **Reasoning Tests**
- Tests multi-hop reasoning over distributed information
- Evaluates ability to connect facts across long contexts
- **Complexity levels**: Simple, medium, complex
- **Use case**: Answering questions requiring synthesis

#### 4. **Edge Case Handling**
- Tests contradictory information resolution
- Tests handling of repeated vs. unique information
- Tests numeric precision in calculations
- **Use case**: Robust processing of real-world messy data

#### 5. **Document Summarization** (New!)
- Tests ability to summarize 25k-90k token documents
- Evaluates key point coverage (70% threshold) and compression ratio (≤0.5)
- Generates realistic technical reports with marked key points
- **Use case**: Distilling long documents into concise summaries

### Running Tests

#### Quick Sanity Check
```bash
uv run python tests/test_rlm_comprehensive.py --quick
```

#### Run Specific Test Types
```bash
# Needle-in-haystack (256k tokens)
uv run python tests/test_rlm_comprehensive.py --test needle --model grok-4-1-fast-reasoning --provider xai

# Multi-needle retrieval
uv run python tests/test_rlm_comprehensive.py --test multi --model grok-4-1-fast-reasoning --provider xai

# Reasoning test
uv run python tests/test_rlm_comprehensive.py --test reasoning --model grok-4-1-fast-reasoning --provider xai

# Summarization test (50k-90k tokens recommended)
uv run python tests/test_rlm_comprehensive.py --test summarization --model grok-4-1-fast-reasoning --provider xai

# Edge cases
uv run python tests/test_rlm_comprehensive.py --test edge --model grok-4-1-fast-reasoning --provider xai
```

#### Run Full Test Suite
```bash
# With all tests (expensive, ~$1-2)
uv run python tests/test_rlm_comprehensive.py --model grok-4-1-fast-reasoning --provider xai

# Skip expensive 256k test
uv run python tests/test_rlm_comprehensive.py --model grok-4-1-fast-reasoning --provider xai --no-256k

# Skip summarization test
uv run python tests/test_rlm_comprehensive.py --model grok-4-1-fast-reasoning --provider xai --no-summarization

# Minimal test suite (skip both)
uv run python tests/test_rlm_comprehensive.py --model grok-4-1-fast-reasoning --provider xai --no-256k --no-summarization
```

#### Save Results
```bash
uv run python tests/test_rlm_comprehensive.py \
  --model grok-4-1-fast-reasoning \
  --provider xai \
  --output results.json
```

### Test Results & Metrics

Each test tracks comprehensive metrics:
- **Cost**: Total USD spent on API calls
- **Tokens**: Input/output token counts
- **Sub-calls**: Number of recursive LLM invocations
- **Iterations**: Number of code generation cycles
- **Reasoning tokens**: Extended thinking tokens (for reasoning models)
- **Duration**: Wall-clock time
- **Pass/Fail**: Based on accuracy thresholds

**Summarization-specific metrics:**
- **Key point coverage**: Percentage of marked key points found in summary
- **Compression ratio**: Summary length / context length
- **Summary quality**: Pass requires 70%+ coverage AND ≤0.5 compression ratio

### Supported Providers & Models

#### OpenAI
```bash
# GPT-5 mini (recommended for testing)
--model gpt-5-mini --provider openai

# GPT-4o
--model gpt-4o --provider openai
```

#### xAI (Grok)
```bash
# Grok 4 with fast reasoning (recommended)
--model grok-4-1-fast-reasoning --provider xai

# Grok 4 standard
--model grok-4-1 --provider xai
```

### Cost Estimates

Approximate costs per test (with Grok-4-1-fast-reasoning):

| Test Type | Token Count | Typical Cost |
|-----------|-------------|--------------|
| Quick sanity | ~500 | $0.001 |
| Multi-needle | ~100k | $0.01-0.05 |
| Reasoning | ~50k | $0.01-0.03 |
| Summarization | ~50-90k | $0.006-0.03 |
| Edge cases | ~5k each | $0.005-0.01 |
| Needle (256k) | ~256k | $0.10-0.30 |
| **Full suite** | ~500k+ | **$0.20-0.50** |

Use `--max-cost` to set spending limits:
```bash
uv run python tests/test_rlm_comprehensive.py --max-cost 1.0  # Stop if any test exceeds $1
```

## Troubleshooting

### "Budget exceeded" Error

```python
# Increase budget or optimize approach
rlm = RecursiveLanguageModel(..., max_cost=10.0)
```

### "Max iterations reached"

```python
# Increase iteration limit
result = rlm.run(..., max_iterations=100)
```

### High Costs

- Check metrics: `rlm.print_metrics()`
- Enable caching: `enable_cache=True`
- Use cheaper sub-model: `sub_model="gpt-4o-mini"`
- Review call history in exported metrics

### Code Execution Fails

- Check logs for specific error
- Verify context is valid
- Ensure helpers are used correctly
- Try with `enable_security=False` for debugging (not recommended for production)

## Research Reference

Based on the Recursive Language Model paper concepts:

- **What**: Recursive sub-calls for long-context processing
- **How**: REPL-based decomposition with programmatic context access
- **Why**: Scales beyond context limits, improves accuracy on dense tasks
- **Patterns**: Filtering, chunking, verification, map-reduce

## License

MIT License - See [LICENSE](LICENSE) file

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Async/parallel sub-calls
- [ ] Additional model providers (Anthropic, etc.)
- [ ] More advanced helpers
- [ ] Visualization of recursion trees
- [ ] Performance benchmarks
- [ ] Additional examples
- [ ] Enhanced test coverage (100k+ token summarization, multi-document fusion)
- [ ] Test result visualization and analytics

## Contact

For issues and feature requests, please open an issue on GitHub.

---
