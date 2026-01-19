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
├── tests/                        # Unit tests
│   ├── test_helpers.py           # Helper function tests
│   ├── test_cache.py             # Caching tests
│   ├── test_metrics.py           # Metrics tests
│   ├── test_mock.py              # Mock/stub tests
│   ├── test_rlm_comprehensive.py # Integration tests
│   └── test_data_generator.py    # Test data generation
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

# Install dependencies
uv pip install -r requirements.txt

# Install in development mode
uv pip install -e .
```

### Or install with pip

```bash
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

### Quickstart Examples

For the simplest possible usage, start with the quickstart examples:

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

# Run tests
make test

# Run with coverage
make test-coverage
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

## Contact

For issues and feature requests, please open an issue on GitHub.

---

**Built with ❤️ for processing massive contexts recursively**
