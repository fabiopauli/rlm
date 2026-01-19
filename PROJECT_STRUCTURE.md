# Project Structure

```
Rlm/
│
├── rlm/                          # Main package
│   ├── __init__.py              # Package initialization
│   ├── core.py                  # Enhanced RecursiveLanguageModel
│   ├── core_legacy.py           # Legacy implementation (reference)
│   ├── providers.py             # Multi-provider support (OpenAI, xAI)
│   ├── metrics.py               # Metrics tracking system
│   ├── cache.py                 # LRU caching system
│   ├── security.py              # Sandboxing and security
│   └── helpers.py               # Advanced helper functions
│
├── examples/                     # Usage examples
│   ├── quickstart_grok.py       # Minimal Grok example
│   ├── quickstart_gpt5.py       # Minimal GPT-5 example
│   ├── basic_usage.py           # Needle-in-haystack pattern
│   ├── classification_example.py # Classification & aggregation
│   ├── verification_example.py  # Verification pattern
│   ├── long_output_example.py   # Long output generation
│   ├── advanced_patterns.py     # Map-reduce pattern
│   ├── grok_basic_example.py    # Grok integration example
│   ├── grok_reasoning_example.py # Grok reasoning metrics
│   └── multi_provider_example.py # Cross-provider comparison
│
├── tests/                        # Unit tests
│   ├── test_helpers.py          # Test helper functions
│   ├── test_cache.py            # Test caching system
│   ├── test_metrics.py          # Test metrics tracking
│   ├── test_mock.py             # Mock/stub tests
│   ├── test_rlm_comprehensive.py # Integration tests
│   └── test_data_generator.py   # Test data generation
│
├── README.md                     # Main documentation
├── PROJECT_STRUCTURE.md          # This file
│
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Package metadata
├── .env.example                 # Environment configuration template
├── Makefile                     # Development shortcuts (uv-based)
│
├── LICENSE                       # MIT License
└── .gitignore                    # Git ignore rules
```

## Component Descriptions

### Core Package (`rlm/`)

#### `core.py`
Main RLM implementation:
- `RecursiveLanguageModel` class
- REPL environment management
- LLM API integration
- Iteration loop
- Code execution
- Integration of all components

#### `metrics.py`
Comprehensive metrics tracking:
- `RLMMetrics` class
- `CallMetrics` dataclass
- Token counting
- Cost calculation
- Recursion depth tracking
- Performance statistics
- JSON export

#### `cache.py`
LRU caching for sub-calls:
- `RLMCache` class
- Hash-based lookup
- TTL support
- Hit/miss tracking
- Size management
- Statistics

#### `security.py`
Sandboxing for safe code execution:
- `RestrictedGlobals` - Safe globals dictionary
- `ExecutionMonitor` - Code safety checks
- `SafeFileAccess` - Restricted file operations
- Blocked dangerous operations

#### `helpers.py`
Advanced helper utilities:
- `TextProcessor` - Chunking, splitting, truncation
- `SearchHelper` - Regex, keyword search, sections
- `AggregationHelper` - Various aggregation strategies
- `VerificationHelper` - Answer validation
- `RecursionHelper` - Recursive patterns

### Examples (`examples/`)

Each example demonstrates a specific RLM pattern:

1. **basic_usage.py**: Needle-in-haystack with filtering + probing
2. **classification_example.py**: Chunking + classification + aggregation
3. **verification_example.py**: Extraction + verification loop
4. **long_output_example.py**: Section extraction + summarization
5. **advanced_patterns.py**: Map-reduce with sentiment analysis

### Tests (`tests/`)

Unit tests for all components:
- **test_helpers.py**: 20+ tests for helper functions
- **test_cache.py**: 10+ tests for caching system
- **test_metrics.py**: 15+ tests for metrics tracking

Run with: `pytest tests/` or `make test`

### Configuration Files

- **requirements.txt**: Python dependencies (openai, tiktoken)
- **setup.py**: Package installation configuration
- **.env.example**: Environment variable template
- **Makefile**: Development commands (install, test, lint, etc.)

### Documentation

- **README.md**: Complete usage guide (300+ lines)
- **IMPROVEMENTS.md**: Detailed comparison with original
- **CHANGELOG.md**: Version history
- **PROJECT_STRUCTURE.md**: This file

### Utility Scripts

- **quick_start.py**: Minimal example for quick testing
- **original_implementation.py**: Original code for reference

## File Statistics

```
Total Files: 24 Python files + 7 documentation files
Lines of Code:
  - rlm/ package: ~2,000 lines
  - examples/: ~500 lines
  - tests/: ~700 lines
  - Total: ~3,200 lines

Documentation:
  - README.md: ~450 lines
  - IMPROVEMENTS.md: ~350 lines
  - Other docs: ~200 lines
  - Total: ~1,000 lines
```

## Development Workflow

```bash
# 1. Install dependencies
make install

# 2. Run tests
make test

# 3. Run quick example
make quick

# 4. Run all examples
make examples

# 5. Lint code
make lint

# 6. Format code
make format

# 7. Clean generated files
make clean
```

## Key Design Decisions

1. **Modular Architecture**: Separate concerns (core, metrics, cache, security, helpers)
2. **Extensibility**: Easy to add new helpers or patterns
3. **Testing**: Comprehensive unit tests for all components
4. **Documentation**: Extensive inline docs + README + examples
5. **Safety**: Sandboxing by default with option to disable
6. **Observability**: Full metrics tracking and export
7. **Performance**: Caching and model flexibility for cost optimization

## Dependencies

### Required
- `openai>=1.0.0` - OpenAI API client
- `tiktoken>=0.5.0` - Token counting

### Optional (Dev)
- `pytest>=7.0.0` - Testing
- `pytest-cov>=4.0.0` - Coverage reports
- `black>=23.0.0` - Code formatting
- `flake8>=6.0.0` - Linting
- `mypy>=1.0.0` - Type checking

## Future Enhancements

See IMPROVEMENTS.md for planned features:
- Async/parallel sub-calls
- Additional model providers
- Recursion tree visualization
- Web UI for monitoring
- Performance benchmarks
