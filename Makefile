# Makefile for Recursive Language Model (RLM)
# Using uv for fast, modern Python package management

.PHONY: help install test clean examples lint format dev

help:
	@echo "Available commands:"
	@echo "  make install       - Install package and dependencies with uv"
	@echo "  make dev           - Install package in development mode"
	@echo "  make test          - Run tests with pytest"
	@echo "  make test-coverage - Run tests with coverage report"
	@echo "  make examples      - Run all example scripts"
	@echo "  make quickstart    - Run quickstart examples"
	@echo "  make lint          - Run linting with flake8"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Remove generated files and cache"

install:
	uv pip install -r requirements.txt

dev:
	uv pip install -e .
	uv pip install -r requirements.txt

test:
	uv run pytest tests/ -v

test-coverage:
	uv run pytest tests/ --cov=rlm --cov-report=html --cov-report=term

examples:
	@echo "Running basic usage example..."
	uv run python examples/basic_usage.py
	@echo "\nRunning classification example..."
	uv run python examples/classification_example.py
	@echo "\nRunning verification example..."
	uv run python examples/verification_example.py
	@echo "\nRunning long output example..."
	uv run python examples/long_output_example.py
	@echo "\nRunning advanced patterns example..."
	uv run python examples/advanced_patterns.py
	@echo "\nRunning Grok examples..."
	uv run python examples/grok_basic_example.py
	uv run python examples/grok_reasoning_example.py
	@echo "\nRunning multi-provider example..."
	uv run python examples/multi_provider_example.py

quickstart:
	@echo "Running quickstart examples..."
	@echo "\nGrok quickstart:"
	uv run python examples/quickstart_grok.py
	@echo "\nGPT-5 quickstart:"
	uv run python examples/quickstart_gpt5.py

lint:
	uv run flake8 rlm/ tests/ examples/ --max-line-length=100

format:
	uv run black rlm/ tests/ examples/ --line-length=100

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache tests/.pytest_cache
	rm -rf htmlcov .coverage
	rm -rf *.egg-info dist build
	rm -rf .mypy_cache
	rm -f *_metrics.json
	rm -f long_output_result.txt
	find . -name "*.pyc" -delete
