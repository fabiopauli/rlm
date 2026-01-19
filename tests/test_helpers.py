"""
Unit tests for RLM helper functions.

Run with: pytest tests/test_helpers.py
"""

import pytest
from rlm.helpers import (
    TextProcessor,
    SearchHelper,
    AggregationHelper,
    RecursionHelper
)


class TestTextProcessor:
    """Test TextProcessor helper functions."""

    def test_chunk_text_basic(self):
        text = "a" * 10000
        chunks = TextProcessor.chunk_text(text, chunk_size=2000, overlap=200)

        assert len(chunks) > 1
        assert all(len(chunk) <= 2000 for chunk in chunks)

    def test_chunk_text_preserve_paragraphs(self):
        text = "Paragraph 1.\n\n" * 100 + "Paragraph 2.\n\n" * 100
        chunks = TextProcessor.chunk_text(
            text, chunk_size=1000, preserve_paragraphs=True
        )

        assert len(chunks) > 1
        # Chunks should not break mid-paragraph
        assert all("\n\n" in chunk or len(chunk) < 100 for chunk in chunks)

    def test_smart_truncate(self):
        text = "This is a long sentence that needs truncation"
        truncated = TextProcessor.smart_truncate(text, max_length=20, suffix="...")

        assert len(truncated) <= 20
        assert truncated.endswith("...")
        assert " " not in truncated[-4:-3]  # Should break at word boundary


class TestSearchHelper:
    """Test SearchHelper functions."""

    def test_regex_search(self):
        text = "The numbers are 42, 123, and 7."
        matches = SearchHelper.regex_search(r"\d+", text)

        assert matches == ["42", "123", "7"]

    def test_regex_search_max_matches(self):
        text = "1 2 3 4 5"
        matches = SearchHelper.regex_search(r"\d", text, max_matches=3)

        assert len(matches) == 3

    def test_regex_search_positions(self):
        text = "abc 123 def"
        matches = SearchHelper.regex_search(r"\d+", text, return_positions=True)

        assert len(matches) == 1
        match_text, start, end = matches[0]
        assert match_text == "123"
        assert text[start:end] == "123"

    def test_find_sections(self):
        text = """
# Section 1
Content 1

## Section 2
Content 2

# Section 3
Content 3
"""
        sections = SearchHelper.find_sections(text, include_content=True)

        assert len(sections) == 3
        assert sections[0]["title"] == "Section 1"
        assert "Content 1" in sections[0]["content"]

    def test_keyword_filter(self):
        text = "This is important. This is not. This is critical. Regular text."
        snippets = SearchHelper.keyword_filter(
            text, keywords=["important", "critical"], context_chars=10
        )

        assert len(snippets) == 2


class TestAggregationHelper:
    """Test AggregationHelper functions."""

    def test_aggregate_join(self):
        results = ["a", "b", "c"]
        aggregated = AggregationHelper.aggregate_results(
            results, method="join", separator=", "
        )

        assert aggregated == "a, b, c"

    def test_aggregate_sum(self):
        results = [1, 2, 3, 4, 5]
        aggregated = AggregationHelper.aggregate_results(results, method="sum")

        assert aggregated == 15

    def test_aggregate_count(self):
        results = ["a", "b", "c", "d"]
        count = AggregationHelper.aggregate_results(results, method="count")

        assert count == 4

    def test_count_frequencies(self):
        items = ["apple", "banana", "apple", "cherry", "banana", "apple"]
        freq = AggregationHelper.count_frequencies(items)

        assert freq == {"apple": 3, "banana": 2, "cherry": 1}

    def test_merge_dicts_sum(self):
        dicts = [
            {"a": 1, "b": 2},
            {"a": 3, "c": 4},
            {"b": 5, "c": 6}
        ]
        merged = AggregationHelper.merge_dicts(dicts, merge_strategy="sum")

        assert merged == {"a": 4, "b": 7, "c": 10}

    def test_merge_dicts_last(self):
        dicts = [{"a": 1}, {"a": 2}, {"a": 3}]
        merged = AggregationHelper.merge_dicts(dicts, merge_strategy="last")

        assert merged == {"a": 3}


class TestRecursionHelper:
    """Test RecursionHelper functions."""

    def test_recursive_split(self):
        text = "a" * 10000

        def condition(t):
            return len(t) <= 1000

        def split_fn(t):
            mid = len(t) // 2
            return [t[:mid], t[mid:]]

        chunks = RecursionHelper.recursive_split(text, condition, split_fn)

        assert all(len(chunk) <= 1000 for chunk in chunks)
        assert "".join(chunks) == text  # Verify no data lost

    def test_map_reduce(self):
        items = [1, 2, 3, 4, 5]

        def map_fn(x):
            return x * 2

        def reduce_fn(results):
            return sum(results)

        result = RecursionHelper.map_reduce(items, map_fn, reduce_fn)

        assert result == 30  # (1+2+3+4+5) * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
