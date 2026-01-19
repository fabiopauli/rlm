"""
Unit tests for RLM cache system.

Run with: pytest tests/test_cache.py
"""

import pytest
import time
from rlm.cache import RLMCache


class TestRLMCache:
    """Test RLMCache functionality."""

    def test_cache_basic_put_get(self):
        cache = RLMCache(max_size=10)

        cache.put("test prompt", "gpt-4o", "test response")
        result = cache.get("test prompt", "gpt-4o")

        assert result == "test response"
        assert cache.hits == 1
        assert cache.misses == 0

    def test_cache_miss(self):
        cache = RLMCache(max_size=10)

        result = cache.get("nonexistent", "gpt-4o")

        assert result is None
        assert cache.hits == 0
        assert cache.misses == 1

    def test_cache_different_models(self):
        cache = RLMCache(max_size=10)

        cache.put("same prompt", "gpt-4o", "response 1")
        cache.put("same prompt", "gpt-4o-mini", "response 2")

        result1 = cache.get("same prompt", "gpt-4o")
        result2 = cache.get("same prompt", "gpt-4o-mini")

        assert result1 == "response 1"
        assert result2 == "response 2"

    def test_cache_lru_eviction(self):
        cache = RLMCache(max_size=3)

        # Fill cache
        cache.put("prompt1", "model", "response1")
        cache.put("prompt2", "model", "response2")
        cache.put("prompt3", "model", "response3")

        # Access prompt1 to make it recently used
        cache.get("prompt1", "model")

        # Add new entry, should evict prompt2 (least recently used)
        cache.put("prompt4", "model", "response4")

        assert cache.get("prompt1", "model") == "response1"  # Still in cache
        assert cache.get("prompt2", "model") is None         # Evicted
        assert cache.get("prompt3", "model") == "response3"  # Still in cache
        assert cache.get("prompt4", "model") == "response4"  # Newly added

    def test_cache_ttl_expiration(self):
        cache = RLMCache(max_size=10, ttl=0.1)  # 100ms TTL

        cache.put("test", "model", "response")

        # Should be in cache immediately
        assert cache.get("test", "model") == "response"

        # Wait for TTL to expire
        time.sleep(0.15)

        # Should be expired now
        assert cache.get("test", "model") is None

    def test_cache_stats(self):
        cache = RLMCache(max_size=10)

        cache.put("p1", "m", "r1")
        cache.put("p2", "m", "r2")

        cache.get("p1", "m")  # Hit
        cache.get("p1", "m")  # Hit
        cache.get("p3", "m")  # Miss

        stats = cache.get_stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["total_requests"] == 3
        assert stats["hit_rate_percent"] == pytest.approx(66.67, rel=0.1)

    def test_cache_clear(self):
        cache = RLMCache(max_size=10)

        cache.put("p1", "m", "r1")
        cache.put("p2", "m", "r2")

        cache.clear()

        assert cache.get("p1", "m") is None
        assert cache.get("p2", "m") is None
        assert cache.get_stats()["size"] == 0
        assert cache.hits == 0
        assert cache.misses == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
