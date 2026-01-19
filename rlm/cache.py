"""
Caching system for RLM sub-calls.

Implements LRU cache to avoid redundant sub-calls and save tokens/costs.
"""

import hashlib
import json
from typing import Optional, Dict, Any
from collections import OrderedDict
import time


class RLMCache:
    """
    LRU Cache for RLM sub-calls.

    Caches results of llm_query() calls to avoid redundant API calls.
    Uses prompt hash as key for fast lookup.
    """

    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached entries (LRU eviction)
            ttl: Time-to-live for cache entries in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _hash_prompt(self, prompt: str, model: str) -> str:
        """Generate hash key for a prompt and model combination."""
        key = f"{model}:{prompt}"
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[str]:
        """
        Retrieve cached result for a prompt.

        Args:
            prompt: The LLM prompt
            model: The model name

        Returns:
            Cached response or None if not found/expired
        """
        key = self._hash_prompt(prompt, model)

        if key not in self._cache:
            self.misses += 1
            return None

        entry = self._cache[key]

        # Check TTL expiration
        if self.ttl is not None:
            age = time.time() - entry["timestamp"]
            if age > self.ttl:
                # Expired, remove and return None
                del self._cache[key]
                self.misses += 1
                return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self.hits += 1

        return entry["response"]

    def put(self, prompt: str, model: str, response: str):
        """
        Store a response in cache.

        Args:
            prompt: The LLM prompt
            model: The model name
            response: The LLM response to cache
        """
        key = self._hash_prompt(prompt, model)

        # Remove oldest entry if at capacity
        if key not in self._cache and len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest (FIFO)

        # Store entry
        self._cache[key] = {
            "prompt": prompt,
            "model": model,
            "response": response,
            "timestamp": time.time()
        }

        # Move to end (most recently used)
        self._cache.move_to_end(key)

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 2),
            "ttl_seconds": self.ttl
        }

    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("RLM Cache Statistics")
        print("="*60)
        print(f"Size: {stats['size']}/{stats['max_size']}")
        print(f"Hits: {stats['hits']}")
        print(f"Misses: {stats['misses']}")
        print(f"Hit rate: {stats['hit_rate_percent']}%")
        if stats['ttl_seconds']:
            print(f"TTL: {stats['ttl_seconds']}s")
        print("="*60 + "\n")

    def export_to_json(self, filepath: str):
        """Export cache contents to JSON file."""
        data = {
            "stats": self.get_stats(),
            "entries": [
                {
                    "prompt_preview": entry["prompt"][:100] + "..." if len(entry["prompt"]) > 100 else entry["prompt"],
                    "model": entry["model"],
                    "response_preview": entry["response"][:100] + "..." if len(entry["response"]) > 100 else entry["response"],
                    "timestamp": entry["timestamp"],
                    "age_seconds": round(time.time() - entry["timestamp"], 2)
                }
                for entry in self._cache.values()
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
