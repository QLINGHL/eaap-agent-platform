"""Model gateway (FR-301/303/304/305/306): provider abstraction, a deterministic
offline mock provider, an OpenAI-compatible provider, primary→fallback chain,
per-tenant token-bucket rate limiting, token counting and cost estimation,
and an in-memory semantic cache (FR-305).
"""
import hashlib
import json
import re
import time
from collections import OrderedDict

import httpx

from . import config


class ProviderError(Exception):
    def __init__(self, message: str, code: str = "PROVIDER_ERROR", retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class RateLimitError(ProviderError):
    def __init__(self, message: str = "rate limit exceeded"):
        super().__init__(message, code="RATE_LIMITED", retryable=True)


# Rough per-1K-token prices (USD) used for cost *estimation* only.
PRICES = {
    "mock-1": (0.001, 0.002),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "default": (0.001, 0.002),
}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICES.get(model, PRICES["default"])
    return round(prompt_tokens / 1000 * pin + completion_tokens / 1000 * pout, 6)


class MockProvider:
    """Deterministic offline provider. Same input → same output.

    Conventions used by the demos:
    - If the prompt contains a non-empty 'SOURCES:' section, the answer is
      grounded in the first source passage and confidence is high (0.92).
    - Otherwise the answer is generic and confidence is low (0.30), which is
      what routes the demo contract agent to human review.
    - Model name 'mock-fail' always raises, to exercise the fallback chain.
    """
    name = "mock"

    def chat(self, messages: list[dict], model: str, temperature: float = 0.2) -> dict:
        if model == "mock-fail":
            raise ProviderError("mock provider instructed to fail (model=mock-fail)")
        prompt = "\n".join(m.get("content", "") for m in messages)
        sources = ""
        if "SOURCES:" in prompt:
            tail = prompt.split("SOURCES:", 1)[1]
            for marker in ("\nQUESTION:", "\nUSER:", "\nINPUT:"):
                if marker in tail:
                    tail = tail.split(marker)[0]
            sources = tail.strip()
        qm = re.search(r"QUESTION:\s*(.+)", prompt, re.S | re.I)
        question = qm.group(1).strip().split("\n")[0] if qm else None
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]

        if sources:
            first = sources.split("\n")[0][:400]
            text = (f"Based on the retrieved sources: {first} "
                    f"(Answering: {question})")
            confidence = 0.92
        elif question is not None:
            text = (f"I don't have verified source material for this question "
                    f"('{question}'). A human should review before this is "
                    f"sent out. [mock:{digest}]")
            confidence = 0.30
        else:
            # Open-ended generation task (no Q&A structure): generic draft.
            headline = prompt.strip().split("\n")[0][:120]
            text = (f"[mock draft] {headline} — drafted from the supplied "
                    f"inputs. [mock:{digest}]")
            confidence = 0.5
        pt = sum(estimate_tokens(m.get("content", "")) for m in messages)
        ct = estimate_tokens(text)
        return {"text": text, "confidence": confidence,
                "prompt_tokens": pt, "completion_tokens": ct}


class OpenAICompatibleProvider:
    """Any endpoint speaking the OpenAI chat-completions protocol."""
    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def chat(self, messages: list[dict], model: str, temperature: float = 0.2) -> dict:
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": model, "messages": messages,
                      "temperature": temperature},
                timeout=60,
            )
        except httpx.HTTPError as e:
            raise ProviderError(f"provider unreachable: {e}")
        if resp.status_code >= 500:
            raise ProviderError(f"provider returned {resp.status_code}", retryable=True)
        if resp.status_code >= 400:
            raise ProviderError(f"provider returned {resp.status_code}: {resp.text[:200]}",
                                retryable=False)
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        pt = usage.get("prompt_tokens") or sum(
            estimate_tokens(m.get("content", "")) for m in messages)
        ct = usage.get("completion_tokens") or estimate_tokens(text)
        return {"text": text, "confidence": None,
                "prompt_tokens": pt, "completion_tokens": ct}


def normalize_prompt(messages: list[dict]) -> str:
    """Canonical form of a chat request for cache lookup (FR-305).

    Two prompts that differ only in case or surrounding/duplicate whitespace
    are treated as the same prompt, so a rephrased-by-formatting repeat hits
    the cache instead of calling the provider again. This is a
    normalization-based semantic match: it needs no embeddings and works
    fully offline. Message roles are kept, so a system/user swap is a miss.
    """
    parts = []
    for m in messages:
        role = str(m.get("role", "")).strip().lower()
        content = re.sub(r"\s+", " ", str(m.get("content", ""))).strip().casefold()
        parts.append(f"{role}:{content}")
    return "\n".join(parts)


class SemanticCache:
    """In-memory LRU cache for gateway responses (FR-305).

    Key: (tenant_id, model, temperature, normalized prompt). Tenant is part
    of the key so one tenant can never read another tenant's cached reply.
    Only successful provider responses are stored; failures and rate-limit
    rejections are never cached. ``stats()`` reports hits, misses, evictions
    and the current size so operators can see whether the cache pays off.
    """

    def __init__(self, max_size: int = 128, ttl_seconds: float = 0):
        self.max_size = max(1, max_size)
        self.ttl_seconds = max(0.0, ttl_seconds)
        self._entries: OrderedDict[tuple, tuple[dict, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _expired(self, stored_at: float) -> bool:
        return (self.ttl_seconds > 0
                and time.monotonic() - stored_at > self.ttl_seconds)

    def get(self, key: tuple) -> dict | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        value, stored_at = entry
        if self._expired(stored_at):
            del self._entries[key]
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return dict(value)

    def put(self, key: tuple, value: dict) -> None:
        if key in self._entries:
            del self._entries[key]
        self._entries[key] = (dict(value), time.monotonic())
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": len(self._entries),
            "max_size": self.max_size,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


class TokenBucket:
    def __init__(self, capacity: int, refill_per_sec: float):
        self.capacity = capacity
        self.refill = refill_per_sec
        self._state: dict[str, tuple[float, float]] = {}

    def take(self, key: str) -> bool:
        now = time.monotonic()
        tokens, last = self._state.get(key, (float(self.capacity), now))
        tokens = min(self.capacity, tokens + (now - last) * self.refill)
        if tokens >= 1:
            self._state[key] = (tokens - 1, now)
            return True
        self._state[key] = (tokens, now)
        return False


class Gateway:
    """Routes a chat request through the provider chain for a model."""

    def __init__(self, cache: SemanticCache | None = None,
                 cache_enabled: bool = True):
        self.mock = MockProvider()
        self.real = None
        if config.llm_base_url() and config.llm_api_key():
            self.real = OpenAICompatibleProvider(config.llm_base_url(),
                                                 config.llm_api_key())
        self.bucket = TokenBucket(config.rate_limit_capacity(),
                                  config.rate_limit_refill_per_sec())
        self.cache_enabled = cache_enabled
        if cache is not None:
            self.cache = cache
        elif cache_enabled:
            self.cache = SemanticCache(config.cache_max_size(),
                                       config.cache_ttl_sec())
        else:
            self.cache = None

    def cache_stats(self) -> dict:
        """Hit/miss statistics; zeros when the cache is disabled."""
        if self.cache is None:
            return {"hits": 0, "misses": 0, "evictions": 0, "size": 0,
                    "max_size": 0, "hit_rate": 0.0, "enabled": False}
        stats = self.cache.stats()
        stats["enabled"] = True
        return stats

    def _cache_key(self, tenant_id: str, messages: list[dict], model: str,
                   temperature: float) -> tuple:
        return (tenant_id, model, round(temperature, 4),
                normalize_prompt(messages))

    def _chain(self, model: str):
        """(provider, model) attempts in order: requested, then fallbacks."""
        if model.startswith("mock"):
            return [(self.mock, model), (self.mock, "mock-1")]
        if self.real is not None:
            return [(self.real, model), (self.mock, "mock-1")]
        return [(self.mock, "mock-1")]

    def chat(self, tenant_id: str, messages: list[dict], model: str,
             temperature: float = 0.2) -> dict:
        # Cache first: a hit answers without spending rate-limit budget or
        # calling the provider. Only misses reach the bucket and the chain.
        key = None
        if self.cache is not None:
            key = self._cache_key(tenant_id, messages, model, temperature)
            cached = self.cache.get(key)
            if cached is not None:
                cached["cache_hit"] = True
                return cached
        if not self.bucket.take(tenant_id):
            raise RateLimitError(f"tenant {tenant_id} exceeded LLM rate limit")
        attempts = self._chain(model)
        last_err: ProviderError | None = None
        for i, (provider, mdl) in enumerate(attempts):
            try:
                out = provider.chat(messages, mdl, temperature)
                total = out["prompt_tokens"] + out["completion_tokens"]
                result = {
                    "text": out["text"],
                    "confidence": out.get("confidence"),
                    "model": mdl,
                    "provider": provider.name,
                    "fallback_used": i > 0,
                    "usage": {"prompt_tokens": out["prompt_tokens"],
                              "completion_tokens": out["completion_tokens"],
                              "total_tokens": total},
                    "cost": estimate_cost(mdl, out["prompt_tokens"],
                                          out["completion_tokens"]),
                    "cache_hit": False,
                }
                if self.cache is not None and key is not None:
                    self.cache.put(key, result)
                return result
            except ProviderError as e:
                last_err = e
                if not e.retryable:
                    raise
        raise last_err or ProviderError("all providers failed")


def extract_json(text: str):
    """Best-effort JSON extraction from an LLM reply (FR-306 helper)."""
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except ValueError:
            return None
    return None
