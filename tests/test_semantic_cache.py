"""FR-305: gateway semantic cache — normalization hits, LRU, TTL, stats."""
import time

import pytest

from app.providers import (Gateway, ProviderError, SemanticCache,
                           TokenBucket, normalize_prompt)


def _msgs(text):
    return [{"role": "user", "content": text}]


def test_normalize_ignores_case_and_whitespace():
    assert normalize_prompt(_msgs("Hello   World")) == \
        normalize_prompt(_msgs("  hello world "))
    assert normalize_prompt(_msgs("Hello World")) != \
        normalize_prompt(_msgs("Goodbye World"))


def test_repeat_prompt_is_cache_hit():
    gw = Gateway()
    first = gw.chat("t1", _msgs("What is EAAP?"), "mock-1")
    second = gw.chat("t1", _msgs("What is EAAP?"), "mock-1")
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["text"] == first["text"]
    stats = gw.cache_stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["hit_rate"] == 0.5 and stats["size"] == 1


def test_case_whitespace_variant_hits():
    gw = Gateway()
    gw.chat("t1", _msgs("What is EAAP?"), "mock-1")
    again = gw.chat("t1", _msgs("  WHAT   is eaap? "), "mock-1")
    assert again["cache_hit"] is True


def test_different_prompt_tenant_temperature_are_misses():
    gw = Gateway()
    gw.chat("t1", _msgs("Prompt A"), "mock-1")
    assert gw.chat("t1", _msgs("Prompt B"), "mock-1")["cache_hit"] is False
    # Tenant isolation: same prompt, another tenant must not read t1's entry.
    assert gw.chat("t2", _msgs("Prompt A"), "mock-1")["cache_hit"] is False
    assert gw.chat("t1", _msgs("Prompt A"), "mock-1",
                  temperature=0.9)["cache_hit"] is False
    assert gw.cache_stats()["size"] == 4


def test_lru_evicts_oldest():
    cache = SemanticCache(max_size=2)
    gw = Gateway(cache=cache)
    gw.chat("t1", _msgs("one"), "mock-1")
    gw.chat("t1", _msgs("two"), "mock-1")
    gw.chat("t1", _msgs("one"), "mock-1")  # refresh "one", "two" now oldest
    gw.chat("t1", _msgs("three"), "mock-1")  # evicts "two"
    assert gw.chat("t1", _msgs("one"), "mock-1")["cache_hit"] is True
    assert gw.chat("t1", _msgs("two"), "mock-1")["cache_hit"] is False
    assert gw.cache_stats()["evictions"] >= 1


def test_ttl_expiry():
    cache = SemanticCache(max_size=10, ttl_seconds=0.05)
    gw = Gateway(cache=cache)
    gw.chat("t1", _msgs("short lived"), "mock-1")
    time.sleep(0.1)
    assert gw.chat("t1", _msgs("short lived"), "mock-1")["cache_hit"] is False


def test_failure_is_not_cached(monkeypatch):
    gw = Gateway()
    monkeypatch.setattr(gw.mock, "chat",
                        lambda *a, **k: (_ for _ in ()).throw(
                            ProviderError("boom", retryable=False)))
    with pytest.raises(ProviderError):
        gw.chat("t1", _msgs("will fail"), "mock-1")
    monkeypatch.undo()
    # Real provider answers now, and the earlier failure left no entry.
    assert gw.chat("t1", _msgs("will fail"), "mock-1")["cache_hit"] is False
    assert gw.cache_stats()["size"] == 1


def test_cache_hit_bypasses_rate_limit():
    gw = Gateway()
    gw.bucket = TokenBucket(1, 0)  # one call only, no refill
    gw.chat("t1", _msgs("cheap repeat"), "mock-1")
    # Bucket is empty, but the identical prompt is served from cache.
    assert gw.chat("t1", _msgs("cheap repeat"), "mock-1")["cache_hit"] is True
    # A new prompt still hits the exhausted bucket.
    with pytest.raises(Exception):
        gw.chat("t1", _msgs("brand new prompt"), "mock-1")


def test_disabled_cache_never_hits():
    gw = Gateway(cache_enabled=False)
    gw.chat("t1", _msgs("no cache here"), "mock-1")
    assert gw.chat("t1", _msgs("no cache here"), "mock-1")["cache_hit"] is False
    stats = gw.cache_stats()
    assert stats["enabled"] is False and stats["size"] == 0


def test_cache_stats_endpoint(client, headers_a):
    resp = client.get("/v1/gateway/cache/stats", headers=headers_a)
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    for key in ("hits", "misses", "evictions", "size", "hit_rate"):
        assert key in body
