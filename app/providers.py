"""Model gateway (FR-301/303/304/306): provider abstraction, a deterministic
offline mock provider, an OpenAI-compatible provider, primary→fallback chain,
per-tenant token-bucket rate limiting, token counting and cost estimation.
"""
import hashlib
import json
import re
import time

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

    def __init__(self):
        self.mock = MockProvider()
        self.real = None
        if config.llm_base_url() and config.llm_api_key():
            self.real = OpenAICompatibleProvider(config.llm_base_url(),
                                                 config.llm_api_key())
        self.bucket = TokenBucket(config.rate_limit_capacity(),
                                  config.rate_limit_refill_per_sec())

    def _chain(self, model: str):
        """(provider, model) attempts in order: requested, then fallbacks."""
        if model.startswith("mock"):
            return [(self.mock, model), (self.mock, "mock-1")]
        if self.real is not None:
            return [(self.real, model), (self.mock, "mock-1")]
        return [(self.mock, "mock-1")]

    def chat(self, tenant_id: str, messages: list[dict], model: str,
             temperature: float = 0.2) -> dict:
        if not self.bucket.take(tenant_id):
            raise RateLimitError(f"tenant {tenant_id} exceeded LLM rate limit")
        attempts = self._chain(model)
        last_err: ProviderError | None = None
        for i, (provider, mdl) in enumerate(attempts):
            try:
                out = provider.chat(messages, mdl, temperature)
                total = out["prompt_tokens"] + out["completion_tokens"]
                return {
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
                }
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
