"""Runtime configuration, read lazily from environment variables."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def db_path() -> str:
    return os.environ.get("EAAP_DB_PATH", str(PROJECT_ROOT / "eaap.db"))


def llm_base_url() -> str:
    return os.environ.get("EAAP_LLM_BASE_URL", "")


def llm_api_key() -> str:
    return os.environ.get("EAAP_LLM_API_KEY", "")


def llm_model() -> str:
    return os.environ.get("EAAP_LLM_MODEL", "gpt-4o-mini")


def rate_limit_capacity() -> int:
    return int(os.environ.get("EAAP_RATE_LIMIT_CAPACITY", "30"))


def rate_limit_refill_per_sec() -> float:
    return float(os.environ.get("EAAP_RATE_LIMIT_REFILL_PER_SEC", "0.5"))


def max_run_steps() -> int:
    return int(os.environ.get("EAAP_MAX_RUN_STEPS", "500"))


def tool_breaker_threshold() -> int:
    """Consecutive tool failures before the circuit breaker opens (FR-407)."""
    return int(os.environ.get("EAAP_TOOL_BREAKER_THRESHOLD", "5"))


def tool_breaker_cooldown_sec() -> float:
    """Seconds a tool stays paused after the breaker opens (FR-407)."""
    return float(os.environ.get("EAAP_TOOL_BREAKER_COOLDOWN_SEC", "60"))
