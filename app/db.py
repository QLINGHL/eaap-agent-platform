"""SQLite persistence layer.

One shared connection guarded by a re-entrant lock — plenty for a core
implementation and keeps transactions simple. Tables mirror PRD §6.3.
"""
import hashlib
import json
import sqlite3
import threading
import time
import uuid

from . import config

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def now_ts() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(config.db_path(), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA busy_timeout=5000")
        _conn.execute("PRAGMA foreign_keys=ON")
    return _conn


def reset_connection() -> None:
    """Drop the cached connection (used by tests with a fresh DB file)."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def q(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with _lock:
        return conn().execute(sql, params).fetchall()


def q1(sql: str, params: tuple = ()) -> sqlite3.Row | None:
    rows = q(sql, params)
    return rows[0] if rows else None


def x(sql: str, params: tuple = ()) -> int:
    with _lock:
        cur = conn().execute(sql, params)
        conn().commit()
        return cur.lastrowid


def jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def jload(s, default=None):
    if s is None:
        return default
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return default


SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, plan TEXT DEFAULT 'free',
    status TEXT DEFAULT 'active', created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, key_hash TEXT NOT NULL,
    scopes TEXT DEFAULT '[]', created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, name TEXT NOT NULL,
    description TEXT DEFAULT '', status TEXT DEFAULT 'draft',
    draft_spec TEXT, current_version_id TEXT,
    created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_versions (
    id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
    version INTEGER NOT NULL, graph_spec TEXT NOT NULL, changelog TEXT DEFAULT '',
    created_by TEXT DEFAULT '', created_at REAL NOT NULL,
    UNIQUE(agent_id, version)
);
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, name TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, kb_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
    name TEXT NOT NULL, content_hash TEXT DEFAULT '', created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, kb_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL, seq INTEGER NOT NULL, content TEXT NOT NULL,
    vector TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, agent_id TEXT NOT NULL,
    version_id TEXT NOT NULL, graph_spec TEXT NOT NULL, status TEXT NOT NULL,
    input TEXT DEFAULT '{}', output TEXT, total_tokens INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0, trace_id TEXT NOT NULL, error TEXT,
    cancel_requested INTEGER DEFAULT 0, hitl_mode TEXT DEFAULT 'pause',
    created_at REAL NOT NULL, started_at REAL, ended_at REAL
);
CREATE TABLE IF NOT EXISTS node_executions (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, node_id TEXT NOT NULL,
    node_type TEXT NOT NULL, attempt INTEGER NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE, input TEXT, output TEXT,
    status TEXT NOT NULL, latency_ms INTEGER DEFAULT 0, tokens INTEGER DEFAULT 0,
    error TEXT, started_at REAL, ended_at REAL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
    seq INTEGER NOT NULL, state TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS run_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
    type TEXT NOT NULL, payload TEXT DEFAULT '{}', created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, node_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL, role TEXT DEFAULT '', message TEXT DEFAULT '',
    status TEXT DEFAULT 'pending', decision TEXT, comment TEXT DEFAULT '',
    modifications TEXT, created_at REAL NOT NULL, decided_at REAL
);
CREATE TABLE IF NOT EXISTS eval_datasets (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, name TEXT NOT NULL,
    version INTEGER DEFAULT 1, cases TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
    agent_id TEXT NOT NULL, agent_version_id TEXT NOT NULL, status TEXT NOT NULL,
    config TEXT DEFAULT '{}', metrics TEXT, created_at REAL NOT NULL, ended_at REAL
);
CREATE TABLE IF NOT EXISTS eval_case_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT, eval_run_id TEXT NOT NULL,
    case_idx INTEGER NOT NULL, input TEXT, output TEXT, passed INTEGER,
    latency_ms INTEGER DEFAULT 0, tokens INTEGER DEFAULT 0, detail TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT, actor TEXT DEFAULT '',
    action TEXT NOT NULL, resource TEXT DEFAULT '', detail TEXT DEFAULT '{}',
    prev_hash TEXT DEFAULT '', hash TEXT DEFAULT '', created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS quota_usage (
    tenant_id TEXT NOT NULL, period TEXT NOT NULL, metric TEXT NOT NULL,
    used REAL DEFAULT 0, PRIMARY KEY (tenant_id, period, metric)
);
CREATE INDEX IF NOT EXISTS idx_runs_tenant ON runs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_chunks_kb ON chunks(kb_id);
CREATE INDEX IF NOT EXISTS idx_ck_run ON checkpoints(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_ev_run ON run_events(run_id, id);
"""


def init_db() -> None:
    with _lock:
        conn().executescript(SCHEMA)
        conn().commit()


# ---------------------------------------------------------------- audit ----

_SENSITIVE = ("key", "token", "secret", "password", "authorization", "credential")


def redact(obj):
    """Recursively redact values whose key looks sensitive (for audit logs)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if any(s in str(k).lower() for s in _SENSITIVE):
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def audit(tenant_id: str | None, actor: str, action: str, resource: str = "",
          detail: dict | None = None) -> None:
    """Append to the audit log with a hash chain (tamper-evident, cf. FR-1007)."""
    detail = redact(detail or {})
    with _lock:
        prev = q1("SELECT hash FROM audit_logs ORDER BY id DESC LIMIT 1")
        prev_hash = prev["hash"] if prev else "GENESIS"
        payload = f"{prev_hash}|{tenant_id}|{actor}|{action}|{resource}|{jdump(detail)}"
        h = hashlib.sha256(payload.encode()).hexdigest()
        x("INSERT INTO audit_logs (tenant_id, actor, action, resource, detail,"
          " prev_hash, hash, created_at) VALUES (?,?,?,?,?,?,?,?)",
          (tenant_id, actor, action, resource, jdump(detail), prev_hash, h, now_ts()))


def add_usage(tenant_id: str, metric: str, amount: float) -> None:
    period = time.strftime("%Y-%m-%d", time.gmtime())
    x("INSERT INTO quota_usage (tenant_id, period, metric, used) VALUES (?,?,?,?) "
      "ON CONFLICT(tenant_id, period, metric) DO UPDATE SET used = used + excluded.used",
      (tenant_id, period, metric, amount))
