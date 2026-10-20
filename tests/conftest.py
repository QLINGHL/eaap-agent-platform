import hashlib
import os
import sys
import tempfile
import time

# Point the app at a throwaway DB and disable rate limiting BEFORE imports.
_TMP = tempfile.mkdtemp(prefix="eaap-test-")
os.environ["EAAP_DB_PATH"] = os.path.join(_TMP, "test.db")
os.environ["EAAP_RATE_LIMIT_CAPACITY"] = "100000"
os.environ["EAAP_LLM_BASE_URL"] = ""
os.environ["EAAP_LLM_API_KEY"] = ""

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from app import db  # noqa: E402

TENANT_A = "tenant_a"
TENANT_B = "tenant_b"
KEY_A = "test-key-a"
KEY_B = "test-key-b"


def _make_tenant(tid: str, key: str) -> None:
    if db.q1("SELECT id FROM tenants WHERE id=?", (tid,)) is None:
        db.x("INSERT INTO tenants (id, name, plan, status, created_at)"
             " VALUES (?,?,?,?,?)", (tid, tid, "pro", "active", db.now_ts()))
        db.x("INSERT INTO api_keys (id, tenant_id, key_hash, scopes, created_at)"
             " VALUES (?,?,?,?,?)",
             (db.new_id("key"), tid, hashlib.sha256(key.encode()).hexdigest(),
              '["*"]', db.now_ts()))


@pytest.fixture(scope="session", autouse=True)
def _db_ready():
    db.reset_connection()
    db.init_db()
    _make_tenant(TENANT_A, KEY_A)
    _make_tenant(TENANT_B, KEY_B)
    return True


@pytest.fixture(scope="session")
def headers_a():
    return {"Authorization": f"Bearer {KEY_A}"}


@pytest.fixture(scope="session")
def headers_b():
    return {"Authorization": f"Bearer {KEY_B}"}


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def wait_for_run(client, run_id: str, headers: dict, timeout: float = 15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/v1/runs/{run_id}", headers=headers).json()
        if run["status"] in ("completed", "failed", "cancelled",
                             "waiting_approval"):
            return run
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} did not settle in {timeout}s")
