"""FR-803: HITL approval timeout — a human_review node with
timeout_seconds auto-rejects its approval once it expires; runs parked
in waiting_approval then proceed down the rejected branch. Approvals
without a timeout never expire, and a late decision is refused."""
import pytest

from app import agents as agents_mod
from app import db, engine
from conftest import wait_for_run

T = "tenant_a"


def _publish(name: str, graph: dict) -> str:
    agent = agents_mod.create_agent(T, name)
    agents_mod.save_draft(T, agent["id"], graph)
    agents_mod.publish(T, agent["id"])
    return agent["id"]


def _graph(timeout):
    cfg = {"role": "ops", "message": "Review this"}
    if timeout is not None:
        cfg["timeout_seconds"] = timeout
    return {
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "human", "type": "human_review", "config": cfg},
            {"id": "ok", "type": "end", "config": {"output": "approved-path"}},
            {"id": "no", "type": "end", "config": {"output": "rejected-path"}},
        ],
        "edges": [
            {"from": "start", "to": "human"},
            {"from": "human", "to": "ok", "when": "approved"},
            {"from": "human", "to": "no", "when": "rejected"},
        ],
    }


def _parked_run(name: str, timeout):
    agent_id = _publish(name, _graph(timeout))
    run = engine.create_run(T, agent_id, {})
    run = engine.execute_run(run["id"])
    assert run["status"] == "waiting_approval"
    approval = db.q1("SELECT * FROM approvals WHERE run_id=?", (run["id"],))
    assert approval["status"] == "pending"
    return run, approval


def _backdate(approval_id: str, seconds: float) -> None:
    db.x("UPDATE approvals SET created_at = created_at - ? WHERE id=?",
         (seconds, approval_id))


def test_expired_approval_auto_rejects_and_run_takes_rejected_branch():
    run, approval = _parked_run("timeout-agent", 60)
    _backdate(approval["id"], 61)
    expired = engine.expire_approvals(tenant_id=T)
    assert [a["id"] for a in expired] == [approval["id"]]
    after = db.q1("SELECT * FROM approvals WHERE id=?", (approval["id"],))
    assert after["status"] == "rejected"
    assert after["decision"] == "rejected"
    assert "expired" in (after["comment"] or "")
    resumed = engine.resume_run(T, run["id"], sync=True)
    assert resumed["status"] == "completed"
    assert resumed["output"] == "rejected-path"


def test_not_yet_expired_approval_stays_pending():
    run, approval = _parked_run("timeout-not-yet-agent", 60)
    _backdate(approval["id"], 30)
    assert engine.expire_approvals(tenant_id=T) == []
    after = db.q1("SELECT * FROM approvals WHERE id=?", (approval["id"],))
    assert after["status"] == "pending"
    with pytest.raises(ValueError, match="still pending"):
        engine.resume_run(T, run["id"], sync=True)


def test_approval_without_timeout_never_expires():
    run, approval = _parked_run("timeout-none-agent", None)
    _backdate(approval["id"], 10 ** 6)
    assert engine.expire_approvals(tenant_id=T) == []
    after = db.q1("SELECT * FROM approvals WHERE id=?", (approval["id"],))
    assert after["status"] == "pending"


def test_late_decision_is_refused_and_approval_is_rejected():
    run, approval = _parked_run("timeout-late-agent", 60)
    _backdate(approval["id"], 61)
    with pytest.raises(ValueError, match="expired"):
        engine.decide_approval(T, approval["id"], "approved")
    after = db.q1("SELECT * FROM approvals WHERE id=?", (approval["id"],))
    assert after["status"] == "rejected"
    assert after["decision"] == "rejected"
    resumed = engine.resume_run(T, run["id"], sync=True)
    assert resumed["status"] == "completed"
    assert resumed["output"] == "rejected-path"


def test_invalid_timeout_config_fails_the_run():
    agent_id = _publish("timeout-invalid-agent", _graph(0))
    run = engine.create_run(T, agent_id, {})
    run = engine.execute_run(run["id"])
    assert run["status"] == "failed"
    assert "timeout_seconds" in (run["error"] or "")


def test_expire_endpoint_resumes_parked_run(client, headers_a):
    run, approval = _parked_run("timeout-api-agent", 60)
    _backdate(approval["id"], 61)
    r = client.post("/v1/approvals/expire", headers=headers_a)
    assert r.status_code == 200
    assert r.json()["expired"] == [approval["id"]]
    settled = wait_for_run(client, run["id"], headers_a)
    assert settled["status"] == "completed"
    assert settled["output"] == "rejected-path"
