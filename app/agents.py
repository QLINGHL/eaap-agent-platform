"""Agent lifecycle: create, draft, validate, publish (immutable versions,
FR-106). Publishing re-validates the graph — invalid graphs cannot publish.
"""
from . import db
from .graph import assert_valid, validate_graph


def create_agent(tenant_id: str, name: str, description: str = "") -> dict:
    agent_id = db.new_id("agt")
    now = db.now_ts()
    db.x("INSERT INTO agents (id, tenant_id, name, description, status,"
         " created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
         (agent_id, tenant_id, name, description, "draft", now, now))
    db.audit(tenant_id, "api", "agent.created", f"agent:{agent_id}",
             {"name": name})
    return get_agent(tenant_id, agent_id)


def get_agent(tenant_id: str, agent_id: str) -> dict | None:
    row = db.q1("SELECT * FROM agents WHERE id=? AND tenant_id=?",
                (agent_id, tenant_id))
    if row is None:
        return None
    agent = dict(row)
    agent["draft_spec"] = db.jload(agent["draft_spec"])
    versions = db.q("SELECT id, version, changelog, created_at FROM"
                    " agent_versions WHERE agent_id=? ORDER BY version",
                    (agent_id,))
    agent["versions"] = [dict(v) for v in versions]
    return agent


def list_agents(tenant_id: str) -> list[dict]:
    rows = db.q("SELECT id FROM agents WHERE tenant_id=? ORDER BY created_at",
                (tenant_id,))
    return [get_agent(tenant_id, r["id"]) for r in rows]


def save_draft(tenant_id: str, agent_id: str, spec: dict) -> dict:
    agent = get_agent(tenant_id, agent_id)
    if agent is None:
        raise ValueError("agent not found")
    db.x("UPDATE agents SET draft_spec=?, updated_at=? WHERE id=?",
         (db.jdump(spec), db.now_ts(), agent_id))
    return get_agent(tenant_id, agent_id)


def validate_draft(tenant_id: str, agent_id: str) -> list[str]:
    agent = get_agent(tenant_id, agent_id)
    if agent is None:
        raise ValueError("agent not found")
    if agent["draft_spec"] is None:
        return ["agent has no draft graph"]
    return validate_graph(agent["draft_spec"])


def publish(tenant_id: str, agent_id: str, changelog: str = "") -> dict:
    agent = get_agent(tenant_id, agent_id)
    if agent is None:
        raise ValueError("agent not found")
    if agent["draft_spec"] is None:
        raise ValueError("agent has no draft graph to publish")
    assert_valid(agent["draft_spec"])  # raises GraphValidationError
    last = db.q1("SELECT COALESCE(MAX(version),0) AS v FROM agent_versions"
                 " WHERE agent_id=?", (agent_id,))
    version_no = (last["v"] if last else 0) + 1
    version_id = db.new_id("ver")
    db.x("INSERT INTO agent_versions (id, agent_id, tenant_id, version,"
         " graph_spec, changelog, created_by, created_at)"
         " VALUES (?,?,?,?,?,?,?,?)",
         (version_id, agent_id, tenant_id, version_no,
          db.jdump(agent["draft_spec"]), changelog, "api", db.now_ts()))
    db.x("UPDATE agents SET current_version_id=?, status='published',"
         " updated_at=? WHERE id=?", (version_id, db.now_ts(), agent_id))
    db.audit(tenant_id, "api", "agent.published", f"agent:{agent_id}",
             {"version": version_no, "version_id": version_id})
    return {"agent_id": agent_id, "version_id": version_id,
            "version": version_no}
