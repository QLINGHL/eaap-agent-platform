from app import agents as agents_mod
from app import engine, knowledge
from conftest import TENANT_A, TENANT_B, wait_for_run

GRAPH = {
    "nodes": [
        {"id": "start", "type": "start"},
        {"id": "llm", "type": "llm",
         "config": {"model": "mock-1", "prompt": "QUESTION: hi"}},
        {"id": "end", "type": "end", "config": {"output": "done"}},
    ],
    "edges": [{"from": "start", "to": "llm"}, {"from": "llm", "to": "end"}],
}


def _agent_for(tenant: str, name: str) -> str:
    agent = agents_mod.create_agent(tenant, name)
    agents_mod.save_draft(tenant, agent["id"], GRAPH)
    agents_mod.publish(tenant, agent["id"])
    return agent["id"]


def test_cross_tenant_run_invisible():
    agent_id = _agent_for(TENANT_A, "a-private-agent")
    run = engine.execute_run(engine.create_run(TENANT_A, agent_id, {})["id"])
    assert engine.get_run(TENANT_B, run["id"]) is None
    assert engine.get_trace(TENANT_B, run["id"]) is None


def test_cross_tenant_agent_cannot_run():
    agent_id = _agent_for(TENANT_A, "a-private-agent-2")
    try:
        engine.create_run(TENANT_B, agent_id, {})
        raise AssertionError("tenant B should not run tenant A's agent")
    except ValueError:
        pass


def test_kb_tenant_isolation():
    kb = knowledge.create_kb(TENANT_A, "a-secrets")
    knowledge.add_document(TENANT_A, kb["id"], "Secret Doc",
                           "The launch code is BLUE-FALCON-42 and the "
                           "launch window opens on Friday.")
    own = knowledge.search(TENANT_A, [kb["id"]], "launch code", 5)
    assert own and "BLUE-FALCON" in own[0]["content"]
    other = knowledge.search(TENANT_B, [kb["id"]], "launch code", 5)
    assert other == []
    try:
        knowledge.add_document(TENANT_B, kb["id"], "evil", "injected")
        raise AssertionError("tenant B should not write into tenant A's KB")
    except ValueError:
        pass


def test_api_cross_tenant_404(client, headers_a, headers_b):
    agent_id = _agent_for(TENANT_A, "a-api-agent")
    resp = client.post(f"/v1/agents/{agent_id}/runs",
                       json={"input": {}}, headers=headers_a)
    assert resp.status_code == 200
    run = wait_for_run(client, resp.json()["id"], headers_a)
    assert run["status"] == "completed"
    # Tenant B sees nothing.
    assert client.get(f"/v1/runs/{run['id']}", headers=headers_b).status_code == 404
    assert client.get(f"/v1/runs/{run['id']}/trace",
                      headers=headers_b).status_code == 404
    assert client.get(f"/v1/agents/{agent_id}",
                      headers=headers_b).status_code == 404
    # And cannot start a run on it.
    resp_b = client.post(f"/v1/agents/{agent_id}/runs",
                         json={"input": {}}, headers=headers_b)
    assert resp_b.status_code == 400


def test_auth_required(client):
    assert client.get("/v1/agents").status_code == 401
    assert client.get("/v1/agents",
                      headers={"Authorization": "Bearer wrong"}).status_code == 401
