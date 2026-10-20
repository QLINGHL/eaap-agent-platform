from conftest import wait_for_run

GRAPH = {
    "nodes": [
        {"id": "start", "type": "start"},
        {"id": "llm", "type": "llm",
         "config": {"model": "mock-1", "prompt": "QUESTION: {{input.q}}"}},
        {"id": "human", "type": "human_review",
         "config": {"role": "ops", "message": "please review"}},
        {"id": "end", "type": "end", "config": {"output": "sent"}},
    ],
    "edges": [
        {"from": "start", "to": "llm"}, {"from": "llm", "to": "human"},
        {"from": "human", "to": "end"},
    ],
}


def test_full_lifecycle_and_hitl_over_api(client, headers_a):
    r = client.post("/v1/agents", json={"name": "api-lifecycle"}, headers=headers_a)
    assert r.status_code == 200
    agent_id = r.json()["id"]

    r = client.put(f"/v1/agents/{agent_id}/draft",
                   json={"graph": GRAPH}, headers=headers_a)
    assert r.status_code == 200

    r = client.post(f"/v1/agents/{agent_id}/validate", headers=headers_a)
    assert r.json() == {"valid": True, "errors": []}

    r = client.post(f"/v1/agents/{agent_id}/publish",
                    json={"changelog": "v1"}, headers=headers_a)
    assert r.status_code == 200
    assert r.json()["version"] == 1

    r = client.post(f"/v1/agents/{agent_id}/runs",
                    json={"input": {"q": "draft the email"}}, headers=headers_a)
    assert r.status_code == 200
    run = wait_for_run(client, r.json()["id"], headers_a)
    assert run["status"] == "waiting_approval"

    r = client.get("/v1/approvals", headers=headers_a)
    pending = [a for a in r.json() if a["run_id"] == run["id"]]
    assert len(pending) == 1

    r = client.post(f"/v1/approvals/{pending[0]['id']}/decision",
                    json={"decision": "approved", "comment": "looks good"},
                    headers=headers_a)
    assert r.status_code == 200

    run = wait_for_run(client, run["id"], headers_a)
    assert run["status"] == "completed"
    assert run["output"] == "sent"
    assert run["total_tokens"] > 0

    trace = client.get(f"/v1/runs/{run['id']}/trace", headers=headers_a).json()
    types = [e["type"] for e in trace["events"]]
    assert "run.started" in types and "approval.requested" in types
    assert "run.completed" in types
    assert trace["checkpoints"] >= 3

    # SSE replay of the finished run's events.
    with client.stream("GET", f"/v1/runs/{run['id']}/events",
                       headers=headers_a) as resp:
        assert resp.status_code == 200
        body = "".join(chunk for chunk in resp.iter_text())
    assert "event: run.completed" in body

    usage = client.get("/v1/usage", headers=headers_a).json()
    assert usage["totals"].get("tokens", 0) > 0
    assert usage["runs"] >= 1


def test_publish_invalid_graph_returns_422(client, headers_a):
    r = client.post("/v1/agents", json={"name": "api-invalid"}, headers=headers_a)
    agent_id = r.json()["id"]
    bad = {"nodes": [{"id": "start", "type": "start"}], "edges": []}
    client.put(f"/v1/agents/{agent_id}/draft", json={"graph": bad},
               headers=headers_a)
    r = client.post(f"/v1/agents/{agent_id}/publish", json={},
                    headers=headers_a)
    assert r.status_code == 422
    assert r.json()["detail"]["errors"]


def test_kb_endpoints_and_search(client, headers_a):
    r = client.post("/v1/knowledge-bases", json={"name": "api-kb"},
                    headers=headers_a)
    kb_id = r.json()["id"]
    r = client.post(f"/v1/knowledge-bases/{kb_id}/documents",
                    json={"name": "Refund Policy",
                          "content": "Refunds are issued within 14 days "
                                     "of purchase to the original payment "
                                     "method."},
                    headers=headers_a)
    assert r.status_code == 200
    assert r.json()["chunks"] >= 1
    r = client.post(f"/v1/knowledge-bases/{kb_id}/search",
                    json={"query": "how long do refunds take"},
                    headers=headers_a)
    results = r.json()["results"]
    assert results and "14 days" in results[0]["content"]
    assert results[0]["doc_name"] == "Refund Policy"
    assert results[0]["citation"] == "[Refund Policy#0]"


def test_tools_endpoints(client, headers_a):
    r = client.get("/v1/tools", headers=headers_a)
    names = [t["name"] for t in r.json()["tools"]]
    assert "calculator" in names and "http_fetch" in names
    r = client.post("/v1/tools", json={"name": "calculator",
                                       "args": {"expression": "10 / 4"}},
                    headers=headers_a)
    assert r.json()["result"]["result"] == 2.5
    r = client.post("/v1/tools", json={"name": "calculator",
                                       "args": {"expression": 5}},
                    headers=headers_a)
    assert r.status_code == 400
