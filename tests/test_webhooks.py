"""FR-1304 webhook delivery: registration, filtering, retry-once, audit."""
import pytest

from app import webhooks
from conftest import wait_for_run

SIMPLE_GRAPH = {
    "nodes": [
        {"id": "start", "type": "start"},
        {"id": "end", "type": "end", "config": {"output": "done"}},
    ],
    "edges": [{"from": "start", "to": "end"}],
}


def _make_agent_run(client, headers_a, name="wh-agent"):
    r = client.post("/v1/agents", json={"name": name}, headers=headers_a)
    agent_id = r.json()["id"]
    client.put(f"/v1/agents/{agent_id}/draft",
               json={"graph": SIMPLE_GRAPH}, headers=headers_a)
    client.post(f"/v1/agents/{agent_id}/publish", json={},
                headers=headers_a)
    r = client.post(f"/v1/agents/{agent_id}/runs", json={"input": {}},
                    headers=headers_a)
    return r.json()["id"]


def test_webhook_crud_and_validation(client, headers_a, headers_b):
    r = client.post("/v1/webhooks",
                    json={"url": "https://example.com/hook"},
                    headers=headers_a)
    assert r.status_code == 200
    wh = r.json()
    assert wh["events"] == list(webhooks.TERMINAL_EVENTS)
    assert wh["active"] is True

    # Tenant isolation: another tenant sees nothing and cannot delete it.
    assert client.get("/v1/webhooks", headers=headers_b).json() == []
    r = client.delete(f"/v1/webhooks/{wh['id']}", headers=headers_b)
    assert r.status_code == 404

    # Invalid URL and unknown event are rejected.
    r = client.post("/v1/webhooks", json={"url": "ftp://x/y"},
                    headers=headers_a)
    assert r.status_code == 400
    r = client.post("/v1/webhooks",
                    json={"url": "https://example.com/h",
                          "events": ["run.started"]},
                    headers=headers_a)
    assert r.status_code == 400

    r = client.get("/v1/webhooks", headers=headers_a)
    assert [w["id"] for w in r.json()] == [wh["id"]]
    r = client.delete(f"/v1/webhooks/{wh['id']}", headers=headers_a)
    assert r.json() == {"deleted": wh["id"]}
    assert client.get("/v1/webhooks", headers=headers_a).json() == []


def test_completed_run_delivered_once_and_filtered(client, headers_a,
                                                   monkeypatch):
    calls = []

    def fake_post(url, payload):
        calls.append((url, payload))
        return 200

    monkeypatch.setattr(webhooks, "_post", fake_post)

    r = client.post("/v1/webhooks",
                    json={"url": "https://example.com/all"},
                    headers=headers_a)
    all_id = r.json()["id"]
    r = client.post("/v1/webhooks",
                    json={"url": "https://example.com/failed-only",
                          "events": ["run.failed"]},
                    headers=headers_a)
    failed_only_id = r.json()["id"]

    run_id = _make_agent_run(client, headers_a)
    run = wait_for_run(client, run_id, headers_a)
    assert run["status"] == "completed"

    # Only the unfiltered endpoint got the completed event, exactly once.
    assert len(calls) == 1
    assert calls[0][0] == "https://example.com/all"
    assert calls[0][1]["event"] == "run.completed"
    assert calls[0][1]["run_id"] == run_id

    deliveries = client.get(f"/v1/webhooks/{all_id}/deliveries",
                            headers=headers_a).json()
    assert len(deliveries) == 1
    assert deliveries[0]["status"] == "delivered"
    assert deliveries[0]["attempts"] == 1
    assert deliveries[0]["event"] == "run.completed"
    assert client.get(f"/v1/webhooks/{failed_only_id}/deliveries",
                      headers=headers_a).json() == []


def test_failed_delivery_retried_exactly_once(client, headers_a, monkeypatch):
    calls = []

    def failing_post(url, payload):
        calls.append(url)
        return 500

    monkeypatch.setattr(webhooks, "_post", failing_post)
    r = client.post("/v1/webhooks",
                    json={"url": "https://example.com/down"},
                    headers=headers_a)
    wh_id = r.json()["id"]

    run_id = _make_agent_run(client, headers_a, name="wh-retry-agent")
    wait_for_run(client, run_id, headers_a)

    # One retry: two attempts total, then recorded as failed.
    # (The session DB is shared, so earlier tests' endpoints may also fire.)
    assert calls.count("https://example.com/down") == 2
    deliveries = client.get(f"/v1/webhooks/{wh_id}/deliveries",
                            headers=headers_a).json()
    assert len(deliveries) == 1
    assert deliveries[0]["status"] == "failed"
    assert deliveries[0]["attempts"] == 2
    assert deliveries[0]["last_status_code"] == 500


def test_delivery_failure_does_not_break_run(client, headers_a, monkeypatch):
    monkeypatch.setattr(webhooks, "_post",
                        lambda url, payload: (_ for _ in ()).throw(
                            ConnectionError("unreachable")))
    client.post("/v1/webhooks", json={"url": "https://example.com/dead"},
                headers=headers_a)
    run_id = _make_agent_run(client, headers_a, name="wh-robust-agent")
    run = wait_for_run(client, run_id, headers_a)
    assert run["status"] == "completed"
