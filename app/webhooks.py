"""FR-1304 webhook delivery: POST run terminal events to tenant URLs.

A tenant registers one or more endpoints (optionally filtered to a subset
of terminal events). When a run reaches a terminal state — completed,
failed or cancelled — each matching active endpoint receives one POST
with the event payload. A failed delivery (network error or non-2xx
response) is retried exactly once; every attempt is recorded in
webhook_deliveries so tenants can audit what was sent. Delivery failures
never change the run's own status.
"""
import json
import time
import urllib.error
import urllib.request

from . import db

TERMINAL_EVENTS = ("run.completed", "run.failed", "run.cancelled")
MAX_ATTEMPTS = 2  # initial attempt + exactly one retry
TIMEOUT_SECONDS = 5


def validate_url(url: str) -> None:
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")


def validate_events(events: list | None) -> list[str]:
    if not events:
        return list(TERMINAL_EVENTS)
    unknown = [e for e in events if e not in TERMINAL_EVENTS]
    if unknown:
        raise ValueError(f"unknown webhook event(s): {', '.join(unknown)}")
    return list(dict.fromkeys(events))


def create_endpoint(tenant_id: str, url: str,
                    events: list | None = None) -> dict:
    validate_url(url)
    events = validate_events(events)
    endpoint_id = db.new_id("wh")
    db.x("INSERT INTO webhook_endpoints (id, tenant_id, url, events, active,"
         " created_at) VALUES (?,?,?,?,?,?)",
         (endpoint_id, tenant_id, url, db.jdump(events), 1, db.now_ts()))
    return get_endpoint(tenant_id, endpoint_id)


def get_endpoint(tenant_id: str, endpoint_id: str) -> dict | None:
    row = db.q1("SELECT * FROM webhook_endpoints WHERE id=? AND tenant_id=?",
                (endpoint_id, tenant_id))
    if row is None:
        return None
    endpoint = dict(row)
    endpoint["events"] = db.jload(endpoint["events"], [])
    endpoint["active"] = bool(endpoint["active"])
    return endpoint


def list_endpoints(tenant_id: str) -> list[dict]:
    rows = db.q("SELECT * FROM webhook_endpoints WHERE tenant_id=?"
                " ORDER BY created_at", (tenant_id,))
    out = []
    for row in rows:
        endpoint = dict(row)
        endpoint["events"] = db.jload(endpoint["events"], [])
        endpoint["active"] = bool(endpoint["active"])
        out.append(endpoint)
    return out


def delete_endpoint(tenant_id: str, endpoint_id: str) -> bool:
    if get_endpoint(tenant_id, endpoint_id) is None:
        return False
    db.x("DELETE FROM webhook_endpoints WHERE id=? AND tenant_id=?",
         (endpoint_id, tenant_id))
    return True


def list_deliveries(tenant_id: str, endpoint_id: str) -> list[dict]:
    if get_endpoint(tenant_id, endpoint_id) is None:
        return []
    rows = db.q("SELECT * FROM webhook_deliveries WHERE tenant_id=?"
                " AND endpoint_id=? ORDER BY created_at DESC",
                (tenant_id, endpoint_id))
    return [dict(r) for r in rows]


def _post(url: str, payload: dict) -> int:
    """Single HTTP POST; returns the status code, raises on network error."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def deliver_run_event(tenant_id: str, run_id: str, event: str,
                      payload: dict | None = None) -> list[dict]:
    """Deliver one terminal run event to every matching endpoint.

    Returns the delivery rows created. Never raises: a delivery failure is
    recorded, not propagated to the run engine.
    """
    if event not in TERMINAL_EVENTS:
        return []
    body = {"event": event, "run_id": run_id, "tenant_id": tenant_id,
            "payload": payload or {}, "delivered_at": time.time()}
    deliveries = []
    for endpoint in list_endpoints(tenant_id):
        if not endpoint["active"] or event not in endpoint["events"]:
            continue
        delivery_id = db.new_id("whd")
        db.x("INSERT INTO webhook_deliveries (id, endpoint_id, tenant_id,"
             " run_id, event, attempts, status, created_at, updated_at)"
             " VALUES (?,?,?,?,?,0,'pending',?,?)",
             (delivery_id, endpoint["id"], tenant_id, run_id, event,
              db.now_ts(), db.now_ts()))
        attempts = 0
        status_code = None
        last_error = None
        status = "failed"
        for _ in range(MAX_ATTEMPTS):
            attempts += 1
            try:
                status_code = _post(endpoint["url"], body)
                if 200 <= status_code < 300:
                    status = "delivered"
                    last_error = None
                    break
                last_error = f"HTTP {status_code}"
            except Exception as e:  # noqa: BLE001 - recorded, never raised
                last_error = str(e)
        db.x("UPDATE webhook_deliveries SET attempts=?, status=?,"
             " last_status_code=?, last_error=?, updated_at=? WHERE id=?",
             (attempts, status, status_code, last_error, db.now_ts(),
              delivery_id))
        deliveries.append(dict(db.q1(
            "SELECT * FROM webhook_deliveries WHERE id=?", (delivery_id,))))
    return deliveries
