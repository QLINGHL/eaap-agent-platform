"""EAAP REST API (PRD §7.2 core set) + SSE streaming (§7.3) + a built-in
HTML trace viewer. Tenant identity comes ONLY from the API key header —
never from the request body (PRD §7.1).
"""
import asyncio
import hashlib
import html
import json
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from . import agents as agents_mod
from . import db, engine, evals, knowledge
from .graph import GraphValidationError
from .tools import list_tools


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="EAAP — Enterprise Agent Application Platform",
              version="1.0.0", lifespan=lifespan)


# ------------------------------------------------------------------- auth --

def current_tenant(authorization: str | None = Header(default=None),
                   x_api_key: str | None = Header(default=None)) -> str:
    key = x_api_key
    if not key and authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    if not key:
        raise HTTPException(401, "missing API key")
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    row = db.q1("SELECT tenant_id FROM api_keys WHERE key_hash=?", (key_hash,))
    if row is None:
        raise HTTPException(401, "invalid API key")
    return row["tenant_id"]


def err(message: str, status: int = 400, code: str = "BAD_REQUEST"):
    raise HTTPException(status, detail={"code": code, "message": message})


# ----------------------------------------------------------------- models --

class CreateAgentBody(BaseModel):
    name: str
    description: str = ""


class DraftBody(BaseModel):
    graph: dict


class PublishBody(BaseModel):
    changelog: str = ""


class RunBody(BaseModel):
    input: dict = {}
    version_id: str | None = None
    stream: bool = False


class KBBody(BaseModel):
    name: str


class DocumentBody(BaseModel):
    name: str
    content: str


class SearchBody(BaseModel):
    query: str
    kb_ids: list[str] | None = None
    top_k: int = 5


class ToolBody(BaseModel):
    name: str
    args: dict = {}


class DecisionBody(BaseModel):
    decision: str
    comment: str = ""
    modifications: dict | None = None


class DatasetBody(BaseModel):
    name: str
    cases: list[dict] | None = None
    jsonl: str | None = None


class EvalRunBody(BaseModel):
    dataset_id: str
    agent_id: str
    version_id: str
    config: dict = {}


def _kb_or_404(tenant: str, kb_id: str) -> dict:
    kb = knowledge.get_kb(tenant, kb_id) or knowledge.find_kb_by_name(tenant, kb_id)
    if kb is None:
        err(f"knowledge base '{kb_id}' not found", 404, "NOT_FOUND")
    return kb


# ----------------------------------------------------------------- agents --

@app.get("/health")
def health():
    return {"status": "ok", "service": "eaap"}


@app.post("/v1/agents")
def create_agent(body: CreateAgentBody, tenant: str = Depends(current_tenant)):
    return agents_mod.create_agent(tenant, body.name, body.description)


@app.get("/v1/agents")
def list_agents(tenant: str = Depends(current_tenant)):
    return agents_mod.list_agents(tenant)


@app.get("/v1/agents/{agent_id}")
def get_agent(agent_id: str, tenant: str = Depends(current_tenant)):
    agent = agents_mod.get_agent(tenant, agent_id)
    if agent is None:
        err("agent not found", 404, "NOT_FOUND")
    return agent


@app.put("/v1/agents/{agent_id}/draft")
def save_draft(agent_id: str, body: DraftBody,
               tenant: str = Depends(current_tenant)):
    try:
        return agents_mod.save_draft(tenant, agent_id, body.graph)
    except ValueError as e:
        err(str(e), 404, "NOT_FOUND")


@app.post("/v1/agents/{agent_id}/validate")
def validate_agent(agent_id: str, tenant: str = Depends(current_tenant)):
    try:
        errors = agents_mod.validate_draft(tenant, agent_id)
    except ValueError as e:
        err(str(e), 404, "NOT_FOUND")
    return {"valid": not errors, "errors": errors}


@app.post("/v1/agents/{agent_id}/publish")
def publish_agent(agent_id: str, body: PublishBody,
                  tenant: str = Depends(current_tenant)):
    try:
        return agents_mod.publish(tenant, agent_id, body.changelog)
    except GraphValidationError as e:
        raise HTTPException(422, detail={"code": "GRAPH_INVALID",
                                         "message": "graph validation failed",
                                         "errors": e.errors})
    except ValueError as e:
        err(str(e), 404, "NOT_FOUND")


# ------------------------------------------------------------------- runs --

@app.post("/v1/agents/{agent_id}/runs")
def create_run(agent_id: str, body: RunBody,
               tenant: str = Depends(current_tenant)):
    try:
        run = engine.create_run(tenant, agent_id, body.input,
                                version_id=body.version_id)
    except ValueError as e:
        err(str(e), 400, "RUN_CREATE_FAILED")
    engine.start_run_async(run["id"])
    return engine.get_run(tenant, run["id"])


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str, tenant: str = Depends(current_tenant)):
    run = engine.get_run(tenant, run_id)
    if run is None:
        err("run not found", 404, "NOT_FOUND")
    return run


@app.post("/v1/runs/{run_id}/cancel")
def cancel_run(run_id: str, tenant: str = Depends(current_tenant)):
    try:
        return engine.cancel_run(tenant, run_id)
    except ValueError as e:
        err(str(e), 404, "NOT_FOUND")


@app.post("/v1/runs/{run_id}/resume")
def resume_run(run_id: str, tenant: str = Depends(current_tenant)):
    try:
        return engine.resume_run(tenant, run_id)
    except ValueError as e:
        err(str(e), 400, "RESUME_FAILED")


@app.get("/v1/runs/{run_id}/trace")
def get_trace(run_id: str, tenant: str = Depends(current_tenant)):
    trace = engine.get_trace(tenant, run_id)
    if trace is None:
        err("run not found", 404, "NOT_FOUND")
    return trace


@app.get("/v1/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request,
                        tenant: str = Depends(current_tenant)):
    """SSE stream of run events (PRD §7.3). Honors Last-Event-ID."""
    run = engine.get_run(tenant, run_id)
    if run is None:
        err("run not found", 404, "NOT_FOUND")
    last_id = int(request.headers.get("last-event-id", "0") or 0)

    async def gen():
        cursor = last_id
        idle_rounds = 0
        while idle_rounds < 40:  # ~20s max, then the client can reconnect
            rows = db.q("SELECT * FROM run_events WHERE run_id=? AND id>?"
                        " ORDER BY id", (run_id, cursor))
            for row in rows:
                cursor = row["id"]
                idle_rounds = 0
                payload = db.jload(row["payload"], {})
                yield (f"id: {row['id']}\nevent: {row['type']}\n"
                       f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
            current = engine.get_run(tenant, run_id)
            if current and current["status"] in (
                    "completed", "failed", "cancelled") and not rows:
                break
            idle_rounds += 1
            await asyncio.sleep(0.5)
        yield "event: ping\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# -------------------------------------------------------------- approvals --

@app.get("/v1/approvals")
def list_approvals(tenant: str = Depends(current_tenant),
                   status: str = "pending"):
    rows = db.q("SELECT * FROM approvals WHERE tenant_id=? AND status=?"
                " ORDER BY created_at", (tenant, status))
    return [dict(r) for r in rows]


@app.post("/v1/approvals/{approval_id}/decision")
def decide(approval_id: str, body: DecisionBody,
           tenant: str = Depends(current_tenant)):
    try:
        approval = engine.decide_approval(tenant, approval_id, body.decision,
                                          body.comment, body.modifications)
    except ValueError as e:
        err(str(e), 400, "DECISION_FAILED")
    engine.start_run_async(approval["run_id"])
    return approval


# -------------------------------------------------------------- knowledge --

@app.post("/v1/knowledge-bases")
def create_kb(body: KBBody, tenant: str = Depends(current_tenant)):
    return knowledge.create_kb(tenant, body.name)


@app.post("/v1/knowledge-bases/{kb_id}/documents")
def add_document(kb_id: str, body: DocumentBody,
                 tenant: str = Depends(current_tenant)):
    kb = _kb_or_404(tenant, kb_id)
    return knowledge.add_document(tenant, kb["id"], body.name, body.content)


@app.post("/v1/knowledge-bases/{kb_id}/search")
def search_kb(kb_id: str, body: SearchBody,
              tenant: str = Depends(current_tenant)):
    kb = _kb_or_404(tenant, kb_id)
    ids = body.kb_ids or [kb["id"]]
    results = knowledge.search(tenant, ids, body.query, body.top_k)
    return {"results": results}


# ------------------------------------------------------------------ tools --

@app.get("/v1/tools")
def get_tools(tenant: str = Depends(current_tenant)):
    return {"tools": list_tools()}


@app.post("/v1/tools")
def register_tool_info(body: ToolBody, tenant: str = Depends(current_tenant)):
    """Execute a registered builtin tool directly (registry is code-defined;
    this endpoint is the tool invocation/debug surface)."""
    from .tools import ToolError, execute_tool
    try:
        result = execute_tool(tenant, body.name, body.args)
    except ToolError as e:
        raise HTTPException(400, detail={"code": e.code, "message": str(e)})
    return {"tool": body.name, "result": result}


# ------------------------------------------------------------------ evals --

@app.post("/v1/evals/datasets")
def create_dataset(body: DatasetBody, tenant: str = Depends(current_tenant)):
    if body.cases is not None:
        cases = body.cases
    elif body.jsonl:
        try:
            cases = evals.parse_jsonl(body.jsonl)
        except ValueError as e:
            err(f"invalid JSONL: {e}", 400, "BAD_REQUEST")
    else:
        err("provide 'cases' or 'jsonl'", 400, "BAD_REQUEST")
    return evals.create_dataset(tenant, body.name, cases)


@app.post("/v1/evals/runs")
def create_eval_run(body: EvalRunBody, tenant: str = Depends(current_tenant)):
    try:
        return evals.run_eval(tenant, body.dataset_id, body.agent_id,
                              body.version_id, body.config)
    except ValueError as e:
        err(str(e), 400, "EVAL_FAILED")


@app.get("/v1/evals/runs/{eval_id}")
def get_eval_run(eval_id: str, tenant: str = Depends(current_tenant)):
    ev = evals.get_eval(tenant, eval_id)
    if ev is None:
        err("eval run not found", 404, "NOT_FOUND")
    return ev


@app.get("/v1/evals/compare")
def compare(a: str, b: str, tenant: str = Depends(current_tenant)):
    try:
        return evals.compare_evals(tenant, a, b)
    except ValueError as e:
        err(str(e), 404, "NOT_FOUND")


# ------------------------------------------------------------------ usage --

@app.get("/v1/usage")
def usage(tenant: str = Depends(current_tenant)):
    rows = db.q("SELECT metric, SUM(used) AS total FROM quota_usage"
                " WHERE tenant_id=? GROUP BY metric", (tenant,))
    totals = {r["metric"]: r["total"] for r in rows}
    runs = db.q1("SELECT COUNT(*) AS n FROM runs WHERE tenant_id=?",
                 (tenant,))
    return {"tenant_id": tenant, "totals": totals,
            "runs": runs["n"] if runs else 0}


# ------------------------------------------------- trace viewer (extra) ----

@app.get("/ui/runs/{run_id}", response_class=HTMLResponse)
def trace_viewer(run_id: str, tenant: str = Depends(current_tenant)):
    trace = engine.get_trace(tenant, run_id)
    if trace is None:
        err("run not found", 404, "NOT_FOUND")
    run = trace["run"]
    rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{} ms</td><td>{}</td></tr>"
        .format(html.escape(e["node_id"]), html.escape(e["node_type"]),
                html.escape(e["status"]), e["latency_ms"], e["tokens"])
        for e in trace["node_executions"])
    events = "".join(f"<li>{html.escape(ev['type'])}</li>"
                     for ev in trace["events"])
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>EAAP trace {html.escape(run_id)}</title></head><body>
<h1>Run {html.escape(run_id)}</h1>
<p>Status: <b>{html.escape(run['status'])}</b> · tokens: {run['total_tokens']}
· cost: ${run['total_cost']} · checkpoints: {trace['checkpoints']}</p>
<h2>Node executions</h2>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>Node</th><th>Type</th><th>Status</th><th>Latency</th><th>Tokens</th></tr>
{rows}</table>
<h2>Events</h2><ul>{events}</ul>
</body></html>"""
