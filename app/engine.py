"""Runtime engine (FR-201..209 core): a persistent state machine over the
graph DSL.

- A checkpoint (full execution state) is written after EVERY node, so a run
  can resume after a crash from its last completed node.
- Node executions are recorded with idempotency key run_id:node_id:attempt.
- human_review nodes pause the run (status=waiting_approval) until a
  decision is recorded, then resume from the checkpoint.
- loop nodes implement controlled iteration; parallel nodes run branch
  chains concurrently and join.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from . import config, db, knowledge, tools as tools_mod
from .expr import ExprError, evaluate, render
from .guard import scan
from .providers import Gateway, ProviderError

_gateway: Gateway | None = None
_live_threads: dict[str, threading.Thread] = {}


def gateway() -> Gateway:
    global _gateway
    if _gateway is None:
        _gateway = Gateway()
    return _gateway


def reset_gateway() -> None:
    global _gateway
    _gateway = None


class NodeFailed(Exception):
    pass


class SimulatedCrash(Exception):
    """Test hook: simulates a worker being killed mid-run."""


# ------------------------------------------------------------------ events --

def emit(run_id: str, type_: str, payload: dict | None = None) -> None:
    db.x("INSERT INTO run_events (run_id, type, payload, created_at)"
         " VALUES (?,?,?,?)", (run_id, type_, db.jdump(payload or {}), db.now_ts()))


# ------------------------------------------------------------------- runs --

def create_run(tenant_id: str, agent_id: str, input_data: dict,
               version_id: str | None = None, hitl_mode: str = "pause") -> dict:
    agent = db.q1("SELECT * FROM agents WHERE id=? AND tenant_id=?",
                  (agent_id, tenant_id))
    if agent is None:
        raise ValueError(f"agent '{agent_id}' not found")
    if version_id:
        version = db.q1("SELECT * FROM agent_versions WHERE id=? AND agent_id=?",
                        (version_id, agent_id))
        if version is None:
            raise ValueError(f"version '{version_id}' not found for agent")
    else:
        if not agent["current_version_id"]:
            raise ValueError("agent has no published version")
        version = db.q1("SELECT * FROM agent_versions WHERE id=?",
                        (agent["current_version_id"],))
    run_id = db.new_id("run")
    db.x("INSERT INTO runs (id, tenant_id, agent_id, version_id, graph_spec,"
         " status, input, trace_id, hitl_mode, created_at)"
         " VALUES (?,?,?,?,?,?,?,?,?,?)",
         (run_id, tenant_id, agent_id, version["id"], version["graph_spec"],
          "queued", db.jdump(input_data or {}), db.new_id("trace"),
          hitl_mode, db.now_ts()))
    db.audit(tenant_id, "api", "run.created", f"run:{run_id}",
             {"agent_id": agent_id, "version_id": version["id"]})
    return get_run(tenant_id, run_id)


def get_run(tenant_id: str, run_id: str) -> dict | None:
    row = db.q1("SELECT * FROM runs WHERE id=? AND tenant_id=?",
                (run_id, tenant_id))
    if row is None:
        return None
    run = dict(row)
    run["input"] = db.jload(run["input"], {})
    run["output"] = db.jload(run["output"])
    run.pop("graph_spec", None)
    return run


def get_trace(tenant_id: str, run_id: str) -> dict | None:
    run = get_run(tenant_id, run_id)
    if run is None:
        return None
    execs = db.q("SELECT * FROM node_executions WHERE run_id=? ORDER BY rowid",
                 (run_id,))
    events = db.q("SELECT * FROM run_events WHERE run_id=? ORDER BY id",
                  (run_id,))
    ckpts = db.q1("SELECT COUNT(*) AS n FROM checkpoints WHERE run_id=?",
                  (run_id,))
    return {
        "run": run,
        "node_executions": [
            {**dict(e), "input": db.jload(e["input"]), "output": db.jload(e["output"])}
            for e in execs],
        "events": [{**dict(ev), "payload": db.jload(ev["payload"], {})}
                   for ev in events],
        "checkpoints": ckpts["n"] if ckpts else 0,
    }


def cancel_run(tenant_id: str, run_id: str) -> dict:
    run = get_run(tenant_id, run_id)
    if run is None:
        raise ValueError("run not found")
    db.x("UPDATE runs SET cancel_requested=1 WHERE id=?", (run_id,))
    if run["status"] in ("queued", "waiting_approval"):
        db.x("UPDATE runs SET status='cancelled', ended_at=? WHERE id=?",
             (db.now_ts(), run_id))
        emit(run_id, "run.cancelled", {})
    db.audit(tenant_id, "api", "run.cancelled", f"run:{run_id}", {})
    return get_run(tenant_id, run_id)


# -------------------------------------------------------------- checkpoint --

def _save_checkpoint(run_id: str, state: dict) -> None:
    last = db.q1("SELECT COALESCE(MAX(seq),0) AS s FROM checkpoints WHERE run_id=?",
                 (run_id,))
    db.x("INSERT INTO checkpoints (run_id, seq, state, created_at) VALUES (?,?,?,?)",
         (run_id, (last["s"] + 1) if last else 1, db.jdump(state), db.now_ts()))


def _load_state(run_id: str, spec: dict, input_data: dict) -> dict:
    row = db.q1("SELECT state FROM checkpoints WHERE run_id=? ORDER BY seq DESC"
                " LIMIT 1", (run_id,))
    if row:
        return db.jload(row["state"])
    start = next(n for n in spec["nodes"] if n["type"] == "start")
    return {"pc": start["id"],
            "ctx": {"input": input_data, "nodes": {}, "vars": {}},
            "loop_counts": {}, "step": 0}


# ------------------------------------------------------------- node exec ----

class _Exec:
    """One execution pass over a run."""

    def __init__(self, run_row, spec: dict):
        self.run = dict(run_row)
        self.spec = spec
        self.nodes = {n["id"]: n for n in spec["nodes"]}
        self.edges: dict[str, list[dict]] = {}
        for e in spec.get("edges", []):
            self.edges.setdefault(e["from"], []).append(e)
        self.tenant = self.run["tenant_id"]
        self.run_id = self.run["id"]

    # -- recording ------------------------------------------------------
    def _record(self, node, attempt, input_obj, output_obj, status,
                latency_ms, tokens, error=None) -> None:
        db.x("INSERT INTO node_executions (id, run_id, node_id, node_type,"
             " attempt, idempotency_key, input, output, status, latency_ms,"
             " tokens, error, started_at, ended_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
             (db.new_id("nex"), self.run_id, node["id"], node["type"], attempt,
              f"{self.run_id}:{node['id']}:{attempt}", db.jdump(input_obj),
              db.jdump(output_obj), status, latency_ms, tokens, error,
              db.now_ts(), db.now_ts()))

    def _attempt_of(self, node_id: str) -> int:
        row = db.q1("SELECT COUNT(*) AS n FROM node_executions"
                    " WHERE run_id=? AND node_id=?", (self.run_id, node_id))
        return (row["n"] if row else 0) + 1

    # -- edge selection ---------------------------------------------------
    def select_next(self, node: dict, output, ctx: dict) -> str | None:
        edges = self.edges.get(node["id"], [])
        if not edges:
            return None
        default = None
        for e in edges:
            when = e.get("when")
            if when is None:
                default = e["to"]
                continue
            if isinstance(when, bool):
                if output.get("result") is when or output.get("decision") is when:
                    return e["to"]
                continue
            w = str(when)
            if w in ("true", "false"):
                val = output.get("result", output.get("decision")) \
                    if isinstance(output, dict) else None
                if isinstance(val, bool) and str(val).lower() == w:
                    return e["to"]
                continue
            if isinstance(output, dict) and str(output.get("decision")) == w:
                return e["to"]
            try:
                if evaluate(w, ctx):
                    return e["to"]
            except ExprError:
                continue
        return default

    # -- single node ------------------------------------------------------
    def execute_node(self, node: dict, ctx: dict, state: dict):
        """Execute one node (with retries). Returns (output, forced_next).
        Raises NodeFailed, or _PauseApproval for HITL in pause mode."""
        cfg = node.get("config") or {}
        retry = cfg.get("retry") or {}
        max_attempts = max(1, int(retry.get("max_attempts", 1)))
        backoff = float(retry.get("backoff_ms", 0)) / 1000.0
        attempt = self._attempt_of(node["id"])
        last_err = None
        for i in range(max_attempts):
            started = time.time()
            emit(self.run_id, "node.started",
                 {"node_id": node["id"], "node_type": node["type"]})
            try:
                output, forced_next = self._dispatch(node, cfg, ctx, state)
            except _PauseApproval:
                raise
            except Exception as e:  # noqa: BLE001
                latency = int((time.time() - started) * 1000)
                self._record(node, attempt + i, cfg, None, "failed",
                             latency, 0, str(e))
                last_err = e
                if i + 1 < max_attempts and backoff:
                    time.sleep(backoff * (2 ** i))
                continue
            latency = int((time.time() - started) * 1000)
            tokens = 0
            if isinstance(output, dict) and isinstance(output.get("usage"), dict):
                tokens = output["usage"].get("total_tokens", 0)
            self._record(node, attempt + i, cfg, output, "completed",
                         latency, tokens)
            ctx["nodes"][node["id"]] = output
            emit(self.run_id, "node.completed",
                 {"node_id": node["id"], "latency_ms": latency, "tokens": tokens})
            return output, forced_next
        raise NodeFailed(f"node '{node['id']}' failed after "
                         f"{max_attempts} attempt(s): {last_err}")

    def _dispatch(self, node: dict, cfg: dict, ctx: dict, state: dict):
        t = node["type"]
        handler = getattr(self, f"_node_{t}")
        return handler(node, cfg, ctx, state)

    # -- node types ---------------------------------------------------------
    def _node_start(self, node, cfg, ctx, state):
        return dict(ctx["input"]), None

    def _node_end(self, node, cfg, ctx, state):
        if "output" in cfg:
            out = render(cfg["output"], ctx)
        elif ctx["vars"]:
            out = dict(ctx["vars"])
        else:
            out = {"nodes": list(ctx["nodes"].keys())}
        return out, None

    def _node_llm(self, node, cfg, ctx, state):
        messages = []
        if cfg.get("system"):
            messages.append({"role": "system",
                             "content": str(render(cfg["system"], ctx))})
        messages.append({"role": "user",
                         "content": str(render(cfg["prompt"], ctx))})
        res = gateway().chat(self.tenant, messages, cfg["model"],
                             float(cfg.get("temperature", 0.2)))
        output = {
            "text": res["text"],
            "confidence": res["confidence"] if res["confidence"] is not None else 0.5,
            "model": res["model"], "provider": res["provider"],
            "fallback_used": res["fallback_used"],
            "usage": res["usage"], "cost": res["cost"],
        }
        warnings = scan(res["text"])
        if warnings:
            output["injection_warnings"] = warnings
        db.x("UPDATE runs SET total_tokens = total_tokens + ?,"
             " total_cost = total_cost + ? WHERE id=?",
             (res["usage"]["total_tokens"], res["cost"], self.run_id))
        db.add_usage(self.tenant, "tokens", res["usage"]["total_tokens"])
        db.add_usage(self.tenant, "cost_usd", res["cost"])
        return output, None

    def _node_knowledge_retrieval(self, node, cfg, ctx, state):
        query = cfg.get("query")
        query = str(render(query, ctx)) if query else \
            str(ctx["input"].get("question") or ctx["input"].get("query") or "")
        results = knowledge.search(self.tenant, list(cfg["kb_ids"]), query,
                                   int(cfg.get("top_k", 5)),
                                   float(cfg.get("min_score", 0.0)))
        context_text = "\n\n".join(
            f"[{r['doc_name']}#{r['seq']}] {r['content']}" for r in results)
        return {"chunks": results, "context": context_text,
                "citations": [r["citation"] for r in results],
                "count": len(results)}, None

    def _node_tool(self, node, cfg, ctx, state):
        args = render(cfg.get("args") or {}, ctx)
        result = tools_mod.execute_tool(self.tenant, cfg["tool"], args,
                                        run_id=self.run_id)
        emit(self.run_id, "tool.called",
             {"node_id": node["id"], "tool": cfg["tool"]})
        return (result if isinstance(result, dict) else {"result": result}), None

    def _node_code(self, node, cfg, ctx, state):
        code = str(render(cfg["code"], ctx))
        payload = json.dumps({"input": ctx["input"], "vars": ctx["vars"]})
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code], input=payload,
                capture_output=True, text=True,
                timeout=float(cfg.get("timeout_s", 10)),
                cwd=tempfile.gettempdir(), env=env)
        except subprocess.TimeoutExpired:
            raise NodeFailed(f"code node '{node['id']}' timed out")
        parsed = None
        try:
            parsed = json.loads(proc.stdout) if proc.stdout.strip() else None
        except ValueError:
            parsed = None
        output = {"stdout": proc.stdout[:8000], "stderr": proc.stderr[:2000],
                  "returncode": proc.returncode, "result": parsed}
        if proc.returncode != 0 and not cfg.get("allow_failure"):
            raise NodeFailed(f"code node '{node['id']}' exited "
                             f"{proc.returncode}: {proc.stderr[:300]}")
        return output, None

    def _node_http_request(self, node, cfg, ctx, state):
        method = str(cfg.get("method", "GET")).upper()
        url = str(render(cfg["url"], ctx))
        headers = render(cfg.get("headers") or {}, ctx)
        body = render(cfg["json"], ctx) if "json" in cfg else None
        try:
            resp = httpx.request(method, url, headers=headers, json=body,
                                 timeout=float(cfg.get("timeout_s", 15)))
        except httpx.HTTPError as e:
            if cfg.get("allow_failure"):
                return {"error": str(e), "status": None}, None
            raise NodeFailed(f"http_request failed: {e}")
        parsed = None
        try:
            parsed = resp.json()
        except ValueError:
            pass
        return {"status": resp.status_code, "body": resp.text[:8000],
                "json": parsed}, None

    def _node_variable_assign(self, node, cfg, ctx, state):
        assigned = {}
        for key, val in cfg["assignments"].items():
            if isinstance(val, str) and val.startswith("="):
                assigned[key] = evaluate(val[1:], ctx)
            else:
                assigned[key] = render(val, ctx)
            ctx["vars"][key] = assigned[key]
        return assigned, None

    def _node_condition(self, node, cfg, ctx, state):
        result = bool(evaluate(cfg["expr"], ctx))
        return {"expr": cfg["expr"], "result": result}, None

    def _node_loop(self, node, cfg, ctx, state):
        counts = state.setdefault("loop_counts", {})
        count = counts.get(node["id"], 0)
        cond = bool(evaluate(cfg["condition"], ctx))
        edges = {e.get("when"): e["to"] for e in self.edges.get(node["id"], [])}
        if cond and count < int(cfg["max_iterations"]):
            counts[node["id"]] = count + 1
            return {"iteration": count + 1, "action": "loop"}, edges.get("loop")
        return {"iteration": count, "action": "exit"}, edges.get("exit")

    def _node_parallel(self, node, cfg, ctx, state):
        branches: list[list[str]] = cfg["branches"]
        results: dict[int, dict] = {}
        failures: dict[int, str] = {}

        def run_branch(idx: int, chain: list[str]) -> None:
            local_ctx = copy.deepcopy(ctx)
            outputs = {}
            try:
                for nid in chain:
                    bnode = self.nodes[nid]
                    out, _ = self.execute_node(bnode, local_ctx, state)
                    outputs[nid] = out
                results[idx] = {"outputs": outputs,
                                "vars": local_ctx["vars"],
                                "nodes": local_ctx["nodes"]}
            except Exception as e:  # noqa: BLE001
                failures[idx] = str(e)

        with ThreadPoolExecutor(max_workers=max(2, len(branches))) as pool:
            futures = [pool.submit(run_branch, i, chain)
                       for i, chain in enumerate(branches)]
            for f in futures:
                f.result()
        if failures:
            raise NodeFailed(f"parallel branch(es) failed: {failures}")
        for idx in sorted(results):
            ctx["nodes"].update(results[idx]["nodes"])
            ctx["vars"].update(results[idx]["vars"])
        return {"branches": {str(i): list(results[i]["outputs"].keys())
                             for i in results}}, None

    def _node_human_review(self, node, cfg, ctx, state):
        # Reached only via the pause/resume logic in execute_run.
        raise NodeFailed("human_review handled by engine pause logic")


class _PauseApproval(Exception):
    def __init__(self, approval_id: str):
        self.approval_id = approval_id


# --------------------------------------------------------------- execution --

def execute_run(run_id: str, crash_after: int | None = None) -> dict:
    """Run (or continue) a run synchronously until it completes, fails,
    pauses for approval, or is cancelled. crash_after is a test hook that
    simulates the worker being killed after N nodes in this call."""
    run_row = db.q1("SELECT * FROM runs WHERE id=?", (run_id,))
    if run_row is None:
        raise ValueError("run not found")
    spec = db.jload(run_row["graph_spec"])
    ex = _Exec(run_row, spec)
    tenant = run_row["tenant_id"]
    first_start = run_row["status"] == "queued"
    db.x("UPDATE runs SET status='running', started_at=COALESCE(started_at,?)"
         " WHERE id=?", (db.now_ts(), run_id))
    state = _load_state(run_id, spec, db.jload(run_row["input"], {}))
    if first_start:
        emit(run_id, "run.started", {"trace_id": run_row["trace_id"]})
    ctx = state["ctx"]
    completed_this_call = 0
    try:
        while state["pc"] is not None:
            fresh = db.q1("SELECT cancel_requested, status FROM runs WHERE id=?",
                          (run_id,))
            if fresh and fresh["cancel_requested"]:
                db.x("UPDATE runs SET status='cancelled', ended_at=? WHERE id=?",
                     (db.now_ts(), run_id))
                emit(run_id, "run.cancelled", {})
                return get_run(tenant, run_id)
            node = ex.nodes[state["pc"]]
            if node["type"] == "end":
                output, _ = ex.execute_node(node, ctx, state)
                state["pc"] = None
                _save_checkpoint(run_id, state)
                db.x("UPDATE runs SET status='completed', output=?, ended_at=?"
                     " WHERE id=?", (db.jdump(output), db.now_ts(), run_id))
                emit(run_id, "run.completed", {"output": output})
                db.audit(tenant, "engine", "run.completed", f"run:{run_id}", {})
                return get_run(tenant, run_id)
            if node["type"] == "human_review":
                _handle_human_review(ex, node, ctx, state)
                completed_this_call += 1
                if crash_after and completed_this_call >= crash_after:
                    raise SimulatedCrash()
                continue
            output, forced_next = ex.execute_node(node, ctx, state)
            state["step"] = state.get("step", 0) + 1
            if state["step"] > config.max_run_steps():
                raise NodeFailed(f"run exceeded max steps "
                                 f"({config.max_run_steps()})")
            nxt = forced_next if forced_next is not None else \
                ex.select_next(node, output, ctx)
            if nxt is None and node["type"] != "end":
                raise NodeFailed(f"node '{node['id']}' has no next node")
            state["pc"] = nxt
            _save_checkpoint(run_id, state)
            completed_this_call += 1
            if crash_after and completed_this_call >= crash_after:
                raise SimulatedCrash()
    except _PauseApproval:
        return get_run(tenant, run_id)
    except SimulatedCrash:
        # Worker "killed": status stays 'running', checkpoint is durable,
        # resume_run() will pick up from the last completed node.
        return get_run(tenant, run_id)
    except (NodeFailed, ProviderError, ExprError) as e:
        db.x("UPDATE runs SET status='failed', error=?, ended_at=? WHERE id=?",
             (str(e), db.now_ts(), run_id))
        emit(run_id, "run.failed", {"error": str(e)})
        db.audit(tenant, "engine", "run.failed", f"run:{run_id}",
                 {"error": str(e)})
        return get_run(tenant, run_id)
    return get_run(tenant, run_id)


def _handle_human_review(ex: _Exec, node: dict, ctx: dict, state: dict) -> None:
    run_id, tenant = ex.run_id, ex.tenant
    pending = db.q1("SELECT * FROM approvals WHERE run_id=? AND node_id=?"
                    " ORDER BY created_at DESC LIMIT 1", (run_id, node["id"]))
    decided = pending and pending["status"] != "pending"
    if not decided:
        if pending is None:
            approval_id = db.new_id("appr")
            message = str(render((node.get("config") or {}).get("message", ""),
                                 ctx))
            db.x("INSERT INTO approvals (id, run_id, node_id, tenant_id, role,"
                 " message, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
                 (approval_id, run_id, node["id"], tenant,
                  (node.get("config") or {}).get("role", ""), message,
                  "pending", db.now_ts()))
            emit(run_id, "approval.requested",
                 {"approval_id": approval_id, "node_id": node["id"]})
            db.audit(tenant, "engine", "approval.requested",
                     f"approval:{approval_id}", {"run_id": run_id})
        else:
            approval_id = pending["id"]
        if ex.run["hitl_mode"] == "auto_approve":
            db.x("UPDATE approvals SET status='approved', decision='approved',"
                 " comment='auto-approved (eval mode)', decided_at=?"
                 " WHERE id=?", (db.now_ts(), approval_id))
            decided = True
            pending = db.q1("SELECT * FROM approvals WHERE id=?", (approval_id,))
        else:
            db.x("UPDATE runs SET status='waiting_approval' WHERE id=?",
                 (run_id,))
            _save_checkpoint(run_id, state)
            raise _PauseApproval(approval_id)
    # Decision exists: complete the node and move on.
    decision = pending["decision"] or pending["status"]
    modifications = db.jload(pending["modifications"], {}) or {}
    if modifications:
        ctx["vars"].update(modifications)
    output = {"decision": decision, "comment": pending["comment"] or "",
              "modifications": modifications,
              "approval_id": pending["id"]}
    ex._record(node, ex._attempt_of(node["id"]), node.get("config") or {},
               output, "completed", 0, 0)
    ctx["nodes"][node["id"]] = output
    emit(run_id, "node.completed", {"node_id": node["id"], "decision": decision})
    nxt = ex.select_next(node, output, ctx)
    if nxt is None:
        raise NodeFailed(f"human_review node '{node['id']}' has no next node "
                         f"for decision '{decision}'")
    state["pc"] = nxt
    db.x("UPDATE runs SET status='running' WHERE id=?", (run_id,))
    _save_checkpoint(run_id, state)


def resume_run(tenant_id: str, run_id: str, sync: bool = False) -> dict:
    run = get_run(tenant_id, run_id)
    if run is None:
        raise ValueError("run not found")
    if run["status"] in ("completed", "failed", "cancelled"):
        raise ValueError(f"run is already {run['status']}")
    if run["status"] == "waiting_approval":
        pending = db.q1("SELECT * FROM approvals WHERE run_id=? AND"
                        " status='pending'", (run_id,))
        if pending:
            raise ValueError("approval still pending; record a decision first")
    if sync:
        return execute_run(run_id)
    start_run_async(run_id)
    return get_run(tenant_id, run_id)


def decide_approval(tenant_id: str, approval_id: str, decision: str,
                    comment: str = "", modifications: dict | None = None) -> dict:
    row = db.q1("SELECT * FROM approvals WHERE id=? AND tenant_id=?",
                (approval_id, tenant_id))
    if row is None:
        raise ValueError("approval not found")
    if row["status"] != "pending":
        raise ValueError(f"approval already decided ({row['status']})")
    if decision not in ("approved", "rejected", "modified"):
        raise ValueError("decision must be approved|rejected|modified")
    db.x("UPDATE approvals SET status=?, decision=?, comment=?,"
         " modifications=?, decided_at=? WHERE id=?",
         (decision, decision, comment, db.jdump(modifications or {}),
          db.now_ts(), approval_id))
    db.audit(tenant_id, "api", "approval.decided", f"approval:{approval_id}",
             {"decision": decision, "run_id": row["run_id"]})
    return dict(db.q1("SELECT * FROM approvals WHERE id=?", (approval_id,)))


def start_run_async(run_id: str) -> threading.Thread:
    def _target():
        try:
            execute_run(run_id)
        finally:
            _live_threads.pop(run_id, None)

    thread = threading.Thread(target=_target, daemon=True,
                              name=f"eaap-run-{run_id}")
    _live_threads[run_id] = thread
    thread.start()
    return thread
