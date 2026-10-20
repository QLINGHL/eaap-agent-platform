"""Evaluation center (FR-902/903/904/905): versioned datasets, rule
evaluators (contains / regex / JSON-schema subset), batch re-runs over an
agent version, metrics reports and version-vs-version comparison.
"""
import json
import re
import time

from . import db, engine
from .schemaval import validate


def create_dataset(tenant_id: str, name: str, cases: list[dict]) -> dict:
    ds_id = db.new_id("ds")
    db.x("INSERT INTO eval_datasets (id, tenant_id, name, version, cases,"
         " created_at) VALUES (?,?,?,?,?,?)",
         (ds_id, tenant_id, name, 1, db.jdump(cases), db.now_ts()))
    db.audit(tenant_id, "api", "eval.dataset_created", f"dataset:{ds_id}",
             {"name": name, "cases": len(cases)})
    return {"id": ds_id, "name": name, "cases": len(cases)}


def get_dataset(tenant_id: str, dataset_id: str) -> dict | None:
    row = db.q1("SELECT * FROM eval_datasets WHERE id=? AND tenant_id=?",
                (dataset_id, tenant_id))
    if row is None:
        return None
    ds = dict(row)
    ds["cases"] = db.jload(ds["cases"], [])
    return ds


def parse_jsonl(text: str) -> list[dict]:
    cases = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def evaluate_output(output, evaluators: list[dict]) -> tuple[bool, str]:
    """Apply rule evaluators to a run output. Returns (passed, detail)."""
    if isinstance(output, dict) and "text" in output:
        text = str(output["text"])
    elif isinstance(output, str):
        text = output
    else:
        text = json.dumps(output, ensure_ascii=False)
    details = []
    for ev in evaluators:
        etype = ev.get("type")
        if etype == "contains":
            ok = str(ev.get("value", "")) in text
            details.append(f"contains({ev.get('value')!r})={'ok' if ok else 'FAIL'}")
        elif etype == "not_contains":
            ok = str(ev.get("value", "")) not in text
            details.append(f"not_contains({ev.get('value')!r})={'ok' if ok else 'FAIL'}")
        elif etype == "regex":
            ok = re.search(str(ev.get("pattern", "")), text) is not None
            details.append(f"regex({ev.get('pattern')!r})={'ok' if ok else 'FAIL'}")
        elif etype == "json_schema":
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = output
            errors = validate(parsed, ev.get("schema", {}))
            ok = not errors
            details.append(f"json_schema={'ok' if ok else 'FAIL:' + ';'.join(errors)}")
        else:
            ok = False
            details.append(f"unknown evaluator type {etype!r}")
        if not ok:
            return False, "; ".join(details)
    return True, "; ".join(details)


def run_eval(tenant_id: str, dataset_id: str, agent_id: str,
             version_id: str, config: dict | None = None) -> dict:
    """Synchronously re-run every dataset case against an agent version."""
    config = config or {}
    ds = get_dataset(tenant_id, dataset_id)
    if ds is None:
        raise ValueError("dataset not found")
    version = db.q1("SELECT * FROM agent_versions WHERE id=? AND agent_id=?"
                    " AND tenant_id=?", (version_id, agent_id, tenant_id))
    if version is None:
        raise ValueError("agent version not found")
    eval_id = db.new_id("eval")
    db.x("INSERT INTO eval_runs (id, tenant_id, dataset_id, agent_id,"
         " agent_version_id, status, config, created_at)"
         " VALUES (?,?,?,?,?,?,?,?)",
         (eval_id, tenant_id, dataset_id, agent_id, version_id, "running",
          db.jdump(config), db.now_ts()))
    default_evaluators = config.get("evaluators", [])
    passed = 0
    total_latency = 0
    total_tokens = 0
    for idx, case in enumerate(ds["cases"]):
        evaluators = case.get("evaluators") or default_evaluators or \
            [{"type": "contains", "value": case.get("expected", "")}]
        started = time.time()
        run = engine.create_run(tenant_id, agent_id, case.get("input", {}),
                                version_id=version_id, hitl_mode="auto_approve")
        run = engine.execute_run(run["id"])
        latency_ms = int((time.time() - started) * 1000)
        out_text = None
        if run["status"] == "completed":
            ok, detail = evaluate_output(run["output"], evaluators)
            out_text = run["output"]
        else:
            ok, detail = False, f"run ended with status {run['status']}: {run.get('error')}"
        passed += 1 if ok else 0
        total_latency += latency_ms
        total_tokens += run.get("total_tokens", 0)
        db.x("INSERT INTO eval_case_results (eval_run_id, case_idx, input,"
             " output, passed, latency_ms, tokens, detail)"
             " VALUES (?,?,?,?,?,?,?,?)",
             (eval_id, idx, db.jdump(case.get("input", {})),
              db.jdump(out_text), 1 if ok else 0, latency_ms,
              run.get("total_tokens", 0), detail))
    n = len(ds["cases"])
    metrics = {
        "cases": n,
        "passed": passed,
        "pass_rate": round(passed / n, 4) if n else 0.0,
        "avg_latency_ms": round(total_latency / n, 1) if n else 0,
        "total_tokens": total_tokens,
    }
    db.x("UPDATE eval_runs SET status='completed', metrics=?, ended_at=?"
         " WHERE id=?", (db.jdump(metrics), db.now_ts(), eval_id))
    db.audit(tenant_id, "api", "eval.completed", f"eval:{eval_id}", metrics)
    return get_eval(tenant_id, eval_id)


def get_eval(tenant_id: str, eval_id: str) -> dict | None:
    row = db.q1("SELECT * FROM eval_runs WHERE id=? AND tenant_id=?",
                (eval_id, tenant_id))
    if row is None:
        return None
    ev = dict(row)
    ev["config"] = db.jload(ev["config"], {})
    ev["metrics"] = db.jload(ev["metrics"])
    results = db.q("SELECT * FROM eval_case_results WHERE eval_run_id=?"
                   " ORDER BY case_idx", (eval_id,))
    ev["case_results"] = [
        {**dict(r), "input": db.jload(r["input"], {}),
         "output": db.jload(r["output"])} for r in results]
    return ev


def compare_evals(tenant_id: str, eval_a: str, eval_b: str) -> dict:
    """Version-vs-version comparison: metric deltas + per-case pass flips."""
    a, b = get_eval(tenant_id, eval_a), get_eval(tenant_id, eval_b)
    if a is None or b is None:
        raise ValueError("eval run not found")
    ma, mb = a["metrics"] or {}, b["metrics"] or {}
    flips = []
    ra = {r["case_idx"]: r for r in a["case_results"]}
    rb = {r["case_idx"]: r for r in b["case_results"]}
    for idx in sorted(set(ra) & set(rb)):
        if ra[idx]["passed"] != rb[idx]["passed"]:
            flips.append({"case_idx": idx,
                          "a_passed": bool(ra[idx]["passed"]),
                          "b_passed": bool(rb[idx]["passed"]),
                          "a_detail": ra[idx]["detail"],
                          "b_detail": rb[idx]["detail"]})
    return {
        "a": {"eval_id": eval_a, "version_id": a["agent_version_id"],
              "metrics": ma},
        "b": {"eval_id": eval_b, "version_id": b["agent_version_id"],
              "metrics": mb},
        "delta": {
            "pass_rate": round((mb.get("pass_rate", 0) - ma.get("pass_rate", 0)), 4),
            "avg_latency_ms": round((mb.get("avg_latency_ms", 0) - ma.get("avg_latency_ms", 0)), 1),
            "total_tokens": (mb.get("total_tokens", 0) - ma.get("total_tokens", 0)),
        },
        "case_flips": flips,
    }
