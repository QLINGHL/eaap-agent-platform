from app import agents as agents_mod
from app import db, engine

T = "tenant_a"


def _publish(name: str, graph: dict) -> str:
    agent = agents_mod.create_agent(T, name)
    agents_mod.save_draft(T, agent["id"], graph)
    agents_mod.publish(T, agent["id"])
    return agent["id"]


def _run(agent_id: str, input_data: dict, **kw):
    run = engine.create_run(T, agent_id, input_data, **kw)
    return engine.execute_run(run["id"])


COND_GRAPH = {
    "nodes": [
        {"id": "start", "type": "start"},
        {"id": "llm", "type": "llm",
         "config": {"model": "mock-1", "prompt": "QUESTION: {{input.q}}"}},
        {"id": "check", "type": "condition",
         "config": {"expr": "nodes.llm.confidence < 0.6"}},
        {"id": "low", "type": "end", "config": {"output": "low-confidence"}},
        {"id": "high", "type": "end", "config": {"output": "high-confidence"}},
    ],
    "edges": [
        {"from": "start", "to": "llm"}, {"from": "llm", "to": "check"},
        {"from": "check", "to": "low", "when": "true"},
        {"from": "check", "to": "high", "when": "false"},
    ],
}


def test_condition_branch_low_confidence():
    agent_id = _publish("cond-agent", COND_GRAPH)
    run = _run(agent_id, {"q": "anything without sources"})
    assert run["status"] == "completed"
    assert run["output"] == "low-confidence"


LOOP_GRAPH = {
    "nodes": [
        {"id": "start", "type": "start"},
        {"id": "init", "type": "variable_assign",
         "config": {"assignments": {"counter": "= 0"}}},
        {"id": "loop", "type": "loop",
         "config": {"condition": "vars.counter < 3", "max_iterations": 10}},
        {"id": "inc", "type": "variable_assign",
         "config": {"assignments": {"counter": "= vars.counter + 1"}}},
        {"id": "end", "type": "end",
         "config": {"output": {"counter": "{{vars.counter}}"}}},
    ],
    "edges": [
        {"from": "start", "to": "init"}, {"from": "init", "to": "loop"},
        {"from": "loop", "to": "inc", "when": "loop"},
        {"from": "inc", "to": "loop"},
        {"from": "loop", "to": "end", "when": "exit"},
    ],
}


def test_loop_executes_until_condition_false():
    agent_id = _publish("loop-agent", LOOP_GRAPH)
    run = _run(agent_id, {})
    assert run["status"] == "completed"
    assert run["output"]["counter"] == 3
    trace = engine.get_trace(T, run["id"])
    inc = [e for e in trace["node_executions"] if e["node_id"] == "inc"]
    assert len(inc) == 3
    assert [e["attempt"] for e in inc] == [1, 2, 3]
    keys = [e["idempotency_key"] for e in trace["node_executions"]]
    assert len(keys) == len(set(keys))


PARALLEL_GRAPH = {
    "nodes": [
        {"id": "start", "type": "start"},
        {"id": "par", "type": "parallel",
         "config": {"branches": [["calc"], ["stats"]]}},
        {"id": "calc", "type": "tool",
         "config": {"tool": "calculator", "args": {"expression": "2 + 3"}}},
        {"id": "stats", "type": "tool",
         "config": {"tool": "text_stats", "args": {"text": "hello world hello"}}},
        {"id": "end", "type": "end",
         "config": {"output": {"sum": "{{nodes.calc.result}}",
                               "words": "{{nodes.stats.words}}"}}},
    ],
    "edges": [{"from": "start", "to": "par"}, {"from": "par", "to": "end"}],
}


def test_parallel_branches_both_contribute():
    agent_id = _publish("parallel-agent", PARALLEL_GRAPH)
    run = _run(agent_id, {})
    assert run["status"] == "completed"
    assert run["output"]["sum"] == 5
    assert run["output"]["words"] == 3


CHAIN_GRAPH = {
    "nodes": [
        {"id": "start", "type": "start"},
        {"id": "a", "type": "variable_assign",
         "config": {"assignments": {"x": "= 41"}}},
        {"id": "b", "type": "code",
         "config": {"code": "import json,sys; d=json.load(sys.stdin);"
                            " print(json.dumps({'seen_input': d['input'].get('n')}))"}},
        {"id": "c", "type": "llm",
         "config": {"model": "mock-1", "prompt": "QUESTION: say hi"}},
        {"id": "end", "type": "end",
         "config": {"output": {"x": "{{vars.x}}",
                               "code_result": "{{nodes.b.result}}"}}},
    ],
    "edges": [
        {"from": "start", "to": "a"}, {"from": "a", "to": "b"},
        {"from": "b", "to": "c"}, {"from": "c", "to": "end"},
    ],
}


def test_checkpoint_resume_after_crash():
    agent_id = _publish("crash-agent", CHAIN_GRAPH)
    run = engine.create_run(T, agent_id, {"n": 7})
    crashed = engine.execute_run(run["id"], crash_after=2)
    assert crashed["status"] == "running"  # worker "killed" mid-run
    ck = db.q1("SELECT COUNT(*) AS n FROM checkpoints WHERE run_id=?",
               (run["id"],))
    assert ck["n"] >= 2
    resumed = engine.resume_run(T, run["id"], sync=True)
    assert resumed["status"] == "completed"
    assert resumed["output"]["x"] == 41
    assert resumed["output"]["code_result"]["seen_input"] == 7
    trace = engine.get_trace(T, run["id"])
    by_node = {}
    for e in trace["node_executions"]:
        by_node[e["node_id"]] = by_node.get(e["node_id"], 0) + 1
    # No node executed twice across crash + resume.
    assert all(v == 1 for v in by_node.values()), by_node


HITL_GRAPH = {
    "nodes": [
        {"id": "start", "type": "start"},
        {"id": "llm", "type": "llm",
         "config": {"model": "mock-1", "prompt": "QUESTION: {{input.q}}"}},
        {"id": "human", "type": "human_review",
         "config": {"role": "ops", "message": "Review: {{nodes.llm.text}}"}},
        {"id": "ok", "type": "end", "config": {"output": "approved-path"}},
        {"id": "no", "type": "end", "config": {"output": "rejected-path"}},
    ],
    "edges": [
        {"from": "start", "to": "llm"}, {"from": "llm", "to": "human"},
        {"from": "human", "to": "ok", "when": "approved"},
        {"from": "human", "to": "no", "when": "rejected"},
    ],
}


def test_hitl_rejection_drives_branch():
    agent_id = _publish("hitl-agent", HITL_GRAPH)
    run = engine.create_run(T, agent_id, {"q": "send the refund email"})
    run = engine.execute_run(run["id"])
    assert run["status"] == "waiting_approval"
    approval = db.q1("SELECT * FROM approvals WHERE run_id=?", (run["id"],))
    assert approval["status"] == "pending"
    assert approval["role"] == "ops"
    engine.decide_approval(T, approval["id"], "rejected", comment="tone is off")
    run = engine.resume_run(T, run["id"], sync=True)
    assert run["status"] == "completed"
    assert run["output"] == "rejected-path"


def test_cancel_queued_run():
    agent_id = _publish("cancel-agent", CHAIN_GRAPH)
    run = engine.create_run(T, agent_id, {})
    cancelled = engine.cancel_run(T, run["id"])
    assert cancelled["status"] == "cancelled"
