import pytest

from app import agents as agents_mod
from app import db, evals, knowledge
from app.providers import Gateway, RateLimitError, TokenBucket
from app.tools import ToolError, execute_tool

T = "tenant_a"


def test_tool_argument_validation_blocks_bad_args():
    with pytest.raises(ToolError) as exc:
        execute_tool(T, "calculator", {"expression": 42})
    assert exc.value.code == "INVALID_ARGS"
    with pytest.raises(ToolError):
        execute_tool(T, "calculator", {})
    with pytest.raises(ToolError) as exc2:
        execute_tool(T, "no_such_tool", {})
    assert exc2.value.code == "TOOL_NOT_FOUND"


def test_calculator_and_audit_redaction():
    out = execute_tool(T, "calculator", {"expression": "6 * 7"})
    assert out["result"] == 42
    execute_tool(T, "text_stats",
                 {"text": "hello", "api_key": "sk-should-not-leak"})
    rows = db.q("SELECT detail FROM audit_logs WHERE action='tool.called'"
                " ORDER BY id DESC LIMIT 5")
    joined = " ".join(r["detail"] for r in rows)
    assert "sk-should-not-leak" not in joined
    assert "***" in joined


def test_gateway_fallback_chain():
    gw = Gateway()
    res = gw.chat(T, [{"role": "user", "content": "QUESTION: hi"}],
                  model="mock-fail")
    assert res["fallback_used"] is True
    assert res["model"] == "mock-1"
    assert res["usage"]["total_tokens"] > 0


def test_gateway_rate_limit():
    gw = Gateway()
    gw.bucket = TokenBucket(capacity=1, refill_per_sec=0.0)
    gw.chat(T, [{"role": "user", "content": "QUESTION: one"}], model="mock-1")
    with pytest.raises(RateLimitError):
        gw.chat(T, [{"role": "user", "content": "QUESTION: two"}],
                model="mock-1")


QA_GRAPH = {
    "nodes": [
        {"id": "start", "type": "start"},
        {"id": "kb", "type": "knowledge_retrieval",
         "config": {"kb_ids": ["evalkb"], "top_k": 3, "min_score": 0.2}},
        {"id": "llm", "type": "llm",
         "config": {"model": "mock-1",
                    "prompt": "SOURCES:\n{{nodes.kb.context}}\n"
                              "QUESTION: {{input.question}}"}},
        {"id": "end", "type": "end", "config": {"output": "{{nodes.llm.text}}"}},
    ],
    "edges": [{"from": "start", "to": "kb"}, {"from": "kb", "to": "llm"},
              {"from": "llm", "to": "end"}],
}

QA_GRAPH_V2_NO_SOURCES = {
    "nodes": [
        {"id": "start", "type": "start"},
        {"id": "llm", "type": "llm",
         "config": {"model": "mock-1", "prompt": "QUESTION: {{input.question}}"}},
        {"id": "end", "type": "end", "config": {"output": "{{nodes.llm.text}}"}},
    ],
    "edges": [{"from": "start", "to": "llm"}, {"from": "llm", "to": "end"}],
}


def _eval_setup():
    kb = knowledge.find_kb_by_name(T, "evalkb")
    if kb is None:
        kb = knowledge.create_kb(T, "evalkb")
        knowledge.add_document(
            T, kb["id"], "Payment Terms",
            "Payment terms: invoices are due within 30 days of the "
            "invoice date, payable by bank transfer.")
    agent = agents_mod.create_agent(T, "eval-agent")
    agents_mod.save_draft(T, agent["id"], QA_GRAPH)
    v1 = agents_mod.publish(T, agent["id"], "v1 with retrieval")
    ds = evals.create_dataset(T, "eval-baseline", [
        {"input": {"question": "How many days to pay an invoice?"},
         "expected": "30 days"},
        {"input": {"question": "When must invoices be paid?"},
         "expected": "30 days"},
    ])
    return agent["id"], v1["version_id"], ds["id"]


def test_eval_run_metrics_and_compare():
    agent_id, v1_id, ds_id = _eval_setup()
    ev1 = evals.run_eval(T, ds_id, agent_id, v1_id)
    assert ev1["status"] == "completed"
    assert ev1["metrics"]["cases"] == 2
    assert ev1["metrics"]["pass_rate"] == 1.0
    assert ev1["metrics"]["total_tokens"] > 0

    # v2 drops retrieval -> answers lose the sourced fact -> pass rate falls.
    agents_mod.save_draft(T, agent_id, QA_GRAPH_V2_NO_SOURCES)
    v2 = agents_mod.publish(T, agent_id, "v2 without retrieval")
    ev2 = evals.run_eval(T, ds_id, agent_id, v2["version_id"])
    assert ev2["metrics"]["pass_rate"] == 0.0

    cmp_ = evals.compare_evals(T, ev1["id"], ev2["id"])
    assert cmp_["delta"]["pass_rate"] == -1.0
    assert len(cmp_["case_flips"]) == 2
