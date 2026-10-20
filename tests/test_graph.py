import pytest

from app.graph import GraphValidationError, validate_graph
from app.seed import CONTRACT_QA_GRAPH


def test_prd_example_graph_is_valid():
    assert validate_graph(CONTRACT_QA_GRAPH) == []


def test_uncontrolled_cycle_rejected():
    spec = {
        "nodes": [{"id": "start", "type": "start"},
                  {"id": "a", "type": "variable_assign",
                   "config": {"assignments": {"x": "1"}}},
                  {"id": "b", "type": "variable_assign",
                   "config": {"assignments": {"y": "2"}}},
                  {"id": "end", "type": "end"}],
        "edges": [{"from": "start", "to": "a"}, {"from": "a", "to": "b"},
                  {"from": "b", "to": "a"}, {"from": "b", "to": "end"}],
    }
    errors = validate_graph(spec)
    assert any("uncontrolled cycle" in e for e in errors)


def test_cycle_through_loop_node_allowed():
    spec = {
        "nodes": [{"id": "start", "type": "start"},
                  {"id": "loop", "type": "loop",
                   "config": {"condition": "vars.i < 3", "max_iterations": 5}},
                  {"id": "inc", "type": "variable_assign",
                   "config": {"assignments": {"i": "= vars.i + 1"}}},
                  {"id": "end", "type": "end"}],
        "edges": [{"from": "start", "to": "loop"},
                  {"from": "loop", "to": "inc", "when": "loop"},
                  {"from": "inc", "to": "loop"},
                  {"from": "loop", "to": "end", "when": "exit"}],
    }
    assert validate_graph(spec) == []


def test_orphan_node_rejected():
    spec = {
        "nodes": [{"id": "start", "type": "start"},
                  {"id": "end", "type": "end"},
                  {"id": "ghost", "type": "llm",
                   "config": {"model": "mock-1", "prompt": "hi"}}],
        "edges": [{"from": "start", "to": "end"}],
    }
    errors = validate_graph(spec)
    assert any("unreachable" in e for e in errors)


def test_missing_required_config_rejected():
    spec = {
        "nodes": [{"id": "start", "type": "start"},
                  {"id": "llm", "type": "llm", "config": {"model": "mock-1"}},
                  {"id": "end", "type": "end"}],
        "edges": [{"from": "start", "to": "llm"}, {"from": "llm", "to": "end"}],
    }
    errors = validate_graph(spec)
    assert any("prompt" in e for e in errors)


def test_loop_without_exit_edge_rejected():
    spec = {
        "nodes": [{"id": "start", "type": "start"},
                  {"id": "loop", "type": "loop",
                   "config": {"condition": "true", "max_iterations": 2}},
                  {"id": "end", "type": "end"}],
        "edges": [{"from": "start", "to": "loop"},
                  {"from": "loop", "to": "end", "when": "loop"}],
    }
    errors = validate_graph(spec)
    assert any("when='loop' and when='exit'" in e for e in errors)


def test_publish_invalid_graph_raises():
    from app import agents as agents_mod
    agent = agents_mod.create_agent("tenant_a", "bad-graph-agent")
    agents_mod.save_draft("tenant_a", agent["id"], {
        "nodes": [{"id": "start", "type": "start"}], "edges": []})
    with pytest.raises(GraphValidationError):
        agents_mod.publish("tenant_a", agent["id"])
