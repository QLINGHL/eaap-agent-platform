"""FR-407: per-tool circuit breaker — consecutive failures pause the tool,
calls are rejected with CIRCUIT_OPEN, and a half-open trial after the
cooldown either closes the circuit (success) or re-opens it (failure)."""
import pytest

from app import tools
from app.tools import BREAKER, ToolError, execute_tool

T = "tenant_a"
TOOL = "switchable"


@pytest.fixture
def clock(monkeypatch):
    """Fake monotonic clock; advance with clock.advance(seconds)."""
    now = [1000.0]

    class Clock:
        def __call__(self):
            return now[0]

        def advance(self, seconds: float):
            now[0] += seconds

    c = Clock()
    monkeypatch.setattr(tools, "_clock", c)
    return c


@pytest.fixture
def flaky(monkeypatch, clock):  # noqa: ARG001
    """A tool whose success/failure we control, with a call counter."""
    monkeypatch.setenv("EAAP_TOOL_BREAKER_THRESHOLD", "3")
    monkeypatch.setenv("EAAP_TOOL_BREAKER_COOLDOWN_SEC", "60")
    tools.TOOLS.pop(TOOL, None)
    state = {"fail": True, "calls": 0}

    @tools.register(TOOL, "Togglable tool for breaker tests.",
                    {"type": "object", "properties": {}, "required": []})
    def _flaky(args: dict) -> dict:  # noqa: ARG001
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("kaboom")
        return {"ok": True}

    BREAKER.reset()
    yield state
    tools.TOOLS.pop(TOOL, None)
    BREAKER.reset()


def _fail(tenant=T):
    with pytest.raises(ToolError) as exc:
        execute_tool(tenant, TOOL, {})
    assert exc.value.code == "TOOL_FAILED"


def _open_circuit(state, clock):
    _fail()
    _fail()
    _fail()  # 3rd consecutive failure -> open
    assert BREAKER.state(T, TOOL)["state"] == "open"
    assert state["calls"] == 3


def test_breaker_opens_after_threshold(flaky, clock):
    _open_circuit(flaky, clock)
    with pytest.raises(ToolError) as exc:
        execute_tool(T, TOOL, {})
    assert exc.value.code == "CIRCUIT_OPEN"
    # the tool fn was never invoked while the circuit is open
    assert flaky["calls"] == 3


def test_success_resets_consecutive_count(flaky, clock):
    _fail()
    _fail()
    flaky["fail"] = False
    assert execute_tool(T, TOOL, {}) == {"ok": True}
    st = BREAKER.state(T, TOOL)
    assert st["state"] == "closed"
    assert st["failures"] == 0
    # two more failures do not open the circuit (counter was reset)
    flaky["fail"] = True
    _fail()
    _fail()
    assert BREAKER.state(T, TOOL)["state"] == "closed"


def test_half_open_trial_success_closes(flaky, clock):
    _open_circuit(flaky, clock)
    clock.advance(61)
    flaky["fail"] = False
    assert execute_tool(T, TOOL, {}) == {"ok": True}
    st = BREAKER.state(T, TOOL)
    assert st["state"] == "closed"
    assert st["failures"] == 0
    assert flaky["calls"] == 4  # trial really invoked the tool


def test_half_open_trial_failure_reopens(flaky, clock):
    _open_circuit(flaky, clock)
    clock.advance(61)
    _fail()  # trial fails -> re-open
    assert flaky["calls"] == 4
    assert BREAKER.state(T, TOOL)["state"] == "open"
    with pytest.raises(ToolError) as exc:
        execute_tool(T, TOOL, {})
    assert exc.value.code == "CIRCUIT_OPEN"
    assert flaky["calls"] == 4  # still blocked


def test_reset_endpoint_reopens_tool(flaky, clock, client, headers_a):
    _open_circuit(flaky, clock)
    resp = client.post(f"/v1/tools/{TOOL}/reset", headers=headers_a)
    assert resp.status_code == 200
    assert resp.json()["breaker"]["state"] == "closed"
    flaky["fail"] = False
    assert execute_tool(T, TOOL, {}) == {"ok": True}


def test_get_tools_exposes_breaker_state(flaky, clock, client, headers_a):
    _open_circuit(flaky, clock)
    resp = client.get("/v1/tools", headers=headers_a)
    entry = next(t for t in resp.json()["tools"] if t["name"] == TOOL)
    assert entry["breaker"]["state"] == "open"
    assert entry["breaker"]["threshold"] == 3


def test_validation_errors_do_not_trip_breaker(flaky, clock):
    # unknown tools and bad arguments are client errors, not tool failures
    for _ in range(6):
        with pytest.raises(ToolError) as exc:
            execute_tool(T, "no_such_tool", {})
        assert exc.value.code == "TOOL_NOT_FOUND"
    for _ in range(6):
        with pytest.raises(ToolError) as exc2:
            execute_tool(T, "calculator", {"expression": 42})
        assert exc2.value.code == "INVALID_ARGS"
    assert BREAKER.state(T, "calculator")["state"] == "closed"


def test_breaker_is_per_tenant(flaky, clock):
    _fail("tenant_a")
    _fail("tenant_a")
    _fail("tenant_a")
    assert BREAKER.state("tenant_a", TOOL)["state"] == "open"
    assert BREAKER.state("tenant_b", TOOL)["state"] == "closed"
    # tenant_b can still call the tool
    flaky["fail"] = False
    assert execute_tool("tenant_b", TOOL, {}) == {"ok": True}
