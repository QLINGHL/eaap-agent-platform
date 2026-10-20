"""Tool registry and execution (FR-401/404/406/407-lite).

Tools are registered with a JSON-Schema for their arguments. Arguments are
validated BEFORE execution; invalid arguments never reach the tool. Every
call is written to the audit log with sensitive argument values redacted.
"""
import ast
import time

import httpx

from . import db
from .guard import scan_value
from .schemaval import validate


class ToolError(Exception):
    def __init__(self, message: str, code: str = "TOOL_ERROR"):
        super().__init__(message)
        self.code = code


TOOLS: dict[str, dict] = {}


def register(name: str, description: str, schema: dict):
    def deco(fn):
        TOOLS[name] = {"name": name, "description": description,
                       "schema": schema, "fn": fn}
        return fn
    return deco


# ---------------------------------------------------------------- builtins --

@register("current_time", "Current UTC time as ISO-8601 plus unix timestamp.",
          {"type": "object", "properties": {}, "required": []})
def _current_time(args: dict) -> dict:
    ts = time.time()
    return {"iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
            "unix": int(ts)}


@register("calculator", "Evaluate a basic arithmetic expression safely.",
          {"type": "object",
           "properties": {"expression": {"type": "string"}},
           "required": ["expression"]})
def _calculator(args: dict) -> dict:
    expr = args["expression"]
    tree = ast.parse(expr, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
               ast.USub, ast.Load)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ToolError(f"disallowed element in expression: {type(node).__name__}",
                            code="INVALID_ARGS")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            if not isinstance(n.value, (int, float)):
                raise ToolError("only numeric constants allowed", code="INVALID_ARGS")
            return n.value
        if isinstance(n, ast.UnaryOp):
            return -ev(n.operand)
        left, right = ev(n.left), ev(n.right)
        if isinstance(n.op, ast.Add):
            return left + right
        if isinstance(n.op, ast.Sub):
            return left - right
        if isinstance(n.op, ast.Mult):
            return left * right
        if isinstance(n.op, ast.Div):
            return left / right
        if isinstance(n.op, ast.Mod):
            return left % right
        return left ** right

    return {"expression": expr, "result": ev(tree)}


@register("text_stats", "Character/word/line counts and top words of a text.",
          {"type": "object",
           "properties": {"text": {"type": "string"}},
           "required": ["text"]})
def _text_stats(args: dict) -> dict:
    text = args["text"]
    words = text.split()
    freq: dict[str, int] = {}
    for w in words:
        freq[w.lower()] = freq.get(w.lower(), 0) + 1
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:5]
    return {"chars": len(text), "words": len(words),
            "lines": text.count("\n") + 1 if text else 0,
            "top_words": [{"word": w, "count": c} for w, c in top]}


@register("http_fetch", "Fetch a URL and return status + body (truncated).",
          {"type": "object",
           "properties": {"url": {"type": "string"},
                          "timeout": {"type": "number"}},
           "required": ["url"]})
def _http_fetch(args: dict) -> dict:
    timeout = float(args.get("timeout", 15))
    try:
        resp = httpx.get(args["url"], timeout=timeout,
                         follow_redirects=True,
                         headers={"User-Agent": "EAAP/1.0"})
    except httpx.HTTPError as e:
        raise ToolError(f"fetch failed: {e}", code="UPSTREAM_ERROR")
    return {"status": resp.status_code, "url": str(resp.url),
            "body": resp.text[:8000]}


# --------------------------------------------------------------- execution --

def list_tools() -> list[dict]:
    return [{"name": t["name"], "description": t["description"],
             "schema": t["schema"], "source": "builtin"}
            for t in TOOLS.values()]


def execute_tool(tenant_id: str, name: str, args: dict,
                 run_id: str | None = None) -> dict:
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolError(f"unknown tool '{name}'", code="TOOL_NOT_FOUND")
    errors = validate(args, tool["schema"])
    if errors:
        raise ToolError(f"invalid arguments for tool '{name}': {errors}",
                        code="INVALID_ARGS")
    warnings = scan_value(args)
    started = time.time()
    try:
        result = tool["fn"](args)
    except ToolError:
        raise
    except Exception as e:  # noqa: BLE001 - surfaced as tool error
        raise ToolError(f"tool '{name}' failed: {e}", code="TOOL_FAILED")
    latency_ms = int((time.time() - started) * 1000)
    db.audit(tenant_id, "engine", "tool.called", f"tool:{name}",
             {"run_id": run_id, "args": args, "latency_ms": latency_ms,
              "injection_warnings": warnings})
    if warnings:
        result = dict(result) if isinstance(result, dict) else {"result": result}
        result["_injection_warnings"] = warnings
    return result
