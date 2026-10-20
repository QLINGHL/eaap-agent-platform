"""Graph DSL validation (FR-102/FR-103).

DSL shape follows PRD §7.2: {"nodes": [{id, type, config}], "edges":
[{from, to, when}]}. Invalid graphs cannot be published.

Cycle rule: a cycle is legal only if it passes through a `loop` node
(the controlled-loop construct with max_iterations + condition).
"""
import ast

NODE_TYPES = {
    "start", "end", "llm", "knowledge_retrieval", "tool", "condition",
    "code", "human_review", "variable_assign", "http_request",
    "parallel", "loop",
}

REQUIRED_CONFIG = {
    "llm": ["model", "prompt"],
    "tool": ["tool"],
    "condition": ["expr"],
    "loop": ["condition", "max_iterations"],
    "code": ["code"],
    "http_request": ["url"],
    "knowledge_retrieval": ["kb_ids"],
    "parallel": ["branches"],
    "variable_assign": ["assignments"],
}

# Node types allowed inside a parallel branch chain.
PARALLEL_BRANCH_TYPES = {
    "llm", "tool", "code", "variable_assign", "http_request",
    "knowledge_retrieval",
}


class GraphValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _sccs(nodes: list[str], adj: dict[str, list[str]]) -> list[list[str]]:
    """Kosaraju's algorithm (iterative)."""
    visited: set[str] = set()
    order: list[str] = []
    for start in nodes:
        if start in visited:
            continue
        stack = [(start, 0)]
        while stack:
            node, idx = stack[-1]
            if idx == 0:
                if node in visited:
                    stack.pop()
                    continue
                visited.add(node)
            if idx < len(adj.get(node, [])):
                stack[-1] = (node, idx + 1)
                nxt = adj[node][idx]
                if nxt not in visited:
                    stack.append((nxt, 0))
            else:
                order.append(node)
                stack.pop()
    radj: dict[str, list[str]] = {n: [] for n in nodes}
    for src, targets in adj.items():
        for t in targets:
            radj[t].append(src)
    comps: list[list[str]] = []
    seen: set[str] = set()
    for node in reversed(order):
        if node in seen:
            continue
        comp, stack = [], [node]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(radj.get(cur, []))
        comps.append(comp)
    return comps


def validate_graph(spec) -> list[str]:
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["graph spec must be an object with 'nodes' and 'edges'"]
    nodes = spec.get("nodes")
    edges = spec.get("edges")
    if not isinstance(nodes, list) or not nodes:
        return ["graph must contain a non-empty 'nodes' list"]
    if not isinstance(edges, list):
        errors.append("graph must contain an 'edges' list")
        edges = []

    by_id: dict[str, dict] = {}
    for n in nodes:
        if not isinstance(n, dict) or not n.get("id"):
            errors.append(f"node missing 'id': {n!r}")
            continue
        if n["id"] in by_id:
            errors.append(f"duplicate node id '{n['id']}'")
        by_id[n["id"]] = n
        ntype = n.get("type")
        if ntype not in NODE_TYPES:
            errors.append(f"node '{n['id']}': unknown type '{ntype}'")
            continue
        cfg = n.get("config") or {}
        for req in REQUIRED_CONFIG.get(ntype, []):
            if req not in cfg:
                errors.append(f"node '{n['id']}' ({ntype}): missing required config '{req}'")
        if ntype == "condition":
            _check_expr(cfg.get("expr", ""), n["id"], errors)
        if ntype == "loop":
            _check_expr(cfg.get("condition", ""), n["id"], errors)
            mi = cfg.get("max_iterations")
            if not isinstance(mi, int) or mi < 1:
                errors.append(f"node '{n['id']}' (loop): max_iterations must be a positive integer")
        if ntype == "knowledge_retrieval":
            kb = cfg.get("kb_ids")
            if not isinstance(kb, list) or not kb:
                errors.append(f"node '{n['id']}': kb_ids must be a non-empty list")
        if ntype == "variable_assign":
            if not isinstance(cfg.get("assignments"), dict) or not cfg.get("assignments"):
                errors.append(f"node '{n['id']}': assignments must be a non-empty object")

    starts = [n for n in by_id.values() if n.get("type") == "start"]
    ends = [n for n in by_id.values() if n.get("type") == "end"]
    if len(starts) != 1:
        errors.append(f"graph must have exactly one start node, found {len(starts)}")
    if not ends:
        errors.append("graph must have at least one end node")

    adj: dict[str, list[str]] = {nid: [] for nid in by_id}
    out_edges: dict[str, list[dict]] = {nid: [] for nid in by_id}
    for e in edges:
        if not isinstance(e, dict) or "from" not in e or "to" not in e:
            errors.append(f"malformed edge: {e!r}")
            continue
        if e["from"] not in by_id:
            errors.append(f"edge from unknown node '{e['from']}'")
            continue
        if e["to"] not in by_id:
            errors.append(f"edge to unknown node '{e['to']}'")
            continue
        adj[e["from"]].append(e["to"])
        out_edges[e["from"]].append(e)
        if "when" in e and isinstance(e["when"], str) and e["when"] not in ("true", "false"):
            _check_expr(e["when"], f"edge {e['from']}->{e['to']}", errors)

    for n in by_id.values():
        if n.get("type") == "end" and out_edges.get(n["id"]):
            errors.append(f"end node '{n['id']}' must not have outgoing edges")

    # Loop nodes need both an iterate edge and an exit edge.
    for n in by_id.values():
        if n.get("type") == "loop":
            whens = {e.get("when") for e in out_edges.get(n["id"], [])}
            if "loop" not in whens or "exit" not in whens:
                errors.append(
                    f"loop node '{n['id']}' needs outgoing edges with when='loop' and when='exit'")

    # Parallel branches: referenced nodes exist, types allowed.
    branch_members: set[str] = set()
    for n in by_id.values():
        if n.get("type") != "parallel":
            continue
        cfg = n.get("config") or {}
        branches = cfg.get("branches")
        if not isinstance(branches, list) or not branches:
            errors.append(f"parallel node '{n['id']}': branches must be a non-empty list")
            continue
        for bi, branch in enumerate(branches):
            if not isinstance(branch, list) or not branch:
                errors.append(f"parallel node '{n['id']}': branch {bi} must be a non-empty list of node ids")
                continue
            for nid in branch:
                branch_members.add(nid)
                target = by_id.get(nid)
                if target is None:
                    errors.append(f"parallel node '{n['id']}': branch references unknown node '{nid}'")
                elif target.get("type") not in PARALLEL_BRANCH_TYPES:
                    errors.append(
                        f"parallel node '{n['id']}': node '{nid}' type "
                        f"'{target.get('type')}' not allowed in a branch")

    # Reachability / orphan detection (parallel branch members count as reached).
    if len(starts) == 1:
        reached: set[str] = set()
        stack = [starts[0]["id"]]
        while stack:
            cur = stack.pop()
            if cur in reached:
                continue
            reached.add(cur)
            stack.extend(adj.get(cur, []))
            node = by_id.get(cur)
            if node and node.get("type") == "parallel":
                for branch in (node.get("config") or {}).get("branches") or []:
                    if isinstance(branch, list):
                        stack.extend(branch)
        for nid in by_id:
            if nid not in reached:
                errors.append(f"node '{nid}' is unreachable from start (orphan)")

    # Non-end, non-branch nodes need an outgoing edge.
    for n in by_id.values():
        nid, ntype = n["id"], n.get("type")
        if ntype in ("end",) or nid in branch_members:
            continue
        if not out_edges.get(nid):
            errors.append(f"node '{nid}' ({ntype}) has no outgoing edge")

    # Cycle detection: every cyclic SCC must contain a loop node.
    for comp in _sccs(list(by_id), adj):
        cyclic = len(comp) > 1 or (len(comp) == 1 and comp[0] in adj.get(comp[0], []))
        if cyclic and not any(by_id[c].get("type") == "loop" for c in comp):
            errors.append(
                f"uncontrolled cycle through nodes {sorted(comp)}: "
                "cycles must pass through a loop node")

    return errors


def _check_expr(expr: str, where: str, errors: list[str]) -> None:
    if not isinstance(expr, str) or not expr.strip():
        errors.append(f"{where}: expression must be a non-empty string")
        return
    try:
        ast.parse(expr, mode="eval")
    except SyntaxError as e:
        errors.append(f"{where}: expression '{expr}' has a syntax error: {e}")


def assert_valid(spec) -> None:
    errors = validate_graph(spec)
    if errors:
        raise GraphValidationError(errors)
