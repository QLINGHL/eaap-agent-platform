"""Safe expression evaluation and {{template}} rendering for graph DSL.

Expressions are evaluated against a context dict with roots:
  input  — the run input
  nodes  — outputs of already-executed nodes, by node id
  vars   — variables set by variable_assign nodes

Only a whitelisted subset of Python syntax is allowed (comparisons, boolean
logic, arithmetic, attribute/subscript access, literals). No calls, no imports.
"""
import ast
import json
import re


class ExprError(Exception):
    pass


def resolve_path(path: str, ctx: dict):
    """Resolve 'nodes.llm.confidence' or 'input.question' against ctx."""
    parts = path.strip().split(".")
    cur = ctx
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and p.isdigit() and int(p) < len(cur):
            cur = cur[int(p)]
        else:
            raise ExprError(f"cannot resolve '{path}' (failed at '{p}')")
    return cur


_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.Compare, ast.BinOp,
    ast.Attribute, ast.Name, ast.Constant, ast.Subscript, ast.Load,
    ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
    ast.List, ast.Tuple, ast.Dict,
)


def _root_name(node) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _eval(node, ctx: dict):
    if isinstance(node, ast.Expression):
        return _eval(node.body, ctx)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        # Rebuild the dotted path and resolve it against the context.
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Subscript):  # only simple chains supported
            raise ExprError("subscript access not supported in expressions; use dotted paths")
        if not isinstance(cur, ast.Name):
            raise ExprError("unsupported expression element")
        parts.append(cur.id)
        path = ".".join(reversed(parts))
        if path in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[path]
        return resolve_path(path, ctx)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = _eval(v, ctx)
                if not result:
                    return result
            return result
        result = False
        for v in node.values:
            result = _eval(v, ctx)
            if result:
                return result
        return result
    if isinstance(node, ast.UnaryOp):
        val = _eval(node.operand, ctx)
        if isinstance(node.op, ast.Not):
            return not val
        if isinstance(node.op, ast.USub):
            return -val
        return +val
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left, ctx), _eval(node.right, ctx)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise ExprError("unsupported operator")
    if isinstance(node, ast.Compare):
        left = _eval(node.left, ctx)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, ctx)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            else:
                raise ExprError("unsupported comparison")
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.List):
        return [_eval(e, ctx) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval(e, ctx) for e in node.elts)
    raise ExprError(f"unsupported syntax: {type(node).__name__}")


def evaluate(expr: str, ctx: dict):
    """Evaluate a whitelisted expression string against the context."""
    expr = expr.strip()
    if expr == "true":
        return True
    if expr == "false":
        return False
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"syntax error in '{expr}': {e}")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExprError(f"disallowed syntax in '{expr}': {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in ctx and node.id not in ("true", "false", "null"):
            raise ExprError(f"unknown name '{node.id}' in '{expr}'")
    return _eval(tree, ctx)


_TEMPLATE_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


def render_str(s: str, ctx: dict):
    """Render {{path}} templates. A string that is exactly one template
    returns the raw resolved value (preserving type); otherwise the values
    are stringified into the text."""
    matches = list(_TEMPLATE_RE.finditer(s))
    if not matches:
        return s
    if len(matches) == 1 and matches[0].span() == (0, len(s)):
        return resolve_path(matches[0].group(1), ctx)

    def _sub(m):
        val = resolve_path(m.group(1), ctx)
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        return "" if val is None else str(val)

    return _TEMPLATE_RE.sub(_sub, s)


def render(value, ctx: dict):
    """Recursively render templates in strings/dicts/lists."""
    if isinstance(value, str):
        return render_str(value, ctx)
    if isinstance(value, dict):
        return {k: render(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, ctx) for v in value]
    return value
