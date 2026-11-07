"""Boundary tests for the graph-DSL expression evaluator (app/expr.py).

Covers the three edges that used to leak raw Python exceptions out of
``evaluate`` (which the engine only catches as ``ExprError``):
division by zero, deeply nested expressions, and operand type errors.
"""
import pytest

from app.expr import ExprError, evaluate, render_str, resolve_path

CTX = {
    "input": {"n": 10, "name": "ada", "items": [3, 1, 4], "flag": True},
    "nodes": {"llm": {"confidence": 0.42, "text": "hello"}},
    "vars": {"counter": 3, "ratio": 0.5},
}


# --- division by zero: must surface as ExprError, never ZeroDivisionError ---

@pytest.mark.parametrize("expr", [
    "1 / 0",
    "1 % 0",
    "input.n / (vars.counter - 3)",
    "input.n % (vars.counter - 3)",
    "10 / (2 - 2)",
    "(1 + 2) / (input.n - 10)",
])
def test_division_by_zero_is_expr_error(expr):
    with pytest.raises(ExprError, match="division by zero"):
        evaluate(expr, CTX)


def test_zero_divisor_in_nested_expression():
    # The error must survive nesting instead of bubbling up raw.
    with pytest.raises(ExprError, match="division by zero"):
        evaluate("1 + (10 / (5 - 5)) * 2", CTX)


def test_division_by_nonzero_still_works():
    assert evaluate("input.n / vars.counter", CTX) == pytest.approx(10 / 3)
    assert evaluate("input.n % vars.counter", CTX) == 1
    assert evaluate("10 / 4", CTX) == 2.5


# --- nested expressions: deep parens, chained comparisons, nested logic ---

def test_deeply_nested_arithmetic():
    assert evaluate("((1 + 2) * (3 - 4)) / (2 + 3)", CTX) == pytest.approx(-0.6)
    assert evaluate("((((1 + 1) + 1) + 1) + 1) * 2", CTX) == 10
    # 30 levels of redundant parens must not blow the stack or change value.
    assert evaluate("(" * 30 + "input.n + 1" + ")" * 30, CTX) == 11


def test_nested_boolean_logic():
    assert evaluate("not (input.flag and (vars.counter > 5 or nodes.llm.confidence < 0.5))", CTX) is False
    assert evaluate("(input.n > 5 and input.n < 20) or vars.counter == 99", CTX) is True
    assert evaluate("not not input.flag", CTX) is True


def test_chained_comparison():
    assert evaluate("1 < input.n < 20", CTX) is True
    assert evaluate("1 < input.n < 5", CTX) is False
    assert evaluate("vars.counter == 3 == 3", CTX) is True
    assert evaluate("0 <= vars.counter <= 3", CTX) is True


def test_nested_containers_and_paths():
    assert evaluate("[1, [2, [3, 4]], input.n]", CTX) == [1, [2, [3, 4]], 10]
    assert evaluate("(input.n, vars.counter)", CTX) == (10, 3)
    assert evaluate('{"k": input.n, "nested": {"c": vars.counter}}', CTX) == {
        "k": 10, "nested": {"c": 3}}
    assert evaluate('{"a": 1, "b": 2} == {"b": 2, "a": 1}', CTX) is True
    # Dotted paths resolve through dicts; numeric list indices are only
    # reachable through {{template}} paths (``input.items.0`` is not
    # valid Python syntax, so it cannot appear inside an expression).
    assert evaluate("nodes.llm.text == 'hello'", CTX) is True
    assert render_str("{{input.items.0}} + {{input.items.2}}", CTX) == "3 + 4"


def test_dict_unpacking_is_rejected():
    with pytest.raises(ExprError, match="dict unpacking"):
        evaluate("{**input}", CTX)


# --- type errors: must surface as ExprError, never raw TypeError ---

@pytest.mark.parametrize("expr", [
    '"a" - 1',          # str minus int
    '1 + "a"',          # int plus str
    'input.name - input.n',
    '"a" * "b"',        # str times str
    'None + 1',
    '[1, 2] - 1',       # list minus int
    '{"a": 1} + 1',     # dict plus int
    '-input.name',      # unary minus on str
    'input.n < "a"',    # int compared with str
    'None < 1',
    'input.name > input.n',
])
def test_type_error_is_expr_error(expr):
    with pytest.raises(ExprError):
        evaluate(expr, CTX)


def test_type_error_message_names_the_problem():
    with pytest.raises(ExprError, match="bad operand types"):
        evaluate('"a" - 1', CTX)
    with pytest.raises(ExprError, match="cannot compare"):
        evaluate('input.n < "a"', CTX)


def test_legit_mixed_type_ops_still_work():
    assert evaluate('"a" + "b"', CTX) == "ab"
    assert evaluate('"ab" * 2', CTX) == "abab"
    assert evaluate("[1] + [2]", CTX) == [1, 2]
    assert evaluate("input.n + 0.5", CTX) == 10.5


# --- resolution and syntax boundaries ---

def test_unknown_name_and_missing_path():
    with pytest.raises(ExprError, match="unknown name"):
        evaluate("nope + 1", CTX)
    with pytest.raises(ExprError, match="cannot resolve"):
        evaluate("nodes.missing.confidence > 0", CTX)
    # Numeric list indices are only valid inside {{templates}}; there the
    # out-of-range index must still surface as a resolution error.
    with pytest.raises(ExprError, match="cannot resolve"):
        render_str("{{input.items.9}}", CTX)


def test_disallowed_syntax_still_rejected():
    with pytest.raises(ExprError, match="disallowed syntax"):
        evaluate("2 ** 3", CTX)  # pow is not whitelisted
    with pytest.raises(ExprError, match="disallowed syntax"):
        evaluate('len(input.name)', CTX)  # no calls at all
    with pytest.raises(ExprError, match="subscript access not supported"):
        evaluate('input.items[0]', CTX)  # subscript access is not supported
    with pytest.raises(ExprError):
        evaluate("__import__('os')", CTX)


def test_empty_and_malformed_expressions():
    with pytest.raises(ExprError, match="syntax error"):
        evaluate("", CTX)
    with pytest.raises(ExprError, match="syntax error"):
        evaluate("   ", CTX)
    with pytest.raises(ExprError, match="syntax error"):
        evaluate("input.n +", CTX)


def test_literals_and_keywords():
    assert evaluate("true", CTX) is True
    assert evaluate("false", CTX) is False
    assert evaluate("null", CTX) is None
    assert evaluate("1 in [1, 2, 3]", CTX) is True
    assert evaluate("'ada' in input.name", CTX) is True
    assert evaluate("5 not in [1, 2]", CTX) is True


def test_render_str_with_zero_division_expression_unchanged():
    # render_str only resolves {{path}} templates; expressions are not
    # evaluated there, so a template naming a missing key still raises.
    assert render_str("n={{input.n}}", CTX) == "n=10"
    with pytest.raises(ExprError, match="cannot resolve"):
        render_str("{{input.missing}}", CTX)
    assert resolve_path("input.items.2", CTX) == 4
