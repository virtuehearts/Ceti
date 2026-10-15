"""Deterministic math tier. Math is computed in Python — never by a model.

try_calculate(query) -> str | None: returns the formatted result, or
None if the query isn't a math expression. Division by zero returns an
explicit "undefined" message (still deterministic) instead of falling
through to a model that would freelance the answer.

The evaluator is a whitelisted AST walk — never eval(). Only numbers,
arithmetic operators, whitelisted math functions, and whitelisted
constants are allowed. Anything else (attribute access, comprehensions,
lambdas, imports) is rejected.
"""

from __future__ import annotations

import ast
import math
import operator as _op
import re

_MAX_LEN = 200          # longer than this isn't a calculation
_MAX_EXP = 1000         # exponent magnitude cap (10**10**10 must not hang)
_MAX_MAG = 1e308        # intermediate/result magnitude cap

_BINOPS = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
    ast.Div: _op.truediv, ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod, ast.Pow: _op.pow,
}
_UNARYOPS = {ast.UAdd: _op.pos, ast.USub: _op.neg}

_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "floor": math.floor, "ceil": math.ceil,
    "min": min, "max": max, "pow": pow,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_PREFIXES = ("what is ", "what's ", "whats ", "calculate ", "compute ",
             "solve ", "evaluate ", "eval ")

_PERCENT_OF = re.compile(r"([\d.]+)\s*%\s*of\s*([\d.]+)")
_NAME_RE = re.compile(
    r"\b(%s)\b" % "|".join([*sorted(_FUNCS), *sorted(_CONSTS)]))
_ALLOWED_CHARS = re.compile(r"[0-9+\-*/().,% \ta-zA-Z_]+")


def _strip(q: str) -> str:
    q = q.strip()
    if q.startswith("="):
        q = q[1:]
    low = q.lower()
    for p in _PREFIXES:
        if low.startswith(p):
            q = q[len(p):]
            break
    return q.strip().rstrip("?").strip()


def _eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) \
                or not isinstance(node.value, (int, float)):
            raise ValueError("bad constant")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if type(node.op) is ast.Pow:
            for v in (left, right):
                if isinstance(v, (int, float)) and abs(v) > _MAX_EXP:
                    raise OverflowError("exponent too large")
        v = _BINOPS[type(node.op)](left, right)
        if isinstance(v, (int, float)) and abs(v) > _MAX_MAG:
            raise OverflowError("magnitude too large")
        return v
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*(_eval(a) for a in node.args))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError("not a math expression")


def _format(v) -> str:
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return repr(round(v, 10))  # 0.1+0.2 -> "0.3", not "0.3000...4"
    return str(v)


def try_calculate(query: str) -> str | None:
    """Return the computed result, or None if this isn't math."""
    q = _strip(query)
    if not q or len(q) > _MAX_LEN:
        return None
    m = _PERCENT_OF.fullmatch(q)
    if m:
        return _format(float(m.group(1)) / 100 * float(m.group(2)))
    q = q.replace("^", "**")                      # humans mean power
    q = re.sub(r"(?<=\d),(?=\d)", "", q)           # 1,000 -> 1000
    if not _ALLOWED_CHARS.fullmatch(q):
        return None
    if not any(c.isdigit() for c in q) and not _NAME_RE.search(q):
        return None
    try:
        tree = ast.parse(q, mode="eval")
    except SyntaxError:
        return None
    try:
        return _format(_eval(tree))
    except ZeroDivisionError:
        return "undefined (division by zero)"
    except (ValueError, OverflowError, TypeError, ArithmeticError,
            RecursionError, MemoryError):
        return None
