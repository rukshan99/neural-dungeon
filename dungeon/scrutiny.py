"""Scrutiny: trials that look at *how* you wrote something, not only what it returns.

Some rooms forbid Python loops ("the bridge collapses under a for") or allow only
one incantation (einsum). These helpers inspect a function's source with ``ast``.
They are deliberately simple and a little strict; that is part of the game.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable, Iterable

LOOP_NODES = (ast.For, ast.While, ast.AsyncFor, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def source_of(func: Callable) -> str:
    return textwrap.dedent(inspect.getsource(func))


def ast_of(func: Callable) -> ast.AST:
    return ast.parse(source_of(func))


def python_loops_in(func: Callable) -> list[str]:
    """Names of loop constructs found in ``func`` (empty list means it is loop-free)."""
    found = []
    for node in ast.walk(ast_of(func)):
        if isinstance(node, LOOP_NODES):
            found.append(type(node).__name__)
    return found


def names_called_in(func: Callable) -> set[str]:
    """Every attribute/function name that is *called* inside ``func``.

    ``np.sum(x)`` -> "sum"; ``x.sum()`` -> "sum"; ``sorted(x)`` -> "sorted".
    """
    called: set[str] = set()
    for node in ast.walk(ast_of(func)):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                called.add(f.attr)
            elif isinstance(f, ast.Name):
                called.add(f.id)
    return called


def attributes_used_in(func: Callable) -> set[str]:
    """Every attribute name accessed inside ``func`` (called or not)."""
    return {n.attr for n in ast.walk(ast_of(func)) if isinstance(n, ast.Attribute)}


def operators_used_in(func: Callable) -> set[str]:
    """Binary operator class names used inside ``func``, e.g. {"MatMult", "Add"}."""
    ops: set[str] = set()
    for node in ast.walk(ast_of(func)):
        if isinstance(node, (ast.BinOp, ast.AugAssign)):
            ops.add(type(node.op).__name__)
    return ops


def forbidden_calls_in(func: Callable, forbidden: Iterable[str]) -> set[str]:
    return names_called_in(func) & set(forbidden)


def is_stub(func: Callable) -> bool:
    """True if the function body still just raises NotImplementedError."""
    tree = ast_of(func)
    fn = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return False
    body = [n for n in fn.body if not isinstance(n, ast.Expr) or not isinstance(getattr(n, "value", None), ast.Constant)]
    return (
        len(body) == 1
        and isinstance(body[0], ast.Raise)
        and body[0].exc is not None
        and "NotImplementedError" in ast.dump(body[0].exc)
    )
