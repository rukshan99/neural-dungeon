"""SECRET - THE EINSUM OUBLIETTE

Seven operations. One permitted incantation. The trial reads your source and
throws you back in the cell if anything but einsum is called.
"""

import numpy as np
import pytest

from dungeon.scrutiny import is_stub, names_called_in, operators_used_in
from dungeon.trials import load_room

cell = load_room(__file__, "secret_einsum_oubliette")

pytestmark = pytest.mark.secret

rng = np.random.default_rng(42)

SPELLS = ["batched_matmul", "trace_of_each", "diagonal_of_each", "outer", "bilinear", "attention_scores", "row_dot"]


@pytest.mark.parametrize("name", SPELLS)
def test_only_einsum_is_spoken_in_the_cell(name):
    fn = getattr(cell, name)
    if is_stub(fn):
        raise NotImplementedError(f"{name}() is unwritten")
    called = names_called_in(fn) - {"einsum"}
    assert not called, f"{name}() calls {sorted(called)}. In the oubliette only einsum is permitted."
    assert "MatMult" not in operators_used_in(fn), f"{name}() uses @. The walls close in."
    assert "einsum" in names_called_in(fn), f"{name}() does not call einsum at all. Scratch it into the wall."


def test_batched_matmul():
    a, b = rng.standard_normal((3, 4, 5)), rng.standard_normal((3, 5, 2))
    np.testing.assert_allclose(cell.batched_matmul(a, b), a @ b, atol=1e-12)


def test_trace_of_each():
    a = rng.standard_normal((4, 5, 5))
    np.testing.assert_allclose(cell.trace_of_each(a), np.trace(a, axis1=1, axis2=2), atol=1e-12)


def test_diagonal_of_each():
    a = rng.standard_normal((4, 5, 5))
    np.testing.assert_allclose(cell.diagonal_of_each(a), np.diagonal(a, axis1=1, axis2=2))


def test_outer():
    a, b = rng.standard_normal(3), rng.standard_normal(4)
    np.testing.assert_allclose(cell.outer(a, b), np.outer(a, b))


def test_bilinear():
    x, W, y = rng.standard_normal((6, 3)), rng.standard_normal((3, 4)), rng.standard_normal((6, 4))
    expected = np.array([x[i] @ W @ y[i] for i in range(6)])
    np.testing.assert_allclose(cell.bilinear(x, W, y), expected, atol=1e-12)


def test_attention_scores():
    q, k = rng.standard_normal((2, 5, 8)), rng.standard_normal((2, 7, 8))
    np.testing.assert_allclose(cell.attention_scores(q, k), q @ k.transpose(0, 2, 1), atol=1e-12)


def test_row_dot():
    a, b = rng.standard_normal((6, 3)), rng.standard_normal((6, 3))
    np.testing.assert_allclose(cell.row_dot(a, b), (a * b).sum(axis=1), atol=1e-12)
