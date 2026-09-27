"""ROOM 4.1 - THE FIRST TORCH

    The passage is already lit. Nobody remembers lighting it. On Floors 2 and 3
    you built an autograd engine and an MLP by hand, in the dark, in numpy.
    Down here a library did the same work first, and did it well. The danger
    is no longer that the light goes out. It is that you stop looking at it.

A ``torch.Tensor`` is a numpy array that remembers three extra things: the
*device* it lives on, whether it ``requires_grad``, and, after ``backward()``,
the gradient that flowed into it (``.grad``). Every operation on a tensor that
requires grad is recorded into a graph. ``backward()`` walks that graph and
deposits d(output)/d(leaf) into each leaf's ``.grad``.

Five rules of the light. Each is checked below.

    1. ``.grad`` ACCUMULATES. A second ``backward()`` adds to it. You clear it.
    2. A LEAF that requires grad refuses in-place operations (``x.add_(1)`` is a
       RuntimeError) unless you are inside ``torch.no_grad()``. That exception
       is how every optimizer updates its weights.
    3. ``detach()`` returns a tensor that shares storage but is cut out of the
       graph. ``clone()`` copies the storage and STAYS in the graph.
    4. Inside ``torch.no_grad()`` nothing is recorded: results have no grad_fn.
    5. The graph is freed after ``backward()``. Calling it a second time on the
       same output is a RuntimeError unless you passed ``retain_graph=True``.

Then the Prophecy: six snippets, six predictions. Commit before you run.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def grad_of_polynomial(x: float) -> float:
    """dy/dx at ``x`` for y = 3x^3 - 2x + 1, computed by autograd. Return a Python float.

    Make a leaf tensor from ``x`` with ``requires_grad=True``, build ``y`` from
    it with ordinary arithmetic, call ``y.backward()``, read the leaf's ``.grad``
    and turn it into a float with ``.item()``.

    Do not differentiate by hand: the trial reads your source and expects to see
    ``backward`` spoken. (The answer is 9x^2 - 2; the point is who computes it.)
    """
    raise NotImplementedError("grad_of_polynomial() is unwritten")


def accumulate_twice(x: float) -> float:
    """Backward twice into the same leaf WITHOUT clearing, and return what accumulates.

    Leaf ``t`` holds ``x``. Build ``y = t ** 2`` and call ``y.backward()``. Build
    ``y`` again (a fresh graph) and call ``backward()`` again. Return ``t.grad``
    as a float. Since d(t^2)/dt = 2t, two backwards leave 4x in ``.grad``.

    This is rule 1, and it is why every training loop calls ``zero_grad()``.
    """
    raise NotImplementedError("accumulate_twice() is unwritten")


def zero_then_backward(x: float) -> float:
    """Same as ``accumulate_twice`` but clear ``t.grad`` between the two backwards.

    Either ``t.grad = None`` or ``t.grad.zero_()`` clears it. Return ``t.grad`` as
    a float: 2x, the gradient of the second pass alone.
    """
    raise NotImplementedError("zero_then_backward() is unwritten")


def frozen_copy(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(shadow, copy)``: two ways to get a tensor that autograd ignores.

    * ``shadow`` shares storage with ``t`` (same ``data_ptr()``) but has
      ``requires_grad=False`` and no ``grad_fn``. Writing into it writes into
      ``t``. This is ``detach()``.
    * ``copy`` has its own storage, ``requires_grad=False``, no ``grad_fn``.
      Writing into it leaves ``t`` alone. ``clone()`` alone is not enough: a
      clone of a tracked tensor is still tracked.
    """
    raise NotImplementedError("frozen_copy() is unwritten")


def no_grad_eval(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Call ``f(x)`` with autograd switched off and return the result.

    Even if ``x`` requires grad, the result must have ``grad_fn is None`` and
    ``requires_grad == False``. Autograd must be switched back on afterwards;
    use the context manager, not the global toggle.
    """
    raise NotImplementedError("no_grad_eval() is unwritten")


def manual_sgd_step(w: torch.Tensor, grad: torch.Tensor, lr: float) -> None:
    """One SGD step IN PLACE: ``w <- w - lr * grad``. Return nothing.

    ``w`` is a leaf with ``requires_grad=True``, exactly like a model weight.
    After the step it must still be the same tensor object (same ``data_ptr()``),
    still a leaf, still requiring grad, with no ``grad_fn``. Rule 2 says a plain
    in-place update raises; the exception to rule 2 is the tool you want.
    """
    raise NotImplementedError("manual_sgd_step() is unwritten")


# ---------------------------------------------------------------------------
# THE PROPHECY
# Six snippets. Each ends by assigning ``result``. For each, predict EITHER the
# value of ``result`` (a float for .item(), a list of floats for .tolist()) OR
# the string "RuntimeError" if the snippet never reaches that line.
# Do not run them first. Commit, then let the trial judge you.
# ---------------------------------------------------------------------------
PROPHECY_SNIPPETS: dict[str, str] = {
    "in_place_on_a_leaf": """
x = torch.tensor([1.0, 2.0], requires_grad=True)
x.mul_(2.0)
result = x.tolist()
""",
    "backward_twice": """
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward()
y.backward()
result = x.grad.item()
""",
    "numpy_of_a_tracked_tensor": """
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
result = y.numpy().tolist()
""",
    "in_place_under_no_grad": """
w = torch.tensor([1.0, 2.0], requires_grad=True)
with torch.no_grad():
    w -= 0.5 * torch.tensor([2.0, 4.0])
result = w.tolist()
""",
    "grads_accumulate": """
x = torch.tensor(3.0, requires_grad=True)
(x * x).backward()
(x * x).backward()
result = x.grad.item()
""",
    "detach_cuts_the_chain": """
x = torch.tensor(3.0, requires_grad=True)
y = x * 2
z = y.detach() * x
z.backward()
result = x.grad.item()
""",
}

AUTOGRAD_PROPHECY: dict[str, float | list[float] | str | None] = {
    "in_place_on_a_leaf": None,
    "backward_twice": None,
    "numpy_of_a_tracked_tensor": None,
    "in_place_under_no_grad": None,
    "grads_accumulate": None,
    "detach_cuts_the_chain": None,
}
