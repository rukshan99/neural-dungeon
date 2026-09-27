"""ROOM 4.1 - THE FIRST TORCH  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def grad_of_polynomial(x: float) -> float:
    """dy/dx of y = 3x^3 - 2x + 1 at x, by autograd."""
    t = torch.tensor(float(x), requires_grad=True)
    y = 3 * t**3 - 2 * t + 1
    y.backward()
    return t.grad.item()


def accumulate_twice(x: float) -> float:
    """Two backwards into one leaf without clearing: .grad holds the SUM (4x)."""
    t = torch.tensor(float(x), requires_grad=True)
    (t**2).backward()
    (t**2).backward()  # a fresh graph each time; the .grad they write into is shared
    return t.grad.item()


def zero_then_backward(x: float) -> float:
    """Clear between the two backwards: .grad holds only the second pass (2x)."""
    t = torch.tensor(float(x), requires_grad=True)
    (t**2).backward()
    t.grad = None  # or t.grad.zero_(); None is what optimizer.zero_grad() does by default
    (t**2).backward()
    return t.grad.item()


def frozen_copy(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(shadow, copy): detach shares storage; detach().clone() owns its own."""
    shadow = t.detach()
    copy = t.detach().clone()  # clone() first would keep the graph; detach() first drops it
    return shadow, copy


def no_grad_eval(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Run f with recording switched off; the context manager restores it afterwards."""
    with torch.no_grad():
        return f(x)


def manual_sgd_step(w: torch.Tensor, grad: torch.Tensor, lr: float) -> None:
    """The heart of every optimizer: an in-place update on a leaf, allowed only under no_grad."""
    with torch.no_grad():
        w -= lr * grad  # in place: w stays the same leaf object, autograd is not watching


# The Prophecy. Each snippet ends by assigning ``result``; predict it or "RuntimeError".
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
    # rule 2: a leaf that requires grad refuses in-place ops outside no_grad
    "in_place_on_a_leaf": "RuntimeError",
    # rule 5: the graph (and the saved x for d/dx x^3) is freed by the first backward
    "backward_twice": "RuntimeError",
    # numpy cannot carry a graph; you must detach() first
    "numpy_of_a_tracked_tensor": "RuntimeError",
    # the exception to rule 2: an SGD step. [1 - 0.5*2, 2 - 0.5*4]
    "in_place_under_no_grad": [0.0, 0.0],
    # rule 1: 2*3 + 2*3
    "grads_accumulate": 12.0,
    # y.detach() is a constant 6 as far as autograd knows: dz/dx = 6
    "detach_cuts_the_chain": 6.0,
}
