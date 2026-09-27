"""TRIAL 4.1 - THE FIRST TORCH

Autograd is judged: backward, accumulation, detach vs clone, no_grad, the
in-place rules. Then the Prophecy is read aloud and six snippets are run.
"""

import numpy as np
import pytest

from dungeon.scrutiny import is_stub, names_called_in
from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

room = load_room(__file__, "room_1_first_torch")


def _must_call_backward(fn, name):
    if is_stub(fn):
        raise NotImplementedError(f"{name}() is unwritten")
    assert "backward" in names_called_in(fn), (
        f"{name}() never calls backward(). The room is about letting autograd do the differentiating, "
        "not about remembering calculus."
    )


# ---------------------------------------------------------------- grad_of_polynomial
@pytest.mark.parametrize("x", [0.0, 1.0, -2.0, 0.5])
def test_the_first_torch_differentiates_a_polynomial(x):
    _must_call_backward(room.grad_of_polynomial, "grad_of_polynomial")
    got = room.grad_of_polynomial(x)
    expected = 9 * x**2 - 2
    assert isinstance(got, float), f"Return a Python float (.item()), not {type(got).__name__}."
    assert got == pytest.approx(expected), f"dy/dx of 3x^3 - 2x + 1 at x={x} is {expected}; autograd told you {got}."


# ---------------------------------------------------------------- accumulation
def test_gradients_pile_up_when_nobody_clears_them():
    _must_call_backward(room.accumulate_twice, "accumulate_twice")
    got = room.accumulate_twice(3.0)
    assert got == pytest.approx(12.0), (
        f"Two backwards of t**2 at t=3 should leave 2*3 + 2*3 = 12 in .grad; you have {got}. "
        "Rule 1: .grad accumulates. Did you clear it, or only backward once?"
    )


def test_clearing_between_backwards_keeps_only_the_second():
    _must_call_backward(room.zero_then_backward, "zero_then_backward")
    assert room.zero_then_backward(3.0) == pytest.approx(6.0), (
        "After clearing .grad between the two backwards only the second pass remains: 2*3 = 6."
    )
    assert room.zero_then_backward(-1.5) == pytest.approx(-3.0)


# ---------------------------------------------------------------- detach vs clone
def test_the_shadow_shares_storage_and_the_copy_does_not():
    t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    shadow, copy = room.frozen_copy(t)
    assert shadow.data_ptr() == t.data_ptr(), (
        "The shadow must share storage with t: that is detach(). clone() gives new memory."
    )
    assert copy.data_ptr() != t.data_ptr(), "The copy must own its memory: clone() it."
    for name, tensor in (("shadow", shadow), ("copy", copy)):
        assert not tensor.requires_grad, (
            f"{name} still requires grad. A clone() of a tracked tensor is tracked; detach() first."
        )
        assert tensor.grad_fn is None, f"{name} still has a grad_fn: it is still in the graph."
    assert torch.equal(copy, t.detach()), "The copy should hold the same values as t."

    copy.add_(100.0)
    assert torch.equal(t.detach(), torch.tensor([1.0, 2.0, 3.0])), (
        "Writing into the copy changed t. The copy shares memory; clone() it."
    )
    shadow.add_(1.0)
    assert torch.equal(t.detach(), torch.tensor([2.0, 3.0, 4.0])), (
        "Writing into the shadow did NOT change t, so it does not share storage. detach() shares."
    )


# ---------------------------------------------------------------- no_grad
def test_no_grad_eval_builds_no_graph():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    out = room.no_grad_eval(lambda z: (z * 3).sum(), x)
    assert torch.is_tensor(out) and out.item() == pytest.approx(9.0), "no_grad_eval must still return f(x)."
    assert out.grad_fn is None, "The result has a grad_fn: a graph was recorded. Wrap the call in torch.no_grad()."
    assert not out.requires_grad, "The result requires grad; nothing computed under no_grad should."


def test_no_grad_eval_turns_the_light_back_on_afterwards():
    x = torch.tensor([1.0], requires_grad=True)
    room.no_grad_eval(lambda z: z * 2, x)
    assert torch.is_grad_enabled(), (
        "Autograd is still switched off after the call. Use `with torch.no_grad():`, "
        "which restores the previous mode, rather than torch.set_grad_enabled(False)."
    )
    assert (x * 2).requires_grad, "Operations after the call should be tracked again."


# ---------------------------------------------------------------- manual SGD step
def test_manual_sgd_step_updates_the_leaf_in_place():
    w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    ptr = w.data_ptr()
    try:
        room.manual_sgd_step(w, torch.tensor([1.0, 1.0, 1.0]), lr=0.5)
    except RuntimeError as exc:
        pytest.fail(
            f"RuntimeError: {exc}\nAn in-place update of a leaf that requires grad is refused (rule 2) "
            "unless autograd is not watching: do it inside `with torch.no_grad():`."
        )
    assert w.data_ptr() == ptr, (
        "w is a different tensor now. The step must be IN PLACE (w -= ..., w.sub_(...)), "
        "not a fresh tensor assigned to a local name that the caller never sees."
    )
    assert torch.allclose(w.detach(), torch.tensor([0.5, 1.5, 2.5])), f"Expected w - 0.5*grad = [0.5, 1.5, 2.5], got {w.tolist()}."
    assert w.requires_grad and w.is_leaf and w.grad_fn is None, (
        "After the step w must still be a leaf that requires grad with no grad_fn, "
        "exactly like a model weight after optimizer.step()."
    )


# ---------------------------------------------------------------- the Prophecy
SNIPPETS = {
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


def _run(snippet):
    namespace = {"torch": torch}
    try:
        exec(snippet, namespace)  # noqa: S102 - the snippets are ours
    except RuntimeError as exc:
        return "RuntimeError", str(exc).splitlines()[0]
    return namespace["result"], None


def test_the_prophecy_snippets_are_untouched():
    assert dict(room.PROPHECY_SNIPPETS) == SNIPPETS, "Do not edit the snippets; predict them."
    assert set(room.AUTOGRAD_PROPHECY) == set(SNIPPETS), "Do not add or remove prophecies; fill them in."


@pytest.mark.parametrize("name", list(SNIPPETS))
def test_the_prophecy_holds(name):
    prediction = room.AUTOGRAD_PROPHECY.get(name)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied {name!r}. Fill in AUTOGRAD_PROPHECY.")
    actual, message = _run(SNIPPETS[name])
    if actual == "RuntimeError":
        assert prediction == "RuntimeError", (
            f"{name}: you predicted {prediction!r}, but torch raised RuntimeError: {message}"
        )
        return
    assert prediction != "RuntimeError", f"{name}: you predicted a RuntimeError, but it ran fine and produced {actual!r}."
    pred = np.asarray(prediction, dtype=float)
    act = np.asarray(actual, dtype=float)
    assert pred.shape == act.shape and np.allclose(pred, act), (
        f"{name}: the prophecy said {prediction!r}, the machine produced {actual!r}."
    )
