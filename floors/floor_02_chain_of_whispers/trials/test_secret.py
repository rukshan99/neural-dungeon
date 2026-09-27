"""SECRET - THE FUSED WHISPER

One node must do the work of six: match the composed version, stay finite on
enormous logits, and pass a gradient check.
"""

import numpy as np
import pytest

from dungeon.trials import load_room

cell = load_room(__file__, "secret_fused_whisper")
Tensor = cell.Tensor

pytestmark = pytest.mark.secret

rng = np.random.default_rng(99)


def _stable_softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _reference_loss(z, targets):
    shifted = z - z.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(targets)), targets].mean())


def _composed(logits: Tensor, targets) -> Tensor:
    """The six-node version, built from Room 2.3 primitives (max-shift done on the data)."""
    n, c = logits.shape
    shift = Tensor(logits.data.max(axis=1, keepdims=True))  # a constant: no gradient flows into it
    e = (logits - shift).exp()
    probs = e / e.sum(axis=1, keepdims=True)
    onehot = np.zeros((n, c))
    onehot[np.arange(n), targets] = 1.0
    return (probs.log() * Tensor(onehot)).sum() * (-1.0 / n)


def test_the_fused_whisper_is_a_single_node():
    logits = Tensor(rng.standard_normal((4, 3)))
    out = cell.softmax_cross_entropy(logits, np.array([0, 2, 1, 1]))
    assert isinstance(out, Tensor), f"return a Tensor, got {type(out).__name__}"
    assert out.shape == (), f"the loss is a scalar: shape () expected, got {out.shape}"
    assert out._prev == {logits}, (
        f"a FUSED op has exactly one child, the logits; yours has {len(out._prev)} "
        "(intermediate Tensors were created). Compute the forward with numpy on logits.data."
    )


def test_the_fused_whisper_matches_the_composed_version():
    z = rng.standard_normal((6, 5)) * 2.0
    targets = rng.integers(0, 5, size=6)
    fused_in, composed_in = Tensor(z), Tensor(z)
    fused = cell.softmax_cross_entropy(fused_in, targets)
    composed = _composed(composed_in, targets)
    assert np.isclose(fused.data, composed.data, rtol=1e-10), (
        f"fused loss {float(fused.data):.8f} differs from the composed loss {float(composed.data):.8f}"
    )
    fused.backward()
    composed.backward()
    np.testing.assert_allclose(fused_in.grad, composed_in.grad, rtol=1e-8, atol=1e-12, err_msg=(
        "the fused backward should give exactly the gradient the six-node version gives"
    ))


def test_the_gradient_is_softmax_minus_onehot_over_n():
    z = rng.standard_normal((5, 4))
    targets = np.array([3, 0, 0, 2, 1])
    logits = Tensor(z)
    cell.softmax_cross_entropy(logits, targets).backward()
    expected = _stable_softmax(z)
    expected[np.arange(5), targets] -= 1.0
    expected /= 5
    np.testing.assert_allclose(logits.grad, expected, rtol=1e-10, atol=1e-12, err_msg=(
        "d loss / d logits should be (softmax - onehot) / N"
    ))
    assert np.allclose(logits.grad.sum(axis=1), 0.0, atol=1e-12), (
        "each row of the gradient sums to zero: adding a constant to a row of logits changes nothing"
    )


def test_the_upstream_gradient_is_respected():
    z = rng.standard_normal((3, 4))
    targets = np.array([1, 2, 0])
    logits = Tensor(z)
    (cell.softmax_cross_entropy(logits, targets) * 3.0).backward()
    expected = _stable_softmax(z)
    expected[np.arange(3), targets] -= 1.0
    np.testing.assert_allclose(logits.grad, 3.0 * expected / 3, rtol=1e-10, err_msg=(
        "the backward must multiply by out.grad: the loss may not be the root of the graph"
    ))


def test_enormous_logits_do_not_overflow():
    z = np.array([[1000.0, 1000.0, 1000.0, 1000.0], [-1000.0, 0.0, 1000.0, 500.0], [3000.0, -3000.0, 0.0, 1.0]])
    targets = np.array([0, 2, 0])
    logits = Tensor(z)
    out = cell.softmax_cross_entropy(logits, targets)
    assert np.isfinite(out.data), f"loss is {out.data}: exp(1000) overflowed. Subtract the row max before exponentiating."
    assert np.isclose(out.data, _reference_loss(z, targets), rtol=1e-12), f"loss should be {_reference_loss(z, targets):.6f}, got {float(out.data):.6f}"
    out.backward()
    assert np.all(np.isfinite(logits.grad)), "the gradient contains inf or NaN"
    # Row 0: all logits equal -> softmax is uniform -> gradient (1/4 - onehot)/3.
    np.testing.assert_allclose(logits.grad[0], (np.full(4, 0.25) - np.array([1, 0, 0, 0])) / 3, atol=1e-12)


def test_the_gradient_check_holds_even_on_large_logits():
    z = rng.standard_normal((4, 5)) * 300.0
    targets = rng.integers(0, 5, size=4)
    logits = Tensor(z)
    cell.softmax_cross_entropy(logits, targets).backward()
    numeric = np.zeros_like(z)
    h = 1e-5
    for idx in np.ndindex(*z.shape):
        old = z[idx]
        z[idx] = old + h
        up = _reference_loss(z, targets)
        z[idx] = old - h
        down = _reference_loss(z, targets)
        z[idx] = old
        numeric[idx] = (up - down) / (2 * h)
    np.testing.assert_allclose(logits.grad, numeric, rtol=1e-4, atol=1e-6, err_msg=(
        "the analytic gradient disagrees with finite differences on logits of size ~300"
    ))


def test_the_whisper_flows_back_through_a_linear_layer():
    X = rng.standard_normal((6, 3))
    W = Tensor(rng.standard_normal((3, 4)) * 0.5)
    targets = rng.integers(0, 4, size=6)
    loss = cell.softmax_cross_entropy(Tensor(X) @ W, targets)
    loss.backward()
    numeric = np.zeros_like(W.data)
    h = 1e-6
    for idx in np.ndindex(*W.data.shape):
        old = W.data[idx]
        W.data[idx] = old + h
        up = _reference_loss(X @ W.data, targets)
        W.data[idx] = old - h
        down = _reference_loss(X @ W.data, targets)
        W.data[idx] = old
        numeric[idx] = (up - down) / (2 * h)
    np.testing.assert_allclose(W.grad, numeric, rtol=1e-5, atol=1e-7, err_msg=(
        "d loss / d W through logits = X @ W disagrees with finite differences"
    ))
