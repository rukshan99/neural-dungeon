"""SECRET - THE FUSED WHISPER   (optional)

    Behind the Wraith's lair, a short corridor where six whisperers used to
    stand. Someone replaced them with one. The message arrives faster, and,
    strangely, it never overflows.

Softmax followed by cross-entropy is the loss of every classifier on the floors
below. Composed from primitive ops it is six graph nodes (subtract max, exp,
sum, divide, log, gather) and each one stores an intermediate array. Fused, it
is ONE node whose backward is a closed form you can write on a napkin:

    p = softmax(z)                      loss = -(1/N) sum_i log p[i, y_i]
    d loss / d z = (p - onehot(y)) / N

Two things the trial checks that a composed version gets wrong or gets slowly:

  * Stability. Compute log-softmax as  (z - max) - log(sum(exp(z - max))).
    Never exponentiate raw logits: exp(1000) is inf and inf/inf is NaN.
  * Fusion. The output Tensor's ``_prev`` must be exactly ``{logits}``: one node,
    one backward closure, no intermediate Tensors.

This is how real frameworks implement it (PyTorch's ``cross_entropy`` is a fused
kernel), and it is why they tell you to pass logits, not probabilities.
"""

from __future__ import annotations

import numpy as np

from .room_3_tensor_whisper import Tensor


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy between softmax(logits) and integer class targets, as ONE graph node.

    logits:  Tensor of shape (N, C).
    targets: integer array of shape (N,), each in [0, C).
    Returns a 0-d Tensor with ``_prev == {logits}`` whose backward adds
    ``(softmax - onehot) / N * out.grad`` to ``logits.grad``.

    Build it the way every op in Room 2.3 is built: compute the forward with
    numpy on ``logits.data``, create ``Tensor(loss, (logits,), "softmax_xent")``,
    attach a ``_backward`` closure, return it.
    """
    raise NotImplementedError("softmax_cross_entropy() is unwritten")
