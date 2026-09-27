"""SECRET - THE FUSED WHISPER  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

One node instead of six. The forward is the numerically stable log-softmax
(subtract the row max, then log-sum-exp). The backward is the famous closed
form: d loss / d logits = (softmax - onehot) / N.
"""

from __future__ import annotations

import numpy as np

from .room_3_tensor_whisper import Tensor


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy between softmax(logits) and integer class targets, as ONE graph node.

    logits: Tensor of shape (N, C). targets: int array of shape (N,) with values in [0, C).
    Returns a scalar Tensor whose only child is ``logits``.
    """
    z = logits.data
    targets = np.asarray(targets, dtype=np.int64)
    n = z.shape[0]
    rows = np.arange(n)

    # Stable log-softmax: shifting by the row max changes nothing mathematically
    # (the shift cancels in log(sum(exp)) - z) and stops exp() from overflowing.
    shifted = z - z.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_sum_exp  # (N, C)
    loss = -log_probs[rows, targets].mean()

    out = Tensor(loss, (logits,), "softmax_xent")

    def _backward() -> None:
        # softmax - onehot, divided by N because the loss is a mean.
        probs = np.exp(log_probs)
        probs[rows, targets] -= 1.0
        logits.grad += probs / n * out.grad

    out._backward = _backward
    return out
