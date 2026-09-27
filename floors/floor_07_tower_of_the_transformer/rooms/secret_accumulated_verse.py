"""SECRET - THE ACCUMULATED VERSE   (optional)

    Behind the Sovereign's throne, a narrow reading room. The lectern holds
    one page at a time, yet the scribe insists the verse be judged whole.
    So the scribe reads a page, remembers, reads the next, remembers, and
    only speaks when all the pages are in.

Gradient accumulation trains with a batch of k*b examples on hardware that
only fits b. Mean loss over a batch is a mean over examples, so the gradient
of the big batch is the mean of the small batches' gradients:

    grad( mean_loss(big batch) ) = (1/k) * sum_i grad( mean_loss(micro-batch i) )

Since ``backward()`` ADDS into ``.grad``, you get the sum for free by not
zeroing between micro-batches. Scale each micro-batch's loss by 1/k before
backward and the accumulated ``.grad`` is exactly the big-batch gradient.
Step the optimizer once per k micro-batches, then zero.

Forget the 1/k and the gradient is k times too large: the same as multiplying
the learning rate by k. Zero between micro-batches and you are training on
a batch of b after all, with k-1 wasted passes.
"""

from __future__ import annotations


def train_step_accumulated(model, batches, optimizer, accum_steps: int) -> float:
    """Run each (x, y) in ``batches`` through the model, stepping once every ``accum_steps`` micro-batches.

    Each micro-batch's loss is scaled by 1/accum_steps before ``backward()``.
    ``optimizer.step()`` and ``optimizer.zero_grad()`` happen after each
    window of ``accum_steps`` micro-batches, never in between. Raise
    ValueError if ``len(batches)`` is 0 or not a multiple of ``accum_steps``.
    Return the mean of the UNSCALED micro-batch losses as a float.
    """
    raise NotImplementedError("train_step_accumulated() is unwritten")


def count_tokens_seen(batches) -> int:
    """The number of target tokens in ``batches``: the sum of ``y.numel()`` over every (x, y). A Python int."""
    raise NotImplementedError("count_tokens_seen() is unwritten")
