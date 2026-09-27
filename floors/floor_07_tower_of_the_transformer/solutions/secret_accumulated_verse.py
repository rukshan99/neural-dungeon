"""SECRET - THE ACCUMULATED VERSE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Gradient accumulation: k micro-batches, each contributing loss / k, add up
to exactly the gradient of one k-times-larger batch. The optimizer steps
once per window of k.
"""

from __future__ import annotations


def train_step_accumulated(model, batches, optimizer, accum_steps: int) -> float:
    """Run every (x, y) in ``batches`` through the model, stepping the optimizer once per ``accum_steps`` micro-batches.

    Each micro-batch's loss is scaled by 1/accum_steps before backward so the
    accumulated gradient equals the mean-loss gradient of the concatenated
    batch. Gradients are zeroed after each step, never between micro-batches.
    Returns the mean (unscaled) loss over all micro-batches.
    """
    if len(batches) == 0 or len(batches) % accum_steps != 0:
        raise ValueError(f"{len(batches)} micro-batches do not divide into windows of {accum_steps}")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for i, (x, y) in enumerate(batches):
        _, loss = model(x, y)
        (loss / accum_steps).backward()
        total += loss.item()
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return total / len(batches)


def count_tokens_seen(batches) -> int:
    """How many target tokens the model was trained on: the total number of elements in every y."""
    return sum(int(y.numel()) for _, y in batches)
