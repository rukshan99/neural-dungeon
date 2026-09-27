"""ROOM 7.5 - THE VOICE

    The tower can score every possible next character. Scoring is not
    speaking. To speak it must choose, and then choose again with its own
    choice in its ear, one character at a time, until it is told to stop.

FROM LOGITS TO ONE TOKEN. A row of logits z (V,) becomes probabilities with
softmax(z / temperature). Then:

    temperature <= 0   greedy: argmax(z). No randomness at all.
    temperature < 1    sharpens the distribution; > 1 flattens it toward uniform.
    top_k              keep the k largest logits, set the rest to -inf, renormalise.
    top_p (nucleus)    sort by probability, keep the smallest prefix whose
                       cumulative probability reaches p, set the rest to -inf.
                       A token is dropped iff the mass BEFORE it is already >= p,
                       so the most probable token always survives.
    then               one torch.multinomial draw per row, with the generator.

Filters apply to the temperature-scaled logits; -inf becomes probability
exactly 0 after softmax, so a filtered token can never be drawn.

THE LOOP. ``generate`` appends ``max_new_tokens`` tokens to every row:

    for each step:
        idx_cond = idx[:, -block_size:]        the model sees at most block_size tokens
        logits, _ = model(idx_cond)             (B, T, V)
        next_id = sample_next(logits[:, -1, :], **sampling)   only the LAST position predicts
        idx = cat([idx, next_id[:, None]], dim=1)

Every step re-runs the whole prefix; nothing is cached. That is O(T^2) per
token and it is fine here. Floor 12 fixes it.
"""

from __future__ import annotations

import torch


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(B, V) logits -> (B,) int64 next-token ids. See the module docstring for the rules."""
    raise NotImplementedError("sample_next() is unwritten")


def generate(model, idx: torch.Tensor, max_new_tokens: int, **sampling) -> torch.Tensor:
    """(B, T0) -> (B, T0 + max_new_tokens), the prompt kept in front. Forward ``**sampling`` to ``sample_next``.

    Crop the context to ``model.cfg.block_size`` before every forward pass
    and run under ``torch.no_grad()``.
    """
    raise NotImplementedError("generate() is unwritten")
