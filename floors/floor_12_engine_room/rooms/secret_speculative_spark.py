"""SECRET - THE SPECULATIVE SPARK   (optional)

    Behind the Leviathan's trench, a single spark jumps a gap it should not be
    able to jump. It has guessed where the current is going and arrived first.
    When it guesses wrong, it costs nothing but the guess.

Decode steps are cheap in arithmetic and expensive in overhead: one forward pass
per token, whatever the pass computes. Speculative decoding spends the spare
capacity of one pass to check SEVERAL guessed tokens at once:

    1. draft   guess the next k tokens somehow (here: prompt lookup)
    2. verify  run ONE forward over prompt + draft; the logits at positions
               T-1, T, ..., T+k-1 say what greedy would emit after each prefix
    3. accept  the longest prefix of the draft that matches those argmaxes,
               plus one more token: the model's own argmax after the accepted
               prefix (a correction if the draft was wrong, a bonus if it was right)

Because every accepted token is exactly what greedy would have produced, the
output is IDENTICAL to plain greedy decoding. Only the number of forward passes
changes: at least one token per pass, up to k + 1.

Prompt lookup is the cheapest draft model there is: find the most recent earlier
occurrence of the last n tokens in the context and propose the tokens that
followed it. Text repeats (names, phrases, code, quoted passages), and when it
does not, the draft is simply empty and you are back to plain greedy.
"""

from __future__ import annotations

import torch


def propose_from_prompt(ids: list[int], n: int = 3, k: int = 5) -> list[int]:
    """Up to ``k`` tokens that followed the most recent EARLIER occurrence of the last ``n`` tokens.

    Search from the right, but never match the final n-gram against itself. If the
    n-gram appears nowhere else, or there are fewer than n+1 tokens, return []. The
    proposal may be shorter than k when the match sits near the end.
    """
    raise NotImplementedError("propose_from_prompt() is unwritten")


def speculative_generate(model, idx: torch.Tensor, max_new_tokens: int, n: int = 3, k: int = 5):
    """Greedy decoding with prompt-lookup drafts for a single sequence idx (1, T).

    Returns ``(idx_with_new_tokens (1, T + new), stats)`` where stats is
    ``{"forward_passes", "proposed", "accepted", "generated"}``.

    Each round: draft = propose_from_prompt(context); trim it so the round cannot
    overshoot ``max_new_tokens`` or block_size (leave room for the one extra token);
    ONE forward over ``cat(idx, draft)``; ``greedy = logits[0, T-1:].argmax(-1)``;
    accept while ``draft[j] == greedy[j]``; append ``draft[:accepted] + [greedy[accepted]]``.
    Generate exactly ``min(max_new_tokens, block_size - T)`` tokens, token-identical
    to ``model.generate(idx, ..., temperature=0)``. ``@torch.no_grad()``.
    """
    raise NotImplementedError("speculative_generate() is unwritten")
