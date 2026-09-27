"""SECRET - THE SPECULATIVE SPARK  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Prompt-lookup decoding: guess the next k tokens by finding the last n-gram
earlier in the context and copying what followed it, then verify the whole
guess with ONE forward pass. Accept the longest prefix that greedy would have
produced, plus the model's own next token. The output is identical to greedy;
only the number of forward passes changes.
"""

from __future__ import annotations

import torch


def propose_from_prompt(ids: list[int], n: int = 3, k: int = 5) -> list[int]:
    """Up to ``k`` tokens that followed the most recent earlier occurrence of the last ``n`` tokens."""
    if len(ids) <= n:
        return []
    pattern = ids[-n:]
    # Search from the right, but never match the pattern against itself (its start is len - n).
    for start in range(len(ids) - n - 1, -1, -1):
        if ids[start:start + n] == pattern:
            following = ids[start + n:start + n + k]
            return list(following[:k])
    return []


@torch.no_grad()
def speculative_generate(model, idx: torch.Tensor, max_new_tokens: int, n: int = 3, k: int = 5):
    """Greedy decoding with prompt-lookup drafts. Returns (idx (1, T + new), stats dict).

    stats: {"forward_passes", "proposed", "accepted", "generated"}.
    """
    assert idx.shape[0] == 1, "one sequence at a time"
    block_size = model.cfg.block_size
    stats = {"forward_passes": 0, "proposed": 0, "accepted": 0, "generated": 0}
    target = min(max_new_tokens, block_size - idx.shape[1])
    while stats["generated"] < target:
        T = idx.shape[1]
        remaining = target - stats["generated"]
        draft = propose_from_prompt(idx[0].tolist(), n, k)
        draft = draft[: max(0, min(len(draft), remaining - 1, block_size - T - 1))]
        stats["proposed"] += len(draft)
        candidate = idx
        if draft:
            candidate = torch.cat([idx, torch.tensor([draft], dtype=torch.long)], dim=1)
        logits, _ = model(candidate)  # ONE pass scores every draft position at once
        stats["forward_passes"] += 1
        # greedy[j] is what greedy would emit after candidate[:, :T + j]
        greedy = logits[0, T - 1:, :].argmax(dim=-1).tolist()
        accepted = 0
        while accepted < len(draft) and draft[accepted] == greedy[accepted]:
            accepted += 1
        new_tokens = draft[:accepted] + [greedy[accepted]]  # accepted prefix + the corrected/bonus token
        stats["accepted"] += accepted
        stats["generated"] += len(new_tokens)
        idx = torch.cat([idx, torch.tensor([new_tokens], dtype=torch.long)], dim=1)
    return idx, stats
