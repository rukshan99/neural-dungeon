"""ROOM 7.5 - THE VOICE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Turning one row of logits into one token (greedy, temperature, top-k,
top-p), and the autoregressive loop that does it again and again.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(B, V) logits -> (B,) int64 token ids.

    temperature <= 0  greedy: the argmax.
    top_k             keep only the k largest logits (others -> -inf).
    top_p             keep the smallest set of most-probable tokens whose
                      cumulative probability reaches p (always at least one).
    Filters are applied to logits / temperature, then softmax, then one
    multinomial draw per row using ``generator``.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1:]  # (B, 1) the k-th largest
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # Drop a token if the mass *before* it already reaches p. The first
        # token has 0 before it, so it always survives.
        remove = (cum - probs) >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, **sampling) -> torch.Tensor:
    """Append ``max_new_tokens`` tokens to every row of ``idx`` (B, T0) -> (B, T0 + max_new_tokens).

    The model can only see ``block_size`` positions, so the context fed to it
    is cropped to the last ``block_size`` tokens each step. Every step re-runs
    the whole (cropped) prefix: no cache yet. That is Floor 12's problem.
    """
    block_size = model.cfg.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        next_id = sample_next(logits[:, -1, :], **sampling)  # (B,)
        idx = torch.cat([idx, next_id[:, None]], dim=1)
    return idx
