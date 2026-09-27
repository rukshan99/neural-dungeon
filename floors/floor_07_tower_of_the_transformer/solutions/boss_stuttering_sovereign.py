"""BOSS - THE STUTTERING SOVEREIGN  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The Sovereign's weakness is decoding. Greedy decoding on a small, confident
model falls into loops; raw sampling at high temperature says nothing.
Scoring speech by BOTH diversity (distinct n-grams) and fluency (mean NLL
under the model) rewards the middle ground.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .room_5_the_voice import sample_next


def distinct_ngram_ratio(ids, n: int) -> float:
    """distinct n-grams / total n-grams in a 1-D sequence. 0.0 if there are no n-grams."""
    ids = [int(t) for t in ids]
    total = len(ids) - n + 1
    if total <= 0:
        return 0.0
    grams = {tuple(ids[i : i + n]) for i in range(total)}
    return len(grams) / total


def repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """CTRL-style penalty: for every token already in the row's history,
    divide a positive logit by ``penalty`` and multiply a negative one.

    logits (B, V), generated_ids (B, T) -> new (B, V) tensor. penalty=1 is a no-op.
    """
    out = logits.clone()
    for b in range(out.size(0)):
        seen = torch.unique(generated_ids[b])
        vals = out[b, seen]
        out[b, seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return out


def block_repeated_ngrams(logits: torch.Tensor, generated_ids: torch.Tensor, n: int) -> torch.Tensor:
    """Set to -inf every token that would complete an n-gram already present in the row.

    The row's last n-1 tokens are the prefix. Every earlier occurrence of that
    prefix names one banned continuation. logits (B, V), generated_ids (B, T).
    """
    out = logits.clone()
    for b in range(out.size(0)):
        seq = generated_ids[b].tolist()
        if len(seq) < n:
            continue
        prefix = tuple(seq[len(seq) - (n - 1) :]) if n > 1 else ()
        banned = {seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i : i + n - 1]) == prefix}
        if banned:
            out[b, list(banned)] = float("-inf")
    return out


@torch.no_grad()
def mean_nll(model, ids) -> float:
    """Mean negative log-likelihood per predicted token of ``ids`` under ``model`` (teacher forcing).

    Every token ids[1:] is predicted from the tokens before it, in consecutive
    non-overlapping windows of block_size: window w feeds ids[wB : wB + B] and
    is scored against ids[wB + 1 : wB + B + 1]. Returns sum(NLL) / (len(ids) - 1).
    """
    ids = torch.as_tensor(ids, dtype=torch.long).flatten()
    block = model.cfg.block_size
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(ids) - 1, block):
        y = ids[start + 1 : start + 1 + block]
        x = ids[start : start + len(y)]
        logits, _ = model(x[None])
        total += F.cross_entropy(logits[0], y, reduction="sum").item()
        count += len(y)
    model.train(was_training)
    return total / count


def speak_without_stuttering(model, tokenizer, prompt: str, n_tokens: int = 300, seed: int = 0) -> str:
    """Prompt + n_tokens generated characters, decoded so that it neither loops nor babbles.

    Nucleus sampling at a modest temperature keeps the text fluent; blocking
    repeated 6-grams makes a loop impossible while rarely forcing an unlikely
    character (a repeated 6-gram in 300 characters is nearly always a loop).
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    block_size = model.cfg.block_size
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(n_tokens):
            logits, _ = model(idx[:, -block_size:])
            logits = block_repeated_ngrams(logits[:, -1, :], idx, 6)
            next_id = sample_next(logits, temperature=0.8, top_p=0.9, generator=g)
            idx = torch.cat([idx, next_id[:, None]], dim=1)
    model.train(was_training)
    return tokenizer.decode(idx[0].tolist())


# The prophecy: how each decoding strategy fares against the boss's two judges.
# "stutters" = fails the diversity judge; "babbles" = passes diversity but
# fails fluency; "speaks" = passes both.
SOVEREIGN_PROPHECY: dict[str, str | None] = {
    "greedy": "stutters",
    "temperature_0.7_top_k_40": "speaks",
    "temperature_2.0": "babbles",
    "greedy_no_repeat_4gram": "speaks",
}
