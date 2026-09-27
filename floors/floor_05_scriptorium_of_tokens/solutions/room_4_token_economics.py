"""ROOM 5.4 - TOKEN ECONOMICS  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

Message = dict[str, str]


def count_tokens(text: str, tokenizer) -> int:
    """How many ids ``tokenizer.encode(text)`` produces. Specials count: they occupy context."""
    return len(tokenizer.encode(text))


def estimate_cost(
    n_input: int,
    n_output: int,
    price_in_per_million: float,
    price_out_per_million: float,
) -> float:
    """Prices are quoted per million tokens; scale each side and add."""
    return n_input / 1e6 * price_in_per_million + n_output / 1e6 * price_out_per_million


def truncate_messages(
    messages: Sequence[Message],
    budget: int,
    count_fn: Callable[[str], int],
    keep_system: bool = True,
) -> list[Message]:
    """Drop the OLDEST droppable message until the conversation fits the budget."""
    kept = list(messages)

    def total(msgs: list[Message]) -> int:
        return sum(count_fn(m["content"]) for m in msgs)

    while total(kept) > budget:
        droppable = [i for i, m in enumerate(kept) if not (keep_system and m["role"] == "system")]
        if not droppable:
            break  # only protected messages remain; return them and let the caller decide
        del kept[droppable[0]]
    return kept


def chunk_by_tokens(ids: Sequence[int], max_tokens: int, overlap: int) -> list[list[int]]:
    """Sliding windows of ``max_tokens`` ids that step by ``max_tokens - overlap``."""
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError(f"overlap must satisfy 0 <= overlap < max_tokens, got {overlap}")
    ids = list(ids)
    if not ids:
        return []
    step = max_tokens - overlap
    chunks: list[list[int]] = []
    start = 0
    while True:
        chunks.append(ids[start : start + max_tokens])
        if start + max_tokens >= len(ids):
            break  # this window reached the end; a further one would add nothing new
        start += step
    return chunks


# ---------------------------------------------------------------------------
# THE LEDGER PROPHECY  (answers)
# ---------------------------------------------------------------------------
TOY_MERGES = [("t", "h"), ("th", "e"), ("the", "</w>"), ("i", "n"), ("in", "g"), ("ing", "</w>")]

TOKEN_PROPHECY: dict[tuple[str, str], int | None] = {
    ("the thing", "char"): 11,  # 9 characters + <bos> + <eos>
    ("the thing", "bpe"): 3,  # the</w> | th ing</w>
    ("then", "bpe"): 3,  # the n </w>: (the,</w>) never fires because n is in the way
    ("singing", "bpe"): 3,  # s ing ing</w>
    ("thing thing thing", "char"): 19,  # 17 characters + 2 specials
    ("thing thing thing", "bpe"): 6,  # th ing</w>, three times
}
