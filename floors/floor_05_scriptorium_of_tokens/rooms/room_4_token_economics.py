"""ROOM 5.4 - TOKEN ECONOMICS

    The last desk belongs to the bursar. Vellum is not free. Every token a
    scribe copies costs a coin, every token the reader speaks back costs
    another, and no page holds more than a fixed number of them. The bursar
    has a ledger for all of it.

Tokens are the unit of everything practical about a language model:

    context   a model reads at most N tokens at once; the rest must be cut or chunked
    cost      hosted models are priced per million tokens, input and output separately
    latency   every output token is one more forward pass

Three recipes, all pure Python:

  * Truncation. A conversation that no longer fits the budget loses its OLDEST
    messages first, never the system message (unless told otherwise), and a
    message is never split in half. Order is preserved for what remains.
  * Chunking with overlap. A long document is cut into windows of
    ``max_tokens`` ids that step forward by ``max_tokens - overlap``, so every
    window shares its first ``overlap`` ids with the previous window. Overlap
    keeps a sentence that straddles a boundary intact in at least one window.
    The last window may be shorter.
  * Cost. price_per_million * tokens / 1e6, separately for input and output.

The functions take any tokenizer with ``.encode(text)``; the trial hands you
the Character Codex from room 5.1 among others.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


def count_tokens(text: str, tokenizer) -> int:
    """Number of ids ``tokenizer.encode(text)`` produces. If encode adds specials, they count."""
    raise NotImplementedError("count_tokens() is unwritten")


def estimate_cost(
    n_input: int,
    n_output: int,
    price_in_per_million: float,
    price_out_per_million: float,
) -> float:
    """Price of a call given token counts and per-million prices. Return a float in the same currency."""
    raise NotImplementedError("estimate_cost() is unwritten")


def truncate_messages(
    messages: Sequence[Message],
    budget: int,
    count_fn: Callable[[str], int],
    keep_system: bool = True,
) -> list[Message]:
    """Drop whole messages, oldest first, until sum(count_fn(m["content"])) <= budget.

    * Messages with role "system" are never dropped while ``keep_system`` is True;
      with ``keep_system=False`` they are ordinary messages.
    * Never split or edit a message; never reorder the survivors.
    * Stop as soon as the total fits: do not drop more than necessary.
    * If only protected messages remain and they still exceed the budget, return
      them anyway (the caller decides what to do).
    * Return a new list; do not mutate the input.
    """
    raise NotImplementedError("truncate_messages() is unwritten")


def chunk_by_tokens(ids: Sequence[int], max_tokens: int, overlap: int) -> list[list[int]]:
    """Sliding windows over ``ids``.

    * Every window has exactly ``max_tokens`` ids except possibly the last.
    * Consecutive windows share exactly ``overlap`` ids: window k starts at
      k * (max_tokens - overlap).
    * Every id appears in at least one window; stop as soon as a window reaches the
      end of ``ids`` (never emit a window that adds no new ids).
    * ``[]`` for empty ``ids``. ValueError if ``max_tokens <= 0`` or not ``0 <= overlap < max_tokens``.
    """
    raise NotImplementedError("chunk_by_tokens() is unwritten")


# ---------------------------------------------------------------------------
# THE LEDGER PROPHECY
# How many tokens does each string cost?
#   "char": the Character Codex of room 5.1, built from a text that contains
#           every letter below, with encode() defaults (so <bos> and <eos> are added).
#   "bpe":  room 5.2's encode_bpe with TOY_MERGES, applied in this order.
# Work it out by hand, then fill in the ints.
# ---------------------------------------------------------------------------
TOY_MERGES = [("t", "h"), ("th", "e"), ("the", "</w>"), ("i", "n"), ("in", "g"), ("ing", "</w>")]

TOKEN_PROPHECY: dict[tuple[str, str], int | None] = {
    ("the thing", "char"): None,
    ("the thing", "bpe"): None,
    ("then", "bpe"): None,
    ("singing", "bpe"): None,
    ("thing thing thing", "char"): None,
    ("thing thing thing", "bpe"): None,
}
