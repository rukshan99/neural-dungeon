"""ROOM 9.3 - THE SCRIBE'S CHUNKS  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Every chunk carries exact offsets: ``text[start:end] == chunk.text`` is the
invariant the whole floor's citations rest on.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A slice of one document. ``text == source_text[start:end]``, always."""

    doc_id: str
    index: int
    start: int
    end: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.doc_id}#{self.index}"


def chunk_text(text: str, size: int, overlap: int, doc_id: str = "doc") -> list[Chunk]:
    """Fixed-size character windows that advance by ``size - overlap``."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must be in [0, size), got overlap={overlap} size={size}")
    chunks: list[Chunk] = []
    if not text:
        return chunks
    step = size - overlap
    start = 0
    while True:
        end = min(start + size, len(text))
        chunks.append(Chunk(doc_id, len(chunks), start, end, text[start:end]))
        if end >= len(text):
            break
        start += step
    return chunks


# A sentence: anything up to . ! or ? that is followed by whitespace or the end.
_SENTENCE = re.compile(r"\S[^.!?]*?[.!?]+(?=\s|$)|\S[^.!?]*$", re.DOTALL)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every sentence, whitespace excluded, in order."""
    spans = []
    for m in _SENTENCE.finditer(text):
        start, end = m.start(), m.end()
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans


def chunk_by_sentences(text: str, max_chars: int, doc_id: str = "doc") -> list[Chunk]:
    """Greedy packing of whole sentences into chunks of at most ``max_chars``."""
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    chunks: list[Chunk] = []
    current: tuple[int, int] | None = None  # (start, end) of the chunk being built
    for s_start, s_end in _sentence_spans(text):
        if current is None:
            current = (s_start, s_end)
        elif s_end - current[0] <= max_chars:
            current = (current[0], s_end)  # the sentence fits: extend
        else:
            chunks.append(Chunk(doc_id, len(chunks), current[0], current[1], text[current[0] : current[1]]))
            current = (s_start, s_end)  # start fresh (even if this one sentence is too long)
    if current is not None:
        chunks.append(Chunk(doc_id, len(chunks), current[0], current[1], text[current[0] : current[1]]))
    return chunks


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def dedupe(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Keep the first chunk for every normalised text."""
    seen: set[str] = set()
    out: list[Chunk] = []
    for chunk in chunks:
        key = _normalise(chunk.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def chunk_corpus(passages: Sequence, size: int, overlap: int) -> list[Chunk]:
    """chunk_text over every passage, with the passage id as doc_id."""
    out: list[Chunk] = []
    for passage in passages:
        out.extend(chunk_text(passage.text, size, overlap, doc_id=passage.id))
    return out
