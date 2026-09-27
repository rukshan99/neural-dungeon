"""ROOM 9.3 - THE SCRIBE'S CHUNKS

    The scribes do not shelve whole books. They copy each one onto cards of a
    fixed size, and the cards are what the echoes come from. Cut the cards too
    small and no card contains a whole thought. Cut them too large and every
    card is about everything, which is the same as being about nothing.

A chunk is the unit of retrieval. It is what gets embedded, what gets returned,
and what the model reads. Three trade-offs decide how to cut:

* Size. Small chunks embed precisely (one topic per vector) but lose context;
  large chunks keep context but average many topics into one vector, and cost
  more prompt tokens per retrieved hit.
* Overlap. Repeating the last ``overlap`` characters of one chunk at the start
  of the next means a sentence that straddles a boundary survives whole in at
  least one chunk. It costs storage and a little duplicate retrieval.
* Boundaries. Cutting at sentence ends keeps thoughts intact at the price of
  uneven chunk sizes.

Two habits matter more than the exact numbers: keep exact character offsets so
every chunk can be traced back to its source (``text[start:end] == chunk.text``,
always), and give every chunk an id that is unique across the whole corpus.
Citations on this floor are chunk ids. A citation that cannot be traced is
worth nothing, which is the boss's whole trick.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A slice of one document. ``text == source_text[start:end]``, always."""

    doc_id: str
    index: int  # 0, 1, 2, ... within the document
    start: int  # character offset, inclusive
    end: int  # character offset, exclusive
    text: str

    @property
    def id(self) -> str:
        """Globally unique when document ids are: ``"p07#2"``."""
        return f"{self.doc_id}#{self.index}"


def chunk_text(text: str, size: int, overlap: int, doc_id: str = "doc") -> list[Chunk]:
    """Fixed-size character chunks with exact offsets.

    Chunk i starts at ``i * (size - overlap)`` and spans ``size`` characters,
    except the last, which ends at ``len(text)`` and may be shorter. Stop as soon
    as a chunk reaches the end of the text (never emit a chunk that is entirely
    inside the previous one). Consecutive chunks therefore overlap by exactly
    ``overlap`` characters. Empty text -> ``[]``. Raise ``ValueError`` if
    ``size <= 0`` or ``overlap >= size`` (the walk would never advance).
    """
    raise NotImplementedError("chunk_text() is unwritten")


def chunk_by_sentences(text: str, max_chars: int, doc_id: str = "doc") -> list[Chunk]:
    """Pack whole sentences into chunks of at most ``max_chars`` characters.

    A sentence ends at ``.``, ``!`` or ``?`` followed by whitespace or the end of
    the text. Pack greedily: append the next sentence to the current chunk if the
    chunk (from the first sentence's first character to the new sentence's last)
    would still be <= ``max_chars``; otherwise start a new chunk. A single
    sentence longer than ``max_chars`` becomes a chunk on its own; never split
    inside a sentence. Chunks carry exact offsets and no leading or trailing
    whitespace.
    """
    raise NotImplementedError("chunk_by_sentences() is unwritten")


def dedupe(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Drop chunks whose *normalised* text was already seen; keep the first, keep order.

    Normalise as lowercase with runs of whitespace collapsed to one space and
    the ends stripped, so "The  Basilisk " and "the basilisk" are duplicates.
    """
    raise NotImplementedError("dedupe() is unwritten")


def chunk_corpus(passages: Sequence, size: int, overlap: int) -> list[Chunk]:
    """``chunk_text`` every passage (objects with ``.id`` and ``.text``), concatenated in order.

    Use each passage's id as ``doc_id`` so every ``Chunk.id`` is unique across
    the corpus. Do not dedupe here; that is a separate, deliberate step.
    """
    raise NotImplementedError("chunk_corpus() is unwritten")
