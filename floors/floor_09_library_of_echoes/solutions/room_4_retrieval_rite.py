"""ROOM 9.4 - THE RETRIEVAL RITE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Embed -> search -> prompt -> generate -> resolve citations to chunk ids.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

from dungeon.artifacts.llm import LLM, Message

from .room_3_scribes_chunks import Chunk

SYSTEM_INSTRUCTION = (
    "You are the reader of the Library of Echoes. Answer only from the numbered sources "
    "provided. Cite the source of every claim as [n]. If the sources do not contain the "
    "answer, say that you cannot find it."
)

CANNOT_FIND = "I cannot find that in the library."

Embedder = Callable[[Sequence[str]], np.ndarray]


class Retriever:
    """Chunks in, (chunk, score) pairs out."""

    def __init__(self, embedder: Embedder, index, chunks: Sequence[Chunk]) -> None:
        self.embedder = embedder
        self.index = index
        self.chunks = list(chunks)
        self._by_id = {c.id: c for c in self.chunks}
        if len(self._by_id) != len(self.chunks):
            raise ValueError("chunk ids must be unique: citations are chunk ids")
        if self.chunks:
            self.index.add(self.embedder([c.text for c in self.chunks]), [c.id for c in self.chunks])

    def retrieve(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        qvec = self.embedder([query])[0]
        return [(self._by_id[cid], float(score)) for cid, score in self.index.search(qvec, k)]


def build_prompt(question: str, retrieved: Sequence[tuple[Chunk, float]]) -> list[Message]:
    """System instruction, then numbered sources and the question in one user message."""
    lines = ["Sources:"]
    for n, (chunk, _score) in enumerate(retrieved, start=1):
        lines.append(f"[{n}] {chunk.text}")
    lines.append("")
    lines.append(f"Question: {question}")
    return [Message.system(SYSTEM_INSTRUCTION), Message.user("\n".join(lines))]


_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_citations(answer_text: str) -> list[int]:
    """[n] and [n, m] markers -> ints, first-appearance order, no repeats."""
    seen: list[int] = []
    for m in _MARKER.finditer(answer_text):
        for part in m.group(1).split(","):
            n = int(part)
            if n not in seen:
                seen.append(n)
    return seen


class RAGPipeline:
    """The plain rite: no gate, no verification. The boss adds those."""

    def __init__(self, retriever: Retriever, llm: LLM, k: int = 3) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def answer(self, question: str, k: int | None = None) -> dict:
        retrieved = self.retriever.retrieve(question, k or self.k)
        messages = build_prompt(question, retrieved)
        text = self.llm.complete(messages).text
        sources = [chunk for chunk, _ in retrieved]
        citations = [sources[n - 1].id for n in parse_citations(text) if 1 <= n <= len(sources)]
        return {"answer": text, "citations": citations, "sources": sources}
