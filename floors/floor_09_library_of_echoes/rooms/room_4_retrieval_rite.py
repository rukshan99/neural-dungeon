"""ROOM 9.4 - THE RETRIEVAL RITE

    Question in, sources out, answer back, with footnotes. The rite has four
    steps, and every one of them is a place to be wrong quietly.

Retrieval-augmented generation (RAG), step by step:

    1. EMBED the question with the same embedder that embedded the chunks.
       (A different embedder gives vectors in a different space; the cosine
       between them means nothing.)
    2. SEARCH the index for the k nearest chunks. k is a budget: more sources,
       more chance the answer is in there, more tokens, more noise.
    3. BUILD a prompt that gives the model ONLY those sources, numbered, with
       three instructions: answer only from them, cite as [n], and say when the
       answer is not there.
    4. GENERATE, then PARSE the [n] markers out of the answer and resolve them
       to chunk ids. The chunk id is the citation; "[2]" alone means nothing
       once the prompt is gone.

Every trial on this floor talks to a mock model from ``dungeon.artifacts.llm``.
The one in this trial is a faithful librarian: it answers a question only when
the passage that contains the answer is in its prompt, and otherwise says it
cannot find it. If your retrieval is off, the rite fails at step 2 and no amount
of prompt wording fixes it.
"""

from __future__ import annotations

import re  # noqa: F401  (parse_citations wants it)
from collections.abc import Callable, Sequence

import numpy as np

from dungeon.artifacts.llm import LLM, Message

from .room_3_scribes_chunks import Chunk

# The instruction every prompt on this floor starts with. Keep the three parts.
SYSTEM_INSTRUCTION = (
    "You are the reader of the Library of Echoes. Answer only from the numbered sources "
    "provided. Cite the source of every claim as [n]. If the sources do not contain the "
    "answer, say that you cannot find it."
)

# What a well-behaved model says when the sources do not hold the answer.
CANNOT_FIND = "I cannot find that in the library."

Embedder = Callable[[Sequence[str]], np.ndarray]


class Retriever:
    """Embeds chunks into an index and turns a question into ``[(Chunk, score), ...]``.

    ``Retriever(embedder, index, chunks)``: ``embedder(list_of_texts) -> (N, D)``;
    ``index`` has ``add(vectors, ids)`` and ``search(query_vec, k)`` (your
    ``FlatIndex`` from room 2 will do); ``chunks`` is a sequence of ``Chunk``.
    The constructor embeds every chunk's text and adds them with ``chunk.id``
    as the id, and remembers ``id -> Chunk`` for the way back.

    ``retrieve(query, k)`` embeds the query, searches, and returns the chunks
    with their scores, best first.
    """

    def __init__(self, embedder: Embedder, index, chunks: Sequence[Chunk]) -> None:
        raise NotImplementedError("Retriever.__init__() is unwritten")

    def retrieve(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        raise NotImplementedError("Retriever.retrieve() is unwritten")


def build_prompt(question: str, retrieved: Sequence[tuple[Chunk, float]]) -> list[Message]:
    """Two messages: the system instruction, then the numbered sources and the question.

    - ``Message.system(SYSTEM_INSTRUCTION)`` first.
    - Then ONE user message containing every retrieved chunk as its own line,
      ``"[n] "`` followed by the chunk text exactly, numbered from 1 in the order
      given; then the question. For example::

          Sources:
          [1] <text of the first chunk>
          [2] <text of the second chunk>

          Question: <question>

    The mock model finds a source by its ``[n]`` marker, so keep the format.
    """
    raise NotImplementedError("build_prompt() is unwritten")


def parse_citations(answer_text: str) -> list[int]:
    """Every ``[n]`` marker in the answer, as ints, in order of first appearance, no repeats.

    Also accept comma lists inside one bracket: ``"[1, 3]"`` -> ``[1, 3]``.
    ``"as [2] says, and [2] again [1]"`` -> ``[2, 1]``. No markers -> ``[]``.
    """
    raise NotImplementedError("parse_citations() is unwritten")


class RAGPipeline:
    """Retrieve, prompt, generate, resolve citations.

    ``RAGPipeline(retriever, llm, k=3)``. ``answer(question, k=None)`` returns::

        {"answer":    the model's text,
         "citations": [chunk ids for every VALID [n] in the answer, in order],
         "sources":   [the retrieved Chunk objects, in [1]..[k] order]}

    A marker whose n is outside 1..len(sources) is dropped here without comment;
    the boss makes a fuss about it. ``llm.complete(messages)`` returns a
    ``Completion`` whose ``.text`` is the answer.
    """

    def __init__(self, retriever: Retriever, llm: LLM, k: int = 3) -> None:
        raise NotImplementedError("RAGPipeline.__init__() is unwritten")

    def answer(self, question: str, k: int | None = None) -> dict:
        raise NotImplementedError("RAGPipeline.answer() is unwritten")
