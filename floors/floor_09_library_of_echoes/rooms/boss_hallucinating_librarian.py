"""BOSS - THE HALLUCINATING LIBRARIAN

               _______
              /       \\          "Of course the book exists. I have
             | (o) (o) |          cited it, have I not? Page forty-two.
             |    ^    |          Third shelf. You may thank me later."
             |  \\___/  |
          ___|_________|___       The Librarian answers EVERY question,
         /  |  [1] [1]  |  \\      instantly, confidently, with a citation.
        /   |___________|   \\     Ask about the Basilisk: a citation. Ask
       |    |  ~~~~~~~  |    |    how to boil an egg: a citation. The
       |    |  ~~~~~~~  |    |    citation is always "[1]". The book is
       |____|___________|____|    sometimes there.

WEAKNESS: a reader who REFUSES when retrieval is weak, and who VERIFIES every
citation against the source it names instead of trusting the number.

Two failure modes, two defences, in this order:

  Phase 1  THE GATE.  If the best retrieval score is low, the library does not
           hold the answer. Do not ask the model; it will invent one. Calibrate
           the threshold from on-topic and off-topic questions, robustly.
  Phase 2  THE VERIFIER.  A citation "[n]" is a claim: "source n supports this
           sentence". Check it lexically: enough of the sentence's content
           words must appear in source n. Numbers outside 1..k are invalid.
  Phase 3  GROUNDED RAG.  Gate, generate, verify, strip what failed, refuse if
           nothing survived. The user only ever sees supported claims.
  Phase 4  THE PROPHECY.  Which defence stops which failure, and which failure
           slips past both.

Run:  dungeon fight 9
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np  # noqa: F401  (np.percentile is handy for the calibration)

from dungeon.artifacts.llm import LLM

from ..assets.embedder import STOPWORDS, content_tokens  # noqa: F401  (for content_words)
from .room_4_retrieval_rite import (  # noqa: F401
    CANNOT_FIND,
    Retriever,
    build_prompt,
    parse_citations,
)

# The only two reasons a grounded reader refuses. `reason` is None when it answers.
LOW_RELEVANCE = "low_relevance"
NO_SUPPORTED_CITATION = "no_supported_citation"
REFUSAL_REASONS = frozenset({LOW_RELEVANCE, NO_SUPPORTED_CITATION})

# A sentence ends at . ! or ? (followed by whitespace or the end), plus any
# citation markers glued to it, so "A fact. [1] Another. [2]" splits in two and
# each keeps its marker. Given, not a stub: the regex is plumbing, not the lesson.
_SENTENCE_END = re.compile(r"[.!?]+(?:\s*\[\d+(?:\s*,\s*\d+)*\])*(?=\s|$)")


def split_sentences(text: str) -> list[str]:
    """``"A fact. [1] Another [2]."`` -> ``["A fact. [1]", "Another [2]."]``."""
    out, pos = [], 0
    for m in _SENTENCE_END.finditer(text):
        piece = text[pos : m.end()].strip()
        if piece:
            out.append(piece)
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        out.append(tail)
    return out


# ------------------------------------------------------------------ phase 1
def relevance_gate(scores: Sequence[float], threshold: float) -> bool:
    """True if retrieval is strong enough to answer from: ``max(scores) >= threshold``.

    False (refuse) when the best score is below the threshold or ``scores`` is empty.
    """
    raise NotImplementedError("relevance_gate() is unwritten")


def calibrate_threshold(on_topic_scores: Sequence[float], off_topic_scores: Sequence[float]) -> float:
    """A threshold that separates the two distributions of top-1 scores, robustly.

    ``on_topic_scores`` are best scores for questions the library CAN answer;
    ``off_topic_scores`` for questions it cannot. Return the midpoint between
    the 10th percentile of the on-topic scores and the 90th percentile of the
    off-topic ones, as a Python float. Percentiles, not min and max: one odd
    question must not drag the gate to the floor or the ceiling.
    """
    raise NotImplementedError("calibrate_threshold() is unwritten")


# ------------------------------------------------------------------ phase 2
def content_words(text: str) -> set[str]:
    """The set of content words of ``text``: lowercase, punctuation stripped, stopwords dropped.

    ``assets.embedder.content_tokens`` does exactly this (plus light stemming, so
    "heads" and "head" agree). Citation markers like ``[2]`` contribute nothing
    but a digit token; strip them with ``re.sub(r"\\[\\d+(?:\\s*,\\s*\\d+)*\\]", " ", text)`` first.
    """
    raise NotImplementedError("content_words() is unwritten")


def citation_supported(answer_sentence: str, chunk_text: str, min_overlap: float = 0.5) -> bool:
    """Does ``chunk_text`` lexically support ``answer_sentence``?

    overlap = |content_words(sentence) & content_words(chunk)| / |content_words(sentence)|
    Return ``overlap >= min_overlap``. A sentence with no content words cannot be
    verified: return False.
    """
    raise NotImplementedError("citation_supported() is unwritten")


def verify_citations(answer: str, sources: Sequence[str], min_overlap: float = 0.5) -> dict:
    """Classify every ``[n]`` in ``answer`` against ``sources`` (texts, in [1]..[k] order).

    Split the answer with ``split_sentences``; for each sentence, each cited n:
      - n outside 1..len(sources)           -> "invalid"
      - citation_supported(sentence, sources[n-1]) is False -> "unsupported"
      - otherwise                           -> "valid"
    A number cited by several sentences is valid only if every one of them is
    supported. Return ``{"valid": [...], "invalid": [...], "unsupported": [...]}``,
    each a list of ints in order of first appearance, no repeats, no number in
    two lists.
    """
    raise NotImplementedError("verify_citations() is unwritten")


# ------------------------------------------------------------------ phase 3
class GroundedRAG:
    """RAG that refuses when it should and shows the user only supported claims.

    ``GroundedRAG(retriever, llm, threshold, k=3, min_overlap=0.5)``

    ``answer(question)`` returns ``{"answer", "citations", "refused", "reason"}``:

      1. retrieve k chunks; if ``relevance_gate`` says no -> refused, reason
         LOW_RELEVANCE, answer CANNOT_FIND, citations [].
      2. ``build_prompt``, ``llm.complete``, take ``.text``.
      3. ``verify_citations`` against the retrieved chunk texts.
      4. no valid citation -> refused, reason NO_SUPPORTED_CITATION, answer
         CANNOT_FIND, citations [].
      5. otherwise: remove every invalid/unsupported ``[n]`` marker from the
         text, ``citations`` = chunk ids of the valid n in order, refused False,
         reason None.
    """

    def __init__(self, retriever: Retriever, llm: LLM, threshold: float, k: int = 3, min_overlap: float = 0.5) -> None:
        raise NotImplementedError("GroundedRAG.__init__() is unwritten")

    def answer(self, question: str) -> dict:
        raise NotImplementedError("GroundedRAG.answer() is unwritten")


# ------------------------------------------------------------------ phase 4
# For each failure, which defence stops it FIRST, given that the gate runs
# before the verifier? Answer "gate", "verifier", or "neither" (it reaches the
# user). Predict before you run the trial.
LIBRARIAN_PROPHECY: dict[str, str | None] = {
    "off-topic question; the model fabricates an answer and cites [1]": None,
    "on-topic question, gold passage retrieved, the model answers from it and cites it": None,
    "on-topic question, gold passage retrieved, the model ignores it and fabricates, citing [1] (an unrelated passage)": None,
    "on-topic question; the model answers correctly but cites [9] when only 3 sources were given": None,
    "on-topic question; the model answers correctly but adds no citation at all": None,
    "on-topic question phrased with none of the passage's words, so the best score is low": None,
    "the model copies a sentence from source [2] word for word, changes one number in it, and cites [2]": None,
    "off-topic question that happens to reuse a passage's words; the model fabricates a sentence sharing most of those words and cites that passage": None,
}
