"""BOSS - THE HALLUCINATING LIBRARIAN  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Two defences in order: a relevance gate before the model is called, and a
lexical citation check after. Neither trusts the model's confidence, and neither
trusts a citation because it looks like one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from dungeon.artifacts.llm import LLM

from ..assets.embedder import content_tokens
from .room_4_retrieval_rite import CANNOT_FIND, Retriever, build_prompt, parse_citations

LOW_RELEVANCE = "low_relevance"
NO_SUPPORTED_CITATION = "no_supported_citation"
REFUSAL_REASONS = frozenset({LOW_RELEVANCE, NO_SUPPORTED_CITATION})

_SENTENCE_END = re.compile(r"[.!?]+(?:\s*\[\d+(?:\s*,\s*\d+)*\])*(?=\s|$)")
_MARKER = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


def split_sentences(text: str) -> list[str]:
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
    """Open (True) only when the best retrieval score reaches the threshold."""
    scores = list(scores)
    return bool(scores) and float(max(scores)) >= threshold


def calibrate_threshold(on_topic_scores: Sequence[float], off_topic_scores: Sequence[float]) -> float:
    """Midpoint between the on-topic 10th percentile and the off-topic 90th percentile."""
    low_on = np.percentile(np.asarray(on_topic_scores, dtype=float), 10)
    high_off = np.percentile(np.asarray(off_topic_scores, dtype=float), 90)
    return float((low_on + high_off) / 2.0)


# ------------------------------------------------------------------ phase 2
def content_words(text: str) -> set[str]:
    """Lowercase, punctuation-free, stopword-free, lightly stemmed word set; markers removed first."""
    return set(content_tokens(_MARKER.sub(" ", text)))


def citation_supported(answer_sentence: str, chunk_text: str, min_overlap: float = 0.5) -> bool:
    """Enough of the sentence's content words must appear in the chunk."""
    words = content_words(answer_sentence)
    if not words:
        return False
    overlap = len(words & content_words(chunk_text)) / len(words)
    return overlap >= min_overlap


def verify_citations(answer: str, sources: Sequence[str], min_overlap: float = 0.5) -> dict:
    """Every [n] in the answer, classified as valid / invalid / unsupported."""
    order: list[int] = []
    supported: dict[int, bool] = {}
    for sentence in split_sentences(answer):
        for n in parse_citations(sentence):
            if n not in order:
                order.append(n)
            if not 1 <= n <= len(sources):
                supported[n] = False  # invalid; classified below by range
                continue
            ok = citation_supported(sentence, sources[n - 1], min_overlap)
            supported[n] = supported.get(n, True) and ok  # every citing sentence must be supported
    report = {"valid": [], "invalid": [], "unsupported": []}
    for n in order:
        if not 1 <= n <= len(sources):
            report["invalid"].append(n)
        elif supported[n]:
            report["valid"].append(n)
        else:
            report["unsupported"].append(n)
    return report


# ------------------------------------------------------------------ phase 3
def _strip_markers(text: str, numbers: Sequence[int]) -> str:
    for n in numbers:
        text = re.sub(rf"\s*\[{n}\]", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


class GroundedRAG:
    """Gate, generate, verify, strip, or refuse."""

    def __init__(self, retriever: Retriever, llm: LLM, threshold: float, k: int = 3, min_overlap: float = 0.5) -> None:
        self.retriever = retriever
        self.llm = llm
        self.threshold = threshold
        self.k = k
        self.min_overlap = min_overlap

    def _refuse(self, reason: str) -> dict:
        return {"answer": CANNOT_FIND, "citations": [], "refused": True, "reason": reason}

    def answer(self, question: str) -> dict:
        retrieved = self.retriever.retrieve(question, self.k)
        if not relevance_gate([score for _, score in retrieved], self.threshold):
            return self._refuse(LOW_RELEVANCE)  # the model is never asked
        text = self.llm.complete(build_prompt(question, retrieved)).text
        chunks = [chunk for chunk, _ in retrieved]
        report = verify_citations(text, [c.text for c in chunks], self.min_overlap)
        if not report["valid"]:
            return self._refuse(NO_SUPPORTED_CITATION)
        cleaned = _strip_markers(text, report["invalid"] + report["unsupported"])
        return {
            "answer": cleaned,
            "citations": [chunks[n - 1].id for n in report["valid"]],
            "refused": False,
            "reason": None,
        }


# ------------------------------------------------------------------ phase 4
LIBRARIAN_PROPHECY: dict[str, str | None] = {
    "off-topic question; the model fabricates an answer and cites [1]": "gate",
    "on-topic question, gold passage retrieved, the model answers from it and cites it": "neither",
    "on-topic question, gold passage retrieved, the model ignores it and fabricates, citing [1] (an unrelated passage)": "verifier",
    "on-topic question; the model answers correctly but cites [9] when only 3 sources were given": "verifier",
    "on-topic question; the model answers correctly but adds no citation at all": "verifier",
    "on-topic question phrased with none of the passage's words, so the best score is low": "gate",
    "the model copies a sentence from source [2] word for word, changes one number in it, and cites [2]": "neither",
    "off-topic question that happens to reuse a passage's words; the model fabricates a sentence sharing most of those words and cites that passage": "neither",
}
