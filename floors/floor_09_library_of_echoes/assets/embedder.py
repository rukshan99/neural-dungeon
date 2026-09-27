"""A deterministic toy text embedder for the Library of Echoes.

Real embedding models are neural networks trained so that texts with similar
*meaning* land near each other. This one is a stand-in that only knows about
*vocabulary*: two texts are close when they share words and word pairs. That is
enough to teach every mechanic on this floor (cosine, indexes, retrieval,
grounding), and it runs in microseconds with no download.

How it works (a hashed bag of words, a.k.a. the "hashing trick"):

1. lowercase, split into ``[a-z0-9]+`` tokens, drop a small stopword list, and
   strip plural endings so "rules" and "rule" become one feature;
2. features are the remaining unigrams and adjacent bigrams (``"kv_cache"``);
3. each feature is hashed (blake2b, so it is identical on every machine and
   every run) to a bucket in ``0..DIM-1`` and to a sign, +1 or -1;
4. the vector accumulates ``sign`` at ``bucket`` for every feature occurrence;
5. the vector is L2-normalised, so a dot product is a cosine similarity.

Sign hashing means two unrelated features that collide in the same bucket cancel
as often as they add, which keeps unrelated texts near-orthogonal even with only
256 dimensions. It is still a bag of words: a question that happens to reuse a
passage's words scores well even when it means something else. The boss floor
exploits exactly that.

``embed(texts) -> np.ndarray`` of shape ``(N, 256)``, dtype float32, rows of
unit length (an all-stopword text gives a zero row).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

import numpy as np

DIM = 256

STOPWORDS: frozenset[str] = frozenset(
    """
    a about after again all also always am an and any are as at be because been before
    between both but by can cannot could did do does down during each either every far few
    first for from go goes going good great had has have he her here his how i if in into
    is it its just last less long look many more most much must never new next no not now of
    often old on once one only or other our out over own s same say says she should so some
    still such than that the their them then there these they this those three to too two up
    very was way we well were what when where whether which while who why will with would
    yes you your
    """.split()
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase ``[a-z0-9]+`` tokens, nothing removed. ``"KV-cache!"`` -> ``["kv", "cache"]``."""
    return _TOKEN.findall(text.lower())


def stem(token: str) -> str:
    """A three-rule plural stripper: ``rules -> rule``, ``gazes -> gaze``, ``copies -> copy``.

    Deliberately crude. It only has to be the same on both sides of a comparison.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
        return token[:-1]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    """``tokenize`` minus stopwords, stemmed, order preserved."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


def features(text: str) -> list[str]:
    """Unigrams and adjacent bigrams of the content tokens."""
    toks = content_tokens(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


def _bucket_and_sign(feature: str) -> tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
    return h % DIM, 1.0 if (h >> 63) & 1 else -1.0


def embed(texts: Sequence[str] | Iterable[str]) -> np.ndarray:
    """Embed texts to an ``(N, DIM)`` float32 array with L2-normalised rows.

    A single string is treated as a one-element batch; pass ``[text]`` to be explicit.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for feature in features(text):
            bucket, sign = _bucket_and_sign(feature)
            out[row, bucket] += sign
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norms, 1e-12)).astype(np.float32)
