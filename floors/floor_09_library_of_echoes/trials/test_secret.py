"""SECRET - THE RERANKER'S LEDGER

BM25 against a straightforward reference, reciprocal rank fusion on toy
rankings, and a hybrid search that must not be worse than either ledger alone.
"""

import itertools
import math
from collections import Counter

import numpy as np
import pytest

from dungeon.trials import load_room
from floors.floor_09_library_of_echoes.assets.corpus import GOLDEN_QA, OFF_TOPIC_QUESTIONS, PASSAGES
from floors.floor_09_library_of_echoes.assets.embedder import embed, tokenize

ledger = load_room(__file__, "secret_rerankers_ledger")
catalogue = load_room(__file__, "room_2_card_catalogue")

pytestmark = pytest.mark.secret

# Calibrated on the reference solution: dense, BM25 and hybrid each put the gold
# passage in the top 3 for 12/12 golden questions.
HYBRID_BAR = 10

DOCS = [tokenize(p.text) for p in PASSAGES]
IDS = [p.id for p in PASSAGES]


def slow_bm25(docs, query, k1=1.5, b=0.75):
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    df = Counter(term for d in docs for term in set(d))
    scores = np.zeros(n)
    for i, d in enumerate(docs):
        tf = Counter(d)
        for term in query:
            if term not in tf:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            f = tf[term]
            scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
    return scores


# ----------------------------------------------------------------------- BM25
@pytest.mark.parametrize("k1,b", [(1.5, 0.75), (1.2, 0.5), (2.0, 1.0), (1.5, 0.0)], ids=["1.5/0.75", "1.2/0.5", "2.0/1.0", "1.5/0.0"])
def test_bm25_matches_the_reference_ledger_to_the_ninth_decimal(k1, b):
    bm25 = ledger.BM25(k1=k1, b=b).fit(DOCS, IDS)
    for qa in GOLDEN_QA + tuple(OFF_TOPIC_QUESTIONS[:2]):
        question = qa.question if hasattr(qa, "question") else qa
        query = tokenize(question)
        got = bm25.score(query)
        expected = slow_bm25(DOCS, query, k1, b)
        assert got.shape == (len(DOCS),), f"score() returns one number per document: shape {got.shape}, expected ({len(DOCS)},)."
        np.testing.assert_allclose(got, expected, atol=1e-9, err_msg=(
            f"BM25 scores differ from the reference for {question!r} with k1={k1}, b={b}. Check the idf formula "
            "ln((N - df + 0.5) / (df + 0.5) + 1) and the length normalisation k1 * (1 - b + b * |d| / avgdl)."
        ))


def test_bm25_remembers_its_ids():
    bm25 = ledger.BM25().fit(DOCS, IDS)
    assert list(bm25.ids) == IDS, "fit(docs, ids) must keep the ids, in order, as bm25.ids."
    assert list(ledger.BM25().fit(DOCS[:3]).ids) == [0, 1, 2], "Without ids, positions 0..N-1 are the ids."


def test_bm25_gives_nothing_for_words_the_library_has_never_seen():
    bm25 = ledger.BM25().fit(DOCS, IDS)
    np.testing.assert_array_equal(bm25.score(["zyzzyva", "quokka"]), 0.0, err_msg="A term in no document contributes 0 everywhere.")


def test_bm25_counts_a_repeated_query_term_twice():
    bm25 = ledger.BM25().fit(DOCS, IDS)
    once = bm25.score(["basilisk"])
    twice = bm25.score(["basilisk", "basilisk"])
    np.testing.assert_allclose(twice, 2 * once, atol=1e-12, err_msg="Sum over the query tokens as given: a repeated token counts twice.")


def test_bm25_prizes_rare_words_over_common_ones():
    bm25 = ledger.BM25().fit(DOCS, IDS)
    rare = bm25.score(["basilisk"]).max()  # in 2 passages
    common = bm25.score(["floor"]).max()  # in a good third of them
    assert rare > common, (
        f"The best score for the rare word 'basilisk' ({rare:.3f}) should beat the best for the common word 'floor' ({common:.3f}). "
        "That is what idf does."
    )


# ------------------------------------------------------------------------ RRF
def test_rrf_merges_two_toy_ledgers():
    fused = ledger.reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]], k=60)
    order = [doc_id for doc_id, _ in fused]
    assert order == ["a", "c", "b", "d"], (
        f"a: 1/61 + 1/62, c: 1/63 + 1/61, b: 1/62, d: 1/62 -> a, c, then b and d tied. Got {order}."
    )
    scores = dict(fused)
    assert abs(scores["a"] - (1 / 61 + 1 / 62)) < 1e-12 and abs(scores["b"] - 1 / 62) < 1e-12
    assert all(isinstance(s, float) for _, s in fused)


def test_rrf_does_not_care_which_ledger_you_opened_first():
    rankings = [["a", "b", "c", "d"], ["d", "c", "b"], ["b", "e"]]
    baseline = ledger.reciprocal_rank_fusion(rankings)
    for perm in itertools.permutations(rankings):
        assert ledger.reciprocal_rank_fusion(list(perm)) == baseline, "Fusion must be invariant to the order of the rankings."


def test_rrf_breaks_ties_the_same_way_every_time():
    fused = ledger.reciprocal_rank_fusion([["x", "b"], ["x", "d"]])
    assert [d for d, _ in fused] == ["x", "b", "d"], "b and d tie at 1/62; break ties by str(id) ascending: b before d."
    assert ledger.reciprocal_rank_fusion([]) == []


def test_rrf_k_softens_the_top_ranks():
    sharp = dict(ledger.reciprocal_rank_fusion([["a", "b"]], k=1))
    soft = dict(ledger.reciprocal_rank_fusion([["a", "b"]], k=60))
    assert sharp["a"] / sharp["b"] > soft["a"] / soft["b"], "A small k makes rank 1 worth much more than rank 2; k=60 flattens the difference."


# --------------------------------------------------------------------- hybrid
def _dense_index():
    index = catalogue.FlatIndex()
    index.add(embed([p.text for p in PASSAGES]), IDS)
    return index


def test_hybrid_search_returns_k_fused_results_best_first():
    dense = _dense_index()
    bm25 = ledger.BM25().fit(DOCS, IDS)
    results = ledger.hybrid_search("What does the Basilisk's gaze do?", dense, bm25, k=4)
    assert len(results) == 4, f"k=4 but {len(results)} results."
    assert all(isinstance(r, tuple) and len(r) == 2 and r[0] in IDS for r in results), f"(passage id, rrf score) pairs expected: {results}"
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "Best first."
    assert results[0][0] in {"p02", "p45"}, (
        f"Both ledgers rank the two Basilisk passages (p02 and its near-duplicate p45) at the top; hybrid put {results[0][0]} first."
    )


def test_hybrid_recall_is_no_worse_than_either_ledger_alone():
    dense = _dense_index()
    bm25 = ledger.BM25().fit(DOCS, IDS)
    dense_hits = bm_hits = hybrid_hits = 0
    for qa in GOLDEN_QA:
        dense_top = [d for d, _ in dense.search(embed([qa.question])[0], 3)]
        scores = bm25.score(tokenize(qa.question))
        bm_top = [IDS[i] for i in np.argsort(-scores, kind="stable")[:3]]
        hybrid_top = [d for d, _ in ledger.hybrid_search(qa.question, dense, bm25, k=3)]
        dense_hits += qa.passage_id in dense_top
        bm_hits += qa.passage_id in bm_top
        hybrid_hits += qa.passage_id in hybrid_top
    assert hybrid_hits >= max(dense_hits, bm_hits), (
        f"recall@3: dense {dense_hits}/12, BM25 {bm_hits}/12, hybrid {hybrid_hits}/12. Fusing two rankings that both "
        "rank the gold passage highly must not lose it."
    )
    assert hybrid_hits >= HYBRID_BAR, f"hybrid recall@3 is {hybrid_hits}/12; the bar is {HYBRID_BAR}."
