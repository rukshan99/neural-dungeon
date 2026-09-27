"""TRIAL 9.2 - THE CARD CATALOGUE

Exact search is the yardstick. k-means must never make things worse. The
inverted file must find almost everything while reading almost nothing.
"""

import numpy as np
import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_2_card_catalogue")

# Calibrated on the reference solution with the data below: recall@10 was 0.99
# and the inverted file scanned ~14% of what the flat index scanned.
IVF_RECALL_BAR = 0.85
IVF_COMPARISON_FRACTION = 0.40


def _blobs(n, d, n_blobs, std, seed):
    """Unit vectors clustered around n_blobs random directions. Returns (X, blob labels)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_blobs, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = rng.integers(0, n_blobs, size=n)
    X = centers[labels] + std * rng.standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True), labels


def _blob_queries(n, d, n_blobs, std, seed_centers, seed_queries):
    """Queries drawn around the SAME centres as _blobs(..., seed_centers)."""
    rng_c = np.random.default_rng(seed_centers)
    centers = rng_c.standard_normal((n_blobs, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rng = np.random.default_rng(seed_queries)
    labels = rng.integers(0, n_blobs, size=n)
    Q = centers[labels] + std * rng.standard_normal((n, d))
    return Q / np.linalg.norm(Q, axis=1, keepdims=True)


def _brute_force(query, vectors, ids, k):
    q = query / np.linalg.norm(query)
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = v @ q
    order = np.argsort(-scores, kind="stable")[:k]
    return [(ids[i], float(scores[i])) for i in order]


# ------------------------------------------------------------------ FlatIndex
def test_the_flat_index_finds_exactly_what_brute_force_finds():
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((300, 16)) * rng.uniform(0.1, 10, size=(300, 1))  # varying norms on purpose
    ids = [f"book_{i}" for i in range(300)]
    index = room.FlatIndex()
    index.add(vectors, ids)
    for q in rng.standard_normal((10, 16)):
        got = index.search(q, 5)
        expected = _brute_force(q, vectors, ids, 5)
        assert [g[0] for g in got] == [e[0] for e in expected], (
            f"Flat search is exact by definition. Expected {[e[0] for e in expected]}, got {[g[0] for g in got]}. "
            "If the order is right but ids differ, are you returning positions instead of ids? If neither, "
            "are you scoring with a raw dot product on un-normalised vectors?"
        )
        np.testing.assert_allclose([g[1] for g in got], [e[1] for e in expected], atol=1e-9)


def test_the_flat_index_speaks_in_ids_and_respects_k():
    rng = np.random.default_rng(2)
    index = room.FlatIndex()
    index.add(rng.standard_normal((40, 8)), [f"scroll_{i}" for i in range(40)])
    results = index.search(rng.standard_normal(8), 6)
    assert len(results) == 6, f"k=6 but {len(results)} results came back."
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results), "Each result is an (id, score) pair."
    assert all(str(r[0]).startswith("scroll_") for r in results), f"Return the ids you were given, not positions: {results[:3]}"
    assert all(a[1] >= b[1] for a, b in zip(results, results[1:])), "Results must be best first."


def test_the_flat_index_counts_every_comparison_and_every_book():
    rng = np.random.default_rng(3)
    index = room.FlatIndex()
    index.add(rng.standard_normal((20, 4)), list(range(20)))
    index.add(rng.standard_normal((30, 4)), list(range(20, 50)))
    assert len(index) == 50, f"20 + 30 books were added; len(index) says {len(index)}."
    for _ in range(3):
        index.search(rng.standard_normal(4), 3)
    assert index.comparisons == 150, (
        f"Three searches over 50 books is 150 comparisons; the counter says {index.comparisons}. "
        "Flat search pays for every stored vector on every query. That is the whole reason the card catalogue exists."
    )


def test_an_empty_flat_index_answers_with_silence():
    index = room.FlatIndex()
    assert index.search(np.ones(4), 3) == [], "Searching an empty index should return [], not crash."


# --------------------------------------------------------------------- KMeans
@pytest.fixture(scope="module")
def blobs():
    X, labels = _blobs(600, 8, 6, 0.08, seed=5)
    km = room.KMeans(n_clusters=6, iters=50, seed=0).fit(X)
    return X, labels, km


def test_kmeans_has_the_right_number_of_shelves(blobs):
    X, _, km = blobs
    assert km.centroids.shape == (6, 8), f"6 clusters in 8 dimensions means centroids of shape (6, 8); got {km.centroids.shape}."
    assigned = km.assign(X)
    assert assigned.shape == (600,) and assigned.min() >= 0 and assigned.max() < 6, "assign() returns one cluster index in 0..5 per point."
    expected = ((X[:, None, :] - km.centroids[None]) ** 2).sum(-1).argmin(1)
    np.testing.assert_array_equal(assigned, expected, err_msg="assign() must pick the nearest centroid by Euclidean distance.")


def test_kmeans_inertia_never_climbs(blobs):
    _, _, km = blobs
    history = np.asarray(km.inertia_history, dtype=float)
    assert history.ndim == 1 and len(history) >= 2, f"inertia_history should hold one float per iteration; got {km.inertia_history!r}."
    climbs = np.diff(history) > 1e-9 * max(1.0, history[0])
    assert not climbs.any(), (
        f"Inertia climbed at iteration(s) {np.flatnonzero(climbs) + 1}: {history.round(4).tolist()}. "
        "Lloyd's steps (assign, then move to the mean) can only lower it. Did you re-seed an empty cluster at random, "
        "or record inertia against the centroids from a different step?"
    )


def test_kmeans_settles_where_every_centroid_is_the_mean_of_its_shelf(blobs):
    X, _, km = blobs
    labels = km.assign(X)
    for j in range(6):
        members = X[labels == j]
        if len(members) == 0:
            continue
        np.testing.assert_allclose(km.centroids[j], members.mean(axis=0), atol=1e-6, err_msg=(
            f"Centroid {j} is not the mean of the points assigned to it. At convergence, it must be."
        ))


def test_kmeans_shelves_the_blobs_far_better_than_one_shelf_would(blobs):
    X, _, km = blobs
    one_shelf = ((X - X.mean(axis=0)) ** 2).sum()
    final = km.inertia_history[-1]
    assert final < 0.3 * one_shelf, (
        f"Final inertia {final:.2f} vs {one_shelf:.2f} for a single centroid. Six well-separated blobs should be "
        "captured almost perfectly. Are the centroids moving at all?"
    )


def test_kmeans_repeats_itself_given_the_same_seed():
    X, _ = _blobs(200, 4, 4, 0.1, seed=6)
    a = room.KMeans(4, iters=20, seed=3).fit(X).centroids
    b = room.KMeans(4, iters=20, seed=3).fit(X).centroids
    np.testing.assert_array_equal(a, b, err_msg="Same seed, same centroids. Draw your initial centres from np.random.default_rng(seed).")


# ------------------------------------------------------------------- IVFIndex
def test_the_catalogue_refuses_books_before_it_has_shelves():
    index = room.IVFIndex(n_clusters=4, nprobe=1)
    with pytest.raises(RuntimeError):
        index.add(np.ones((3, 4)), ["a", "b", "c"])


def test_probing_every_shelf_is_exact_search_with_extra_steps():
    X, _ = _blobs(2000, 32, 16, 0.2, seed=9)
    ids = [f"tome_{i}" for i in range(2000)]
    Q = _blob_queries(20, 32, 16, 0.2, seed_centers=9, seed_queries=10)
    flat = room.FlatIndex()
    flat.add(X, ids)
    ivf = room.IVFIndex(n_clusters=8, nprobe=8, iters=25, seed=0)
    ivf.train(X)
    ivf.add(X, ids)
    for q in Q:
        exact = flat.search(q, 10)
        approx = ivf.search(q, 10)
        assert [a[0] for a in approx] == [e[0] for e in exact], (
            "With nprobe == n_clusters every vector is scored, so the result must equal flat search exactly. "
            f"Flat: {[e[0] for e in exact][:5]}..., IVF: {[a[0] for a in approx][:5]}..."
        )
        np.testing.assert_allclose([a[1] for a in approx], [e[1] for e in exact], atol=1e-9)
    assert ivf.comparisons == flat.comparisons, (
        f"Both indexes scored every vector on every query, so the counters should agree: flat {flat.comparisons}, IVF {ivf.comparisons}."
    )


def test_probing_a_few_shelves_keeps_recall_and_skips_most_of_the_reading():
    X, _ = _blobs(2000, 32, 16, 0.2, seed=9)
    ids = [f"tome_{i}" for i in range(2000)]
    Q = _blob_queries(100, 32, 16, 0.2, seed_centers=9, seed_queries=11)
    flat = room.FlatIndex()
    flat.add(X, ids)
    ivf = room.IVFIndex(n_clusters=16, nprobe=2, iters=25, seed=0)
    ivf.train(X)
    ivf.add(X, ids)
    exact = [flat.search(q, 10) for q in Q]
    approx = [ivf.search(q, 10) for q in Q]
    recall = room.recall_at_k(approx, exact, 10)
    fraction = ivf.comparisons / flat.comparisons
    assert recall >= IVF_RECALL_BAR, (
        f"recall@10 = {recall:.3f} < {IVF_RECALL_BAR}. Probing the 2 nearest of 16 shelves should still find almost "
        "every true neighbour on clustered data. Are you probing the centroids nearest the QUERY, and assigning "
        "with the same distance you probe with?"
    )
    assert fraction < IVF_COMPARISON_FRACTION, (
        f"The inverted file scored {fraction:.1%} of what flat search scored; the bar is {IVF_COMPARISON_FRACTION:.0%}. "
        "2 of 16 shelves is about 12% of the books. Are you scoring every vector and only filtering afterwards?"
    )


# ---------------------------------------------------------------- recall_at_k
def test_recall_at_k_on_toy_ledgers():
    exact = [[("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)], [("x", 0.5), ("y", 0.4), ("z", 0.3), ("w", 0.2)]]
    perfect = [[("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)], [("x", 0.5), ("y", 0.4), ("z", 0.3), ("w", 0.2)]]
    assert room.recall_at_k(perfect, exact, 4) == 1.0, "Identical result lists have recall 1.0."
    half = [[("a", 0.9), ("b", 0.8), ("q", 0.1), ("r", 0.1)], [("x", 0.5), ("y", 0.4), ("q", 0.1), ("r", 0.1)]]
    assert abs(room.recall_at_k(half, exact, 4) - 0.5) < 1e-12, "Two of four true neighbours found in each query is recall 0.5."
    reordered = [[("d", 0.6), ("c", 0.7), ("b", 0.8), ("a", 0.9)], [("w", 0.2), ("z", 0.3), ("y", 0.4), ("x", 0.5)]]
    assert room.recall_at_k(reordered, exact, 4) == 1.0, "Recall is about membership in the top-k set, not order."
    none = [[("q", 0.1)] * 4, [("q", 0.1)] * 4]
    assert room.recall_at_k(none, exact, 4) == 0.0
    ids_only = [["a", "b", "c", "d"], ["x", "y", "z", "w"]]
    assert room.recall_at_k(ids_only, exact, 4) == 1.0, "Plain id lists should be accepted too."
    assert abs(room.recall_at_k([["a", "b", "c", "d"]], [["a", "b", "z", "w", "c"]], 2) - 1.0) < 1e-12, (
        "Only the first k entries of each list count."
    )
    assert isinstance(room.recall_at_k(perfect, exact, 4), float), "Return a Python float."
