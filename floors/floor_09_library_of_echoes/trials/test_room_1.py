"""TRIAL 9.1 - THE ECHO METRIC

The books answer. Is the loudness you measured really the cosine of the angle,
and did you keep only the k loudest without sorting the whole library?
"""

import numpy as np

from dungeon.scrutiny import is_stub, names_called_in
from dungeon.trials import load_room

room = load_room(__file__, "room_1_echo_metric")

rng = np.random.default_rng(91)


def slow_cosine(a, b):
    out = np.zeros((len(a), len(b)))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            nx, ny = np.linalg.norm(x), np.linalg.norm(y)
            out[i, j] = 0.0 if nx == 0 or ny == 0 else float(x @ y) / (nx * ny)
    return out


# ------------------------------------------------------------ normalize_rows
def test_every_echo_is_normalised_to_unit_length_and_silence_stays_silent():
    x = rng.standard_normal((6, 5)) * np.array([[1.0], [10.0], [0.01], [1.0], [1.0], [1.0]])
    x[3] = 0.0
    out = room.normalize_rows(x)
    assert out.shape == x.shape, f"normalize_rows must keep the shape; got {out.shape} for {x.shape}"
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(np.delete(norms, 3), 1.0, atol=1e-9, err_msg="Every non-zero row must have norm 1.")
    assert np.all(out[3] == 0.0) and not np.any(np.isnan(out)), (
        "A zero row divided by its own norm is NaN. Divide by max(norm, eps) instead."
    )


def test_normalising_does_not_scribble_on_the_original():
    x = rng.standard_normal((4, 3)) * 5
    before = x.copy()
    room.normalize_rows(x)
    np.testing.assert_array_equal(x, before, err_msg="normalize_rows modified its input in place. Return a new array.")


# --------------------------------------------------------- cosine_similarity
def test_cosine_matches_the_slow_reference_with_shape_n_by_m():
    a = rng.standard_normal((7, 12)) * rng.uniform(0.1, 10, size=(7, 1))
    b = rng.standard_normal((5, 12)) * rng.uniform(0.1, 10, size=(5, 1))
    out = room.cosine_similarity(a, b)
    assert out.shape == (7, 5), f"(7, 12) x (5, 12) should give (7, 5) similarities, got {out.shape}."
    np.testing.assert_allclose(out, slow_cosine(a, b), atol=1e-9, err_msg="Similarities differ from cos(angle) = a.b / (|a||b|).")


def test_a_book_echoes_itself_at_exactly_one():
    a = rng.standard_normal((6, 8)) * rng.uniform(0.01, 100, size=(6, 1))
    np.testing.assert_allclose(np.diag(room.cosine_similarity(a, a)), 1.0, atol=1e-9, err_msg=(
        "cos(a, a) must be 1 whatever the length of a. If it is not, you forgot to normalise (or normalised only one side)."
    ))


def test_shouting_does_not_change_the_echo():
    a = rng.standard_normal((4, 6))
    b = rng.standard_normal((3, 6))
    base = room.cosine_similarity(a, b)
    np.testing.assert_allclose(room.cosine_similarity(2.0 * a, b), base, atol=1e-9, err_msg=(
        "cos(2a, b) must equal cos(a, b): cosine is scale-invariant. A dot product is not."
    ))
    np.testing.assert_allclose(room.cosine_similarity(a, 0.001 * b), base, atol=1e-9)
    np.testing.assert_allclose(room.cosine_similarity(-a, b), -base, atol=1e-9, err_msg="Flipping a vector flips the sign of every cosine.")


def test_a_silent_vector_echoes_zero_not_nan():
    a = np.zeros((1, 4))
    b = rng.standard_normal((3, 4))
    out = room.cosine_similarity(a, b)
    assert not np.any(np.isnan(out)), "A zero vector produced NaN. The eps in the normalisation is there for exactly this."
    np.testing.assert_allclose(out, 0.0, atol=1e-9)


# -------------------------------------------------------------------- top_k
def test_top_k_hands_back_the_loudest_first():
    scores = rng.standard_normal((5, 40))  # no ties, with overwhelming probability
    idx, vals = room.top_k(scores, 7)
    assert idx.shape == (5, 7) and vals.shape == (5, 7), f"Expected two (5, 7) arrays, got {idx.shape} and {vals.shape}."
    expected = np.argsort(-scores, axis=1)[:, :7]
    np.testing.assert_array_equal(idx, expected, err_msg="These are not the 7 largest per row, in descending order.")
    np.testing.assert_allclose(vals, np.take_along_axis(scores, expected, axis=1), err_msg=(
        "The returned values must be the scores at the returned indices."
    ))
    assert np.all(np.diff(vals, axis=1) <= 0), "Values must be sorted descending within every row."


def test_top_k_breaks_ties_by_the_lower_index():
    scores = np.array([[0.5, 0.9, 0.2, 0.9, 0.9, 0.1]])
    idx, vals = room.top_k(scores, 3)
    np.testing.assert_array_equal(idx[0], [1, 3, 4], err_msg=(
        f"Three books tie at 0.9 (indices 1, 3, 4). They must come back in ascending index order; you returned {idx[0].tolist()}."
    ))
    np.testing.assert_allclose(vals[0], [0.9, 0.9, 0.9])
    idx2, _ = room.top_k(scores, 3)
    np.testing.assert_array_equal(idx, idx2, err_msg="Same input, same output. top_k must be deterministic.")


def test_top_k_with_more_k_than_books_returns_the_whole_shelf_sorted():
    scores = np.array([[0.1, 0.7, 0.4], [0.9, 0.2, 0.5]])
    idx, vals = room.top_k(scores, 10)
    assert idx.shape == (2, 3), f"k=10 on 3 columns should return all 3, got shape {idx.shape}."
    np.testing.assert_array_equal(idx, [[1, 2, 0], [0, 2, 1]])
    np.testing.assert_allclose(vals, [[0.7, 0.4, 0.1], [0.9, 0.5, 0.2]])


def test_top_k_partitions_before_it_sorts():
    if is_stub(room.top_k):
        raise NotImplementedError("top_k() is unwritten")
    called = names_called_in(room.top_k)
    assert "argpartition" in called, (
        f"top_k calls {sorted(called)} but not argpartition. Sorting all M scores is O(M log M); "
        "np.argpartition selects the k largest in O(M), and you sort only those k."
    )


# ------------------------------------------------------------ nearest_neighbors
def test_nearest_neighbors_finds_the_true_neighbours():
    docs = rng.standard_normal((50, 16)) * rng.uniform(0.1, 5, size=(50, 1))
    queries = rng.standard_normal((6, 16))
    idx, scores = room.nearest_neighbors(queries, docs, 4)
    assert idx.shape == (6, 4) and scores.shape == (6, 4), f"Expected (6, 4) indices and scores, got {idx.shape}, {scores.shape}."
    ref = slow_cosine(queries, docs)
    expected = np.argsort(-ref, axis=1)[:, :4]
    np.testing.assert_array_equal(idx, expected, err_msg="Not the 4 most cosine-similar docs per query.")
    np.testing.assert_allclose(scores, np.take_along_axis(ref, expected, axis=1), atol=1e-9)
