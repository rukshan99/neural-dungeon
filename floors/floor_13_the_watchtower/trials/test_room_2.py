"""TRIAL 13.2 - THE DRIFT GLASS

PSI that is zero for the same sample and large for a shifted one, a KS
distance that lives in [0, 1], a symmetric JS bounded by ln 2, an embedding
score that separates rotation from scaling, and a monitor that stays quiet
until the world actually moves.
"""

import math

import numpy as np
import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_2_the_drift_glass")


# -------------------------------------------------------------- histogram bins
def test_histogram_bins_are_reference_quantiles_with_open_ends():
    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, size=10_000)
    edges = room.histogram_bins(ref, n_bins=4)
    assert isinstance(edges, np.ndarray) and edges.shape == (5,), f"4 bins need 5 edges; got shape {getattr(edges, 'shape', None)}."
    assert edges[0] == -np.inf and edges[-1] == np.inf, "The outer edges are -inf and +inf so every current value lands in some bin."
    np.testing.assert_allclose(edges[1:-1], np.quantile(ref, [0.25, 0.5, 0.75]), err_msg="Inner edges are the reference's 1/n .. (n-1)/n quantiles.")
    assert np.all(np.diff(edges) >= 0), "Edges must be non-decreasing."


def test_bin_fractions_split_the_reference_evenly_and_catch_outliers():
    rng = np.random.default_rng(1)
    ref = rng.normal(0.0, 1.0, size=10_000)
    edges = room.histogram_bins(ref, n_bins=10)
    fractions = room.bin_fractions(ref, edges)
    assert fractions.shape == (10,) and math.isclose(fractions.sum(), 1.0)
    np.testing.assert_allclose(fractions, 0.1, atol=0.002, err_msg="Quantile edges put about a tenth of the reference in each of 10 bins.")
    outliers = room.bin_fractions([-100.0, 100.0], edges)
    assert outliers[0] == 0.5 and outliers[-1] == 0.5, f"Values beyond the reference belong in the first/last bin; got {outliers}."


# ------------------------------------------------------------------------ psi
def test_psi_is_zero_for_the_same_sample_and_small_for_the_same_distribution():
    rng = np.random.default_rng(2)
    ref = rng.normal(0.0, 1.0, size=5000)
    assert room.psi(ref, ref) == 0.0, "A sample against itself: every bin fraction matches, PSI is exactly 0."
    same = room.psi(ref, rng.normal(0.0, 1.0, size=5000))
    assert 0.0 <= same < 0.05, f"Two samples of the same distribution: PSI should be near 0 (sampling noise ~0.004 here), got {same:.4f}."
    assert isinstance(same, float)


def test_psi_is_large_for_a_mean_shift():
    rng = np.random.default_rng(3)
    ref = rng.normal(0.0, 1.0, size=5000)
    shifted = room.psi(ref, rng.normal(1.0, 1.0, size=5000))
    assert shifted > room.PSI_MAJOR, f"A one-standard-deviation shift is a major change (PSI ~0.8); you measured {shifted:.3f}."
    assert room.psi_verdict(shifted) == "major"
    assert room.psi_verdict(0.05) == "stable" and room.psi_verdict(0.15) == "moderate", "0.1 and 0.25 are the rule-of-thumb lines."


def test_psi_by_hand_is_in_nats():
    ref = np.arange(1000.0)  # four quantile bins, each holding exactly a quarter of the reference
    current = np.repeat([100.0, 300.0, 600.0, 900.0], [40, 30, 20, 10])  # current fractions 0.4, 0.3, 0.2, 0.1
    expected = sum((c - 0.25) * math.log(c / 0.25) for c in (0.4, 0.3, 0.2, 0.1))
    got = room.psi(ref, current, n_bins=4)
    assert math.isclose(got, expected, rel_tol=1e-6), (
        f"Reference fractions 0.25 each, current 0.4/0.3/0.2/0.1: PSI = sum (c - r) ln(c / r) = {expected:.4f}; you said {got:.4f}. "
        "PSI uses the natural log (log10 would give 0.099, and the 0.1 / 0.25 rule-of-thumb lines would mean something else)."
    )


def test_psi_survives_an_empty_bin_thanks_to_eps():
    ref = np.arange(1000.0)
    current = np.full(200, 5.0)  # everything in one bin
    value = room.psi(ref, current, n_bins=10)
    assert math.isfinite(value) and value > 0, f"Empty bins must not produce inf or nan: clip fractions to eps first. Got {value}."


# ------------------------------------------------------------------------- ks
def test_ks_is_zero_for_identical_samples_and_one_for_disjoint_ones():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert room.ks_statistic(a, a) == 0.0
    assert room.ks_statistic(a, a + 100.0) == 1.0, "Disjoint samples: at some x one CDF is 1 and the other 0."


def test_ks_by_hand_and_symmetric():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [3.0, 4.0, 5.0, 6.0]
    got = room.ks_statistic(a, b)
    assert math.isclose(got, 0.5), f"F_a(2) = 0.5 and F_b(2) = 0, F_a(4) = 1 and F_b(4) = 0.5: the largest gap is 0.5; you said {got}."
    assert math.isclose(room.ks_statistic(b, a), got), "KS is symmetric."
    a, b = [0.0, 10.0, 11.0], [1.0, 2.0, 3.0]
    got = room.ks_statistic(a, b)
    assert math.isclose(got, 2 / 3), (
        f"At x = 3, F_b = 1 and F_a = 1/3: the largest gap is 2/3 and it sits at one of b's values; you said {got:.4f}. "
        "Evaluate both CDFs on the POOLED values: a grid built from one sample alone misses the other sample's jumps."
    )
    assert math.isclose(room.ks_statistic(b, a), 2 / 3), "KS is symmetric, whichever sample supplies the point where the gap is largest."


def test_ks_stays_small_for_the_same_distribution():
    rng = np.random.default_rng(4)
    value = room.ks_statistic(rng.normal(size=2000), rng.normal(size=2000))
    assert 0.0 <= value < 0.08, f"Same distribution, n=2000 each: KS should be about 0.02-0.04, got {value:.3f}."
    with pytest.raises(ValueError):
        room.ks_statistic([], [1.0])


# ------------------------------------------------------------------------- js
def test_js_is_symmetric_zero_for_identical_and_ln2_for_disjoint():
    p = np.array([0.2, 0.3, 0.5])
    assert math.isclose(room.js_divergence(p, p), 0.0, abs_tol=1e-12)
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.5])
    assert math.isclose(room.js_divergence(a, b), math.log(2)), "Disjoint supports give the maximum, ln 2 (in nats)."
    x, y = np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.3, 0.6])
    assert math.isclose(room.js_divergence(x, y), room.js_divergence(y, x)), "JS is symmetric; KL is not."
    assert 0.0 <= room.js_divergence(x, y) <= math.log(2)


def test_js_by_hand_and_from_counts():
    # p = [0.5, 0.5], q = [1, 0]; m = [0.75, 0.25]
    expected = 0.5 * (0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(0.5 / 0.25)) + 0.5 * math.log(1.0 / 0.75)
    got = room.js_divergence([0.5, 0.5], [1.0, 0.0])
    assert math.isclose(got, expected), f"JS([.5,.5],[1,0]) = 0.5 KL(p||m) + 0.5 KL(q||m) = {expected:.4f}; you said {got:.4f}."
    assert math.isclose(room.js_divergence([50, 50], [0.5, 0.5]), 0.0, abs_tol=1e-12), "Counts are fine: normalise them first."
    with pytest.raises(ValueError):
        room.js_divergence([0.5, 0.5], [1.0])


def test_category_fractions_turn_tool_names_into_a_distribution():
    fractions = room.category_fractions(["search", "search", "calc"], categories=["calc", "search"])
    np.testing.assert_allclose(fractions, [1 / 3, 2 / 3])
    with pytest.raises(ValueError):
        room.category_fractions(["search", "teleport"], categories=["calc", "search"])
    ref = room.category_fractions(["search"] * 80 + ["calc"] * 20, ["calc", "search"])
    cur = room.category_fractions(["search"] * 40 + ["calc"] * 60, ["calc", "search"])
    assert room.js_divergence(ref, cur) > 0.05, "A tool-call mix that flips from 80/20 to 40/60 is visible drift."


# ------------------------------------------------------------------ embeddings
def test_embedding_drift_separates_rotation_from_scaling():
    rng = np.random.default_rng(5)
    ref = rng.normal(size=(500, 8)) + np.array([3.0] + [0.0] * 7)
    assert abs(room.embedding_drift(ref, ref)) < 1e-9, "Identical clouds: no drift."
    doubled = room.embedding_drift(ref, 2.0 * ref)
    assert math.isclose(doubled, 1.0, abs_tol=1e-9), f"Doubling every vector keeps the direction (cosine term 0) and doubles the mean norm (+1.0); got {doubled:.4f}."
    rotated = np.roll(ref, 1, axis=1)  # the centroid moves from axis 0 to axis 1: orthogonal, same norms
    value = room.embedding_drift(ref, rotated)
    assert 0.9 < value < 1.1, f"Orthogonal centroids with the same norms: cosine distance ~1, norm change ~0; got {value:.3f}."
    with pytest.raises(ValueError):
        room.embedding_drift(ref, rng.normal(size=(10, 4)))


# --------------------------------------------------------------------- monitor
def _world(seed, d=3):
    rng = np.random.default_rng(seed)
    return rng, rng.normal(size=(4000, d))


def test_the_monitor_report_has_the_expected_shape():
    rng, ref = _world(6)
    monitor = room.DriftMonitor(ref, window=1)
    report = monitor.observe(rng.normal(size=(400, 3)))
    for key in ("n_current", "psi", "ks", "max_psi", "max_ks", "drifted_features", "drifted"):
        assert key in report, f"The report is missing {key!r}."
    assert len(report["psi"]) == 3 and len(report["ks"]) == 3, "One PSI and one KS per feature."
    assert report["n_current"] == 400
    assert isinstance(report["drifted"], bool)


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_the_monitor_stays_quiet_when_nothing_moves(seed):
    rng, ref = _world(seed)
    monitor = room.DriftMonitor(ref)
    reports = [monitor.observe(rng.normal(size=(400, 3))) for _ in range(10)]
    noisy = [r for r in reports if r["drifted"]]
    assert not noisy, (
        f"Ten batches from the reference distribution and the glass flagged {len(noisy)}: max PSI {max(r['max_psi'] for r in reports):.3f}, "
        f"max KS {max(r['max_ks'] for r in reports):.3f}. The default thresholds ({room.DEFAULT_THRESHOLDS}) sit well above sampling noise; are you comparing against the reference's edges?"
    )


def test_the_monitor_flags_the_feature_that_moved():
    rng, ref = _world(10)
    monitor = room.DriftMonitor(ref)
    batch = rng.normal(size=(400, 3))
    batch[:, 1] += 1.0
    report = monitor.observe(batch)
    assert report["drifted"] is True, f"Feature 1 shifted by a full standard deviation; the glass must flag it. Report: {report}."
    assert report["drifted_features"] == [1], f"Name the feature that moved: [1], not {report['drifted_features']}."
    assert report["psi"][1] > room.PSI_MAJOR and report["ks"][1] > 0.3


def test_the_monitor_pools_a_window_of_batches():
    rng, ref = _world(11)
    monitor = room.DriftMonitor(ref, window=3)
    batches = [rng.normal(size=(400, 3)) for _ in range(5)]
    reports = [monitor.observe(batch) for batch in batches]
    sizes = [r["n_current"] for r in reports]
    assert sizes == [400, 800, 1200, 1200, 1200], f"window=3 pools the last three batches: expected [400, 800, 1200, 1200, 1200], got {sizes}."
    pooled = np.concatenate(batches[2:])  # the last three batches are what the fifth report must measure
    for j in range(3):
        assert math.isclose(reports[-1]["psi"][j], room.psi(ref[:, j], pooled[:, j], monitor.edges[j])), (
            f"Feature {j}: the report's PSI must be computed on the POOLED window against the reference's precomputed edges, not on the latest batch alone."
        )
        assert math.isclose(reports[-1]["ks"][j], room.ks_statistic(ref[:, j], pooled[:, j])), f"Feature {j}: KS must also be computed on the pooled window."


def test_either_instrument_alone_is_enough_to_flag_a_feature():
    rng = np.random.default_rng(13)
    ref = rng.normal(size=(4000, 2))
    only_psi = room.DriftMonitor(ref, thresholds={"psi": 1e-9, "ks": 10.0})  # KS lives in [0, 1]: it can never cross 10
    assert only_psi.observe(rng.normal(size=(400, 2)))["drifted_features"] == [0, 1], (
        "PSI over its line is drift even while KS is under its line: a feature drifts when EITHER instrument crosses (or, not and)."
    )
    only_ks = room.DriftMonitor(ref, thresholds={"psi": 10.0, "ks": 1e-9})  # PSI cannot reach 10 with eps = 1e-4 either
    assert only_ks.observe(rng.normal(size=(400, 2)))["drifted_features"] == [0, 1], "KS over its line is drift even while PSI is under its line."


def test_the_monitor_accepts_a_single_feature_and_custom_thresholds():
    rng = np.random.default_rng(12)
    monitor = room.DriftMonitor(rng.normal(size=4000), thresholds={"psi": 1e-9, "ks": 1e-9})
    report = monitor.observe(rng.normal(size=400))
    assert len(report["psi"]) == 1, "A 1-D reference is one feature."
    assert report["drifted"] is True, "With thresholds at ~0 even sampling noise counts: the thresholds must actually be used."
    assert monitor.thresholds["psi"] == 1e-9
