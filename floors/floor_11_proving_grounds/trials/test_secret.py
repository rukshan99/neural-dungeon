"""SECRET - THE CALIBRATION MIRROR

Synthetic champions with known confidence: one honest, one three times too
sure of itself. The mirror measures the gap and a single temperature closes it.
"""

import math

import numpy as np
import pytest

from dungeon.trials import load_room

mirror = load_room(__file__, "secret_calibration_mirror")

pytestmark = pytest.mark.secret

N, K = 4000, 5


def _calibrated_world(seed=5, scale=1.5):
    """Logits whose softmax IS the true label distribution: calibrated by construction."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, scale, size=(N, K))
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(K, p=row) for row in p])
    return z, labels


def _conf_and_correct(probs, labels):
    return probs.max(axis=1), probs.argmax(axis=1) == labels


# ------------------------------------------------------------------- softmax
def test_softmax_rows_sum_to_one_and_temperature_flattens():
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 3.0, size=(6, K))
    p1 = mirror.softmax(z)
    np.testing.assert_allclose(p1.sum(axis=1), 1.0, atol=1e-12)
    p3 = mirror.softmax(z, temperature=3.0)
    assert np.all(p3.max(axis=1) < p1.max(axis=1)), "Dividing logits by T > 1 must lower every row's top probability."
    assert np.array_equal(p3.argmax(axis=1), p1.argmax(axis=1)), "Temperature never changes the argmax."
    big = mirror.softmax(np.array([[1000.0, 1000.0]]))
    assert not np.isnan(big).any(), "exp(1000) overflowed. Subtract the row max first."


def test_mean_nll_by_hand():
    assert math.isclose(mirror.mean_nll([[0.0, 0.0]], [0]), math.log(2)), "Two equal logits: p = 0.5, NLL = ln 2."
    assert math.isclose(mirror.mean_nll([[2.0, 0.0, 0.0]], [0], temperature=2.0), -math.log(math.e / (math.e + 2))), (
        "With T=2 the logits become [1, 0, 0]: p(correct) = e / (e + 2)."
    )
    assert isinstance(mirror.mean_nll([[1.0, 0.0]], [1]), float)


# ----------------------------------------------------------- reliability/ECE
def test_ece_by_hand():
    ece = mirror.expected_calibration_error([0.95, 0.95, 0.95, 0.95], [1, 1, 0, 0], n_bins=10)
    assert math.isclose(ece, 0.45), f"One bin, confidence 0.95, accuracy 0.5: ECE = 0.45; you said {ece:.4f}."
    ece = mirror.expected_calibration_error([0.55, 0.55, 0.85, 0.85], [1, 0, 1, 1], n_bins=10)
    assert math.isclose(ece, 0.1), (
        f"Bin (0.5, 0.6]: |0.5 - 0.55| = 0.05, weight 0.5. Bin (0.8, 0.9]: |1.0 - 0.85| = 0.15, weight 0.5. ECE = 0.1; you said {ece:.4f}."
    )


def test_the_reliability_table_has_the_right_shape_and_counts():
    z, labels = _calibrated_world()
    conf, correct = _conf_and_correct(mirror.softmax(z), labels)
    table = mirror.reliability_table(conf, correct, n_bins=10)
    assert table.shape == (10, 3), f"Expected (10, 3): count, mean confidence, accuracy per bin. Got {table.shape}."
    assert table[:, 0].sum() == N, "Every prediction lands in exactly one bin."
    assert np.isnan(table[0, 1]) and table[0, 0] == 0, "Confidence over 5 classes is at least 0.2, so bin (0, 0.1] is empty: count 0, NaN elsewhere."
    filled = table[:, 0] > 0
    assert np.all((table[filled, 1] >= 0.0) & (table[filled, 1] <= 1.0))
    assert np.all((table[filled, 2] >= 0.0) & (table[filled, 2] <= 1.0))


def test_a_perfectly_calibrated_champion_has_almost_no_ece():
    z, labels = _calibrated_world()
    conf, correct = _conf_and_correct(mirror.softmax(z), labels)
    ece = mirror.expected_calibration_error(conf, correct, n_bins=10)
    assert 0.0 <= ece < 0.03, f"Calibrated by construction: ECE should be within sampling noise of 0 (< 0.03 at n=4000); you measured {ece:.4f}."


def test_an_overconfident_champion_has_large_ece_without_losing_accuracy():
    z, labels = _calibrated_world()
    honest_conf, honest_correct = _conf_and_correct(mirror.softmax(z), labels)
    over_conf, over_correct = _conf_and_correct(mirror.softmax(3.0 * z), labels)
    assert np.array_equal(honest_correct, over_correct), "Scaling logits changes no argmax: accuracy is identical."
    ece = mirror.expected_calibration_error(over_conf, over_correct)
    assert ece > 0.15, f"Three times too sure: ECE should be large (about 0.25); you measured {ece:.4f}."
    assert over_conf.mean() > honest_correct.mean() + 0.15, "Sanity: mean confidence far above accuracy is what over-confidence looks like."


# -------------------------------------------------------- temperature scaling
def test_the_mirror_finds_the_temperature_that_undoes_the_overconfidence():
    z, labels = _calibrated_world()
    over = 3.0 * z
    grid = np.arange(0.5, 5.01, 0.25)
    T = mirror.temperature_scale(over, labels, grid)
    assert isinstance(T, float), f"Return a Python float from the grid, got {type(T).__name__}."
    assert 2.5 <= T <= 3.5, f"The logits were multiplied by 3, so the NLL-minimising temperature is about 3; you chose {T}."
    before = mirror.expected_calibration_error(*_conf_and_correct(mirror.softmax(over), labels))
    after = mirror.expected_calibration_error(*_conf_and_correct(mirror.softmax(over, T), labels))
    assert after < before / 4, f"ECE should collapse after scaling: before {before:.4f}, after {after:.4f}."
    assert after < 0.04


def test_a_calibrated_champion_keeps_temperature_one():
    z, labels = _calibrated_world()
    T = mirror.temperature_scale(z, labels, [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    assert T == 1.0, f"Already calibrated: the NLL is minimised at T = 1.0, not {T}."


def test_the_mirror_refuses_an_empty_grid():
    z, labels = _calibrated_world()
    with pytest.raises(ValueError):
        mirror.temperature_scale(z, labels, [])
