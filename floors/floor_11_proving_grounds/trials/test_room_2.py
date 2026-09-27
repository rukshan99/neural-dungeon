"""TRIAL 11.2 - THE BOOTSTRAP ORACLE

Intervals that contain the truth, intervals that widen when the sample shrinks,
paired resampling that notices small consistent gains, and a prophecy about
which arena results can be told apart.
"""

import math

import numpy as np
import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_2_bootstrap_oracle")


# ---------------------------------------------------------------- bootstrap_ci
def test_the_interval_contains_the_true_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=5.0, scale=2.0, size=200)
    low, high = room.bootstrap_ci(x, rng=np.random.default_rng(2))
    assert isinstance(low, float) and isinstance(high, float), "Return two Python floats."
    assert low <= x.mean() <= high, f"The interval ({low:.3f}, {high:.3f}) must contain the sample mean {x.mean():.3f}."
    assert low <= 5.0 <= high, f"The interval ({low:.3f}, {high:.3f}) should contain the true mean 5.0 for this seed."
    # SE of the mean is 2/sqrt(200) = 0.141, so a 95% interval is about 0.55 wide.
    assert 0.35 < high - low < 0.8, (
        f"For n=200, sd=2 the 95% interval should be about 0.55 wide (4 x 2/sqrt(200)); yours is {high - low:.3f}. "
        "Are you resampling WITH replacement and taking the 2.5th and 97.5th percentiles?"
    )


def test_fewer_stones_means_a_wider_interval():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, size=500)
    lo_big, hi_big = room.bootstrap_ci(x, rng=np.random.default_rng(4))
    lo_small, hi_small = room.bootstrap_ci(x[:20], rng=np.random.default_rng(4))
    assert (hi_small - lo_small) > 2.5 * (hi_big - lo_big), (
        f"n=20 gave width {hi_small - lo_small:.3f}, n=500 gave width {hi_big - lo_big:.3f}. "
        "Sampling error shrinks like 1/sqrt(n): the small sample's interval should be about 5x wider."
    )


def test_a_looser_alpha_gives_a_narrower_interval():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 1.0, size=100)
    lo95, hi95 = room.bootstrap_ci(x, alpha=0.05, rng=np.random.default_rng(6))
    lo80, hi80 = room.bootstrap_ci(x, alpha=0.20, rng=np.random.default_rng(6))
    assert hi80 - lo80 < hi95 - lo95, "An 80% interval is narrower than a 95% one: alpha/2 and 1-alpha/2 are the percentiles."


def test_the_oracle_can_bootstrap_any_statistic():
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.normal(0.0, 1.0, 99), [1000.0]])  # one outlier
    lo_mean, hi_mean = room.bootstrap_ci(x, statistic=np.mean, rng=np.random.default_rng(8))
    lo_med, hi_med = room.bootstrap_ci(x, statistic=np.median, rng=np.random.default_rng(8))
    assert hi_mean - lo_mean > 5.0, "The outlier should make the mean's interval very wide."
    assert -1.0 < lo_med and hi_med < 1.0, f"The median shrugs off the outlier: its interval should sit near 0, got ({lo_med:.3f}, {hi_med:.3f})."


def test_the_same_seed_speaks_the_same_prophecy():
    x = np.arange(30, dtype=float)
    first = room.bootstrap_ci(x, rng=np.random.default_rng(9))
    second = room.bootstrap_ci(x, rng=np.random.default_rng(9))
    assert first == second, "Same rng seed, same interval. Draw randomness only from the rng you were given."


def test_the_oracle_draws_n_boot_stones_and_reads_the_alpha_over_two_percentiles():
    """A statistic that ignores its sample and just counts calls: the recomputations are then exactly 0..n_boot-1,
    so the interval's ends reveal which percentiles were read and how many resamples were drawn."""
    ticks = iter(range(10**6))
    low, high = room.bootstrap_ci(np.zeros(10), statistic=lambda _sample: float(next(ticks)), n_boot=2000, alpha=0.05)
    assert 45 <= low <= 55 and 1945 <= high <= 1955, (
        f"With n_boot=2000 the recomputations are 0..1999, and a 95% interval runs from their 2.5th to their 97.5th "
        f"percentile: about (50, 1950). Got ({low:.1f}, {high:.1f}). About (100, 1900) means you used alpha where "
        "alpha/2 belongs; a high end far below 1950 means fewer than n_boot resamples."
    )
    ticks = iter(range(10**6))
    low, high = room.bootstrap_ci(np.zeros(10), statistic=lambda _sample: float(next(ticks)), n_boot=1000, alpha=0.2)
    assert 95 <= low <= 105 and 895 <= high <= 905, (
        f"n_boot=1000, alpha=0.2: the 10th and 90th percentiles of 0..999 are about (100, 900); got ({low:.1f}, {high:.1f})."
    )


def test_the_oracle_refuses_an_empty_bag():
    with pytest.raises(ValueError):
        room.bootstrap_ci([])


# ------------------------------------------------------------ paired_bootstrap
def test_identical_champions_have_delta_zero_and_no_significance():
    scores = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=float)
    delta, lo, hi, p = room.paired_bootstrap(scores, scores.copy(), rng=np.random.default_rng(10))
    assert delta == 0.0 and lo == 0.0 and hi == 0.0, f"Identical scores: delta and both bounds must be exactly 0; got {(delta, lo, hi)}."
    assert math.isclose(p, 1.0), f"Identical scores: the two-sided p-value is 1.0 (cap it); got {p}."
    assert not room.is_significant((lo, hi))


def test_clearly_different_champions_are_told_apart():
    rng = np.random.default_rng(11)
    a = (rng.random(100) < 0.5).astype(float)
    b = (rng.random(100) < 0.9).astype(float)
    delta, lo, hi, p = room.paired_bootstrap(a, b, rng=np.random.default_rng(12))
    assert math.isclose(delta, b.mean() - a.mean()), f"delta is mean(b) - mean(a) = {b.mean() - a.mean():.3f}; you said {delta:.3f}."
    assert lo > 0.0, f"A 40-point gap at n=100 is unmistakable; the interval ({lo:.3f}, {hi:.3f}) should sit entirely above 0."
    assert p < 0.05, f"p should be tiny here; got {p}."
    assert room.is_significant((lo, hi))


def test_the_oracle_resamples_fate_in_pairs():
    """b is a + 0.05 on every item. Paired: the difference is constant, the interval is a point.
    Resampled independently, the interval would be ~0.5 wide and would contain 0."""
    rng = np.random.default_rng(13)
    a = rng.normal(0.0, 1.0, size=100)
    b = a + 0.05
    delta, lo, hi, p = room.paired_bootstrap(a, b, rng=np.random.default_rng(14))
    assert math.isclose(delta, 0.05, abs_tol=1e-9)
    assert math.isclose(lo, 0.05, abs_tol=1e-9) and math.isclose(hi, 0.05, abs_tol=1e-9), (
        f"Every paired difference is exactly 0.05, so the interval must be (0.05, 0.05); got ({lo:.4f}, {hi:.4f}). "
        "You resampled a and b separately. Draw ONE set of indices and apply it to both."
    )
    assert room.is_significant((lo, hi)), "A consistent gain, however small, is significant when it is consistent."


def test_the_paired_interval_is_as_wide_as_the_noise_in_the_differences():
    """Continuous paired differences with a known spread: the 95% interval must be about 2 x 1.96 standard errors wide."""
    rng = np.random.default_rng(15)
    a = rng.normal(0.0, 1.0, size=400)
    b = a + rng.normal(0.10, 0.5, size=400)
    d = b - a
    delta, lo, hi, p = room.paired_bootstrap(a, b, rng=np.random.default_rng(16))
    assert math.isclose(delta, d.mean()), f"delta is mean(b - a) = {d.mean():.4f}; you said {delta:.4f}."
    normal = 2 * 1.96 * d.std() / math.sqrt(400)
    assert 0.88 * normal < hi - lo < 1.12 * normal, (
        f"The 95% paired interval should be about 2 x 1.96 x sd(b - a) / sqrt(n) = {normal:.4f} wide; yours is "
        f"{hi - lo:.4f}. About a sixth too narrow means you read the 5th and 95th percentiles: use alpha/2 and 1 - alpha/2."
    )
    assert lo > 0.0 and p < 0.05, f"A gain of 0.11 with SE 0.024 is unmistakable: ({lo:.4f}, {hi:.4f}), p={p}."


def test_the_paired_oracle_refuses_unpaired_scores():
    with pytest.raises(ValueError):
        room.paired_bootstrap([1, 0, 1], [1, 0])


# --------------------------------------------------------------- is_significant
def test_significance_is_an_interval_that_avoids_zero():
    assert room.is_significant((0.1, 0.3)) is True
    assert room.is_significant((-0.3, -0.1)) is True, "An interval entirely below zero is also significant (b is worse)."
    assert room.is_significant((-0.1, 0.3)) is False, "An interval straddling zero is not significant."
    assert room.is_significant((0.0, 0.3)) is False, "Touching zero is not excluding zero."


# ------------------------------------------------------- required_sample_size
def test_required_sample_size_follows_the_normal_approximation():
    n = room.required_sample_size(0.5, 0.05)
    assert isinstance(n, int), f"Return a Python int, got {type(n).__name__}."
    assert n == 385, f"z^2 p(1-p)/m^2 = 1.96^2 * 0.25 / 0.0025 = 384.16 -> 385; you said {n}."
    assert room.required_sample_size(0.8, 0.02) == 1537, "A +-2 point margin around 80% needs 1537 items."
    assert room.required_sample_size(0.8, 0.04) == 385, "Halving the margin quadruples n; doubling it quarters n."


def test_required_sample_size_rejects_nonsense():
    with pytest.raises(ValueError):
        room.required_sample_size(1.0, 0.05)
    with pytest.raises(ValueError):
        room.required_sample_size(0.5, 0.0)


# ----------------------------------------------------------------- prophecy
SCENARIOS = {
    "n=50: 80% vs 84%": (50, 0.80, 0.84),
    "n=2000: 80% vs 84%": (2000, 0.80, 0.84),
    "n=200: 60% vs 75%": (200, 0.60, 0.75),
}
VERDICTS = ("distinguishable", "not distinguishable")


def _arena_outcomes(n, p_a, p_b, rng):
    """Two systems scored on the same n items, with EXACTLY the stated accuracies, in a random order."""
    a = np.zeros(n)
    a[: int(round(p_a * n))] = 1.0
    rng.shuffle(a)
    b = np.zeros(n)
    b[: int(round(p_b * n))] = 1.0
    rng.shuffle(b)
    return a, b


def _reference_paired_ci(a, b, rng, n_boot=2000):
    d = b - a
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    deltas = d[idx].mean(axis=1)
    return tuple(np.percentile(deltas, [2.5, 97.5]))


def test_the_prophecy_is_complete():
    assert set(room.ORACLE_PROPHECY) == set(SCENARIOS), "Do not rename or remove the oracle's scenarios; answer them."


@pytest.mark.parametrize("scenario", list(SCENARIOS), ids=list(SCENARIOS))
def test_the_oracle_resamples_fate(scenario):
    prediction = room.ORACLE_PROPHECY.get(scenario)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied {scenario!r}. Fill in ORACLE_PROPHECY.")
    assert prediction in VERDICTS, f"Answer with one of {VERDICTS}, not {prediction!r}."
    n, p_a, p_b = SCENARIOS[scenario]
    rng = np.random.default_rng(2024)
    a, b = _arena_outcomes(n, p_a, p_b, rng)
    lo, hi = _reference_paired_ci(a, b, rng)
    truth = "distinguishable" if (lo > 0 or hi < 0) else "not distinguishable"
    se = math.sqrt(p_a * (1 - p_a) / n + p_b * (1 - p_b) / n)
    assert prediction == truth, (
        f"{scenario}: the paired 95% interval for the gap is ({lo:+.3f}, {hi:+.3f}), so the verdict is "
        f"'{truth}', not '{prediction}'. The gap is {p_b - p_a:.2f} and its standard error is about {se:.3f}; "
        "a gap needs roughly two standard errors to be told from zero."
    )
