"""TRIAL 6.1 - THE SINGLE GAZE

One head, every voice. Scores, a softmax over the keys, the sqrt(d) scaling,
the mask before the softmax, and a prophecy about peaks.
"""

import math

import pytest

from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
F = torch.nn.functional

room = load_room(__file__, "room_1_the_single_gaze")


def _draw(B=2, Tq=8, Tk=8, d=16, dv=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, Tq, d, generator=g)
    k = torch.randn(B, Tk, d, generator=g)
    v = torch.randn(B, Tk, dv or d, generator=g)
    return q, k, v


# --------------------------------------------------------------------- scores
def test_scores_have_one_row_per_query_and_one_column_per_key():
    q, k, _ = _draw(Tq=5, Tk=7)
    scores = room.attention_scores(q, k)
    assert scores.shape == (2, 5, 7), (
        f"scores should be (B, Tq, Tk) = (2, 5, 7), got {tuple(scores.shape)}. "
        "Every query (row) against every key (column): q @ k.transpose(-2, -1)."
    )


def test_scores_are_divided_by_exactly_root_d():
    q, k, _ = _draw(d=16)
    raw = q @ k.transpose(-2, -1)
    scores = room.attention_scores(q, k)
    ratio = (scores / raw).flatten()
    assert torch.allclose(ratio, torch.full_like(ratio, 1 / math.sqrt(16)), atol=1e-6), (
        f"scores / (q @ k^T) should be 1/sqrt(d) = {1 / math.sqrt(16):.4f} everywhere; "
        f"observed {ratio[0].item():.4f}. Divide by sqrt(d) where d = q.shape[-1], not by d or sqrt(T)."
    )


def test_scores_pass_extra_leading_axes_through():
    g = torch.Generator().manual_seed(3)
    q = torch.randn(2, 4, 6, 8, generator=g)  # (B, H, T, hd)
    k = torch.randn(2, 4, 6, 8, generator=g)
    scores = room.attention_scores(q, k)
    assert scores.shape == (2, 4, 6, 6), (
        f"With a head axis, (2, 4, 6, 8) x (2, 4, 6, 8) should give (2, 4, 6, 6), got {tuple(scores.shape)}. "
        "Use transpose(-2, -1) so any leading axes are left alone. Room 6.3 depends on this."
    )


# ------------------------------------------------------------------ attention
def test_every_head_weighs_the_room_to_exactly_one():
    q, k, v = _draw(Tq=6, Tk=9)
    out, weights = room.scaled_dot_product_attention(q, k, v)
    assert weights.shape == (2, 6, 9), f"weights should be (B, Tq, Tk) = (2, 6, 9), got {tuple(weights.shape)}"
    assert out.shape == (2, 6, 16), f"out should be (B, Tq, dv) = (2, 6, 16), got {tuple(out.shape)}"
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), (
        f"each row of weights (one query's hearing) must sum to 1; observed sums like {row_sums.flatten()[:3].tolist()}. "
        "The softmax runs over the KEY axis: dim=-1."
    )
    assert (weights >= 0).all(), "softmax weights are never negative"


def test_identical_keys_make_the_output_the_mean_of_the_values():
    q, k, v = _draw()
    k = k[:, :1].expand_as(k).clone()  # every key is the same vector
    out, weights = room.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(weights, torch.full_like(weights, 1 / 8), atol=1e-6), (
        "When every key is identical, every score in a row is identical and the softmax is uniform (1/Tk each)."
    )
    expected = v.mean(dim=1, keepdim=True).expand_as(out)
    assert torch.allclose(out, expected, atol=1e-6), (
        "With uniform weights, out is the plain mean of v over the keys. Yours is not: check weights @ v."
    )


def test_the_gaze_matches_torch_to_a_millionth():
    q, k, v = _draw(Tq=8, Tk=8, d=32)
    out, _ = room.scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    err = (out - ref).abs().max().item()
    assert err < 1e-6, (
        f"max |yours - torch.nn.functional.scaled_dot_product_attention| = {err:.2e}, want < 1e-6. "
        "Check the order: scale, softmax over keys, then weights @ v."
    )


def test_a_masked_voice_has_exactly_zero_weight_and_the_rest_still_sum_to_one():
    q, k, v = _draw(Tq=6, Tk=8)
    g = torch.Generator().manual_seed(5)
    mask = torch.rand(2, 6, 8, generator=g) > 0.4
    mask[..., 0] = True  # every query may hear at least one key
    out, weights = room.scaled_dot_product_attention(q, k, v, mask=mask)
    assert (weights[~mask] == 0).all(), (
        f"masked keys must have weight exactly 0.0 (exp(-inf) == 0); largest masked weight is "
        f"{weights[~mask].abs().max().item():.3e}. Fill the scores with -inf BEFORE the softmax."
    )
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), (
        f"masked rows must still sum to 1; observed {row_sums.flatten()[:4].tolist()}. If you multiplied "
        "the weights by the mask AFTER the softmax, the denominator still counts the masked keys."
    )
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(out, ref, atol=1e-6), "With a boolean mask (True = attend), torch disagrees with you."


def test_a_single_mask_broadcasts_over_the_batch():
    q, k, v = _draw(Tq=5, Tk=5)
    mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))  # (Tq, Tk), no batch axis
    out, weights = room.scaled_dot_product_attention(q, k, v, mask=mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(out, ref, atol=1e-6), (
        "A (Tq, Tk) mask should broadcast over the batch. masked_fill(~mask, -inf) does that for free."
    )
    assert (weights[:, 0, 1:] == 0).all(), "with a lower-triangular mask, query 0 hears only key 0"


def test_values_may_be_wider_or_narrower_than_keys():
    q, k, v = _draw(Tq=4, Tk=7, d=16, dv=5)
    out, _ = room.scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 5), (
        f"out takes its last dim from v: (B, Tq, dv) = (2, 4, 5), got {tuple(out.shape)}"
    )


# ------------------------------------------------------- the prophecy of peaks
PROPHECY_KEYS = [
    "unscaled, d=4",
    "unscaled, d=64",
    "unscaled, d=1024",
    "scaled, d=4",
    "scaled, d=64",
    "scaled, d=1024",
]


def _mean_max_weight(d, scaled, T=16, rows=128, seed=6):
    """Mean over rows of the largest softmax weight, q and k ~ N(0, 1)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(rows, T, d, generator=g)
    k = torch.randn(rows, T, d, generator=g)
    scores = q @ k.transpose(-2, -1)
    if scaled:
        scores = scores / math.sqrt(d)
    return scores.softmax(dim=-1).max(dim=-1).values.mean().item()


def _bucket(m):
    if m < 0.5:
        return "spread"
    if m <= 0.9:
        return "sharp"
    return "one_hot"


@pytest.mark.parametrize("key", PROPHECY_KEYS)
def test_the_prophecy_of_the_peaks(key):
    prediction = room.PEAKINESS_PROPHECY.get(key)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied {key!r}. Fill in PEAKINESS_PROPHECY.")
    mode, d = key.split(", d=")
    measured = _mean_max_weight(int(d), scaled=(mode == "scaled"))
    truth = _bucket(measured)
    assert prediction == truth, (
        f"For {key}: the mean largest weight is {measured:.3f}, which is {truth!r}; you prophesied {prediction!r}. "
        f"Unscaled scores have standard deviation sqrt(d) = {math.sqrt(int(d)):.0f}; scaled ones always have 1."
    )


def test_the_prophecy_is_complete():
    assert set(room.PEAKINESS_PROPHECY) == set(PROPHECY_KEYS), "Do not add, remove or rename the prophecy's cases."
    assert all(v in (None, "spread", "sharp", "one_hot") for v in room.PEAKINESS_PROPHECY.values()), (
        'Answers are one of "spread", "sharp", "one_hot".'
    )
