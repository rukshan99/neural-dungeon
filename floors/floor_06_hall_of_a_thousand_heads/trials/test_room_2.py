"""TRIAL 6.2 - THE VEIL OF CAUSALITY

The veil must be lower-triangular, applied before the softmax, and your leak
detector must catch every torn veil this trial hands it without reading any
code.
"""

import math

import pytest

from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
F = torch.nn.functional

room = load_room(__file__, "room_2_veil_of_causality")


def _draw(B=2, T=8, d=16, seed=2):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, T, d, generator=g) for _ in range(3))


# ------------------------------------------------------------- torn veils, for the detector
def _mask_after_softmax(q, k, v):
    T = q.shape[1]
    veil = torch.tril(torch.ones(T, T, dtype=torch.bool))
    weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)
    return (weights * veil) @ v


def _off_by_one(q, k, v):
    T = q.shape[1]
    veil = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).masked_fill(~veil, float("-inf"))
    return scores.softmax(dim=-1) @ v


def _unveiled(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


def _honest(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


# ------------------------------------------------------------------- the veil
def test_the_veil_is_lower_triangular_and_boolean():
    veil = room.causal_mask(5)
    assert veil.dtype == torch.bool, f"the mask must be torch.bool, got {veil.dtype}"
    assert veil.shape == (5, 5), f"causal_mask(5) should be (5, 5), got {tuple(veil.shape)}"
    expected = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    assert torch.equal(veil, expected), (
        f"mask[i, j] must be True iff j <= i (lower triangle AND diagonal). Yours:\n{veil.int()}"
    )
    assert veil[0, 0].item() is True, "a position may always hear itself: the diagonal is True"


def test_no_voice_from_the_future_gets_any_weight():
    q, k, v = _draw()
    out, weights = room.causal_attention(q, k, v)
    assert weights.shape == (2, 8, 8) and out.shape == (2, 8, 16)
    future = ~torch.tril(torch.ones(8, 8, dtype=torch.bool))
    leaked = weights[:, future]
    assert (leaked == 0).all(), (
        f"every weight strictly above the diagonal must be exactly 0.0; the largest is {leaked.abs().max().item():.3e}. "
        "Apply the mask as -inf before the softmax."
    )
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), (
        f"rows must still sum to 1 behind the veil; observed {sums[0, :3].tolist()}."
    )


def test_the_first_word_hears_only_itself():
    q, k, v = _draw()
    out, weights = room.causal_attention(q, k, v)
    assert torch.allclose(weights[:, 0, 0], torch.ones(2), atol=1e-6), "query 0 has one allowed key, so its weight is 1"
    assert torch.allclose(out[:, 0], v[:, 0], atol=1e-6), "position 0's output is exactly v[:, 0]: nothing else is audible yet"


def test_the_veiled_gaze_matches_torch_is_causal():
    q, k, v = _draw(T=12, d=32, seed=9)
    out, _ = room.causal_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    err = (out - ref).abs().max().item()
    assert err < 1e-6, f"max |yours - torch is_causal=True| = {err:.2e}, want < 1e-6."


# --------------------------------------------------------------- the detector
def test_your_own_veil_does_not_leak():
    verdict = room.leaks_future(lambda q, k, v: room.causal_attention(q, k, v)[0], B=2, T=8, d=8, seed=0)
    assert isinstance(verdict, bool), f"leaks_future must return a Python bool, got {type(verdict).__name__}"
    assert verdict is False, (
        "leaks_future says your own causal_attention leaks. Either the veil is torn or the detector "
        "compares the wrong positions: only out[:, :t+1] must be unchanged when positions > t are perturbed."
    )


def test_the_detector_trusts_an_honest_veil():
    assert room.leaks_future(_honest, B=2, T=8, d=8, seed=1) is False, (
        "torch's is_causal=True attention is honest, but your detector accuses it. Are you perturbing "
        "positions <= t as well, or comparing the perturbed positions themselves?"
    )


def test_the_detector_catches_a_mask_applied_after_softmax():
    assert room.leaks_future(_mask_after_softmax, B=2, T=8, d=8, seed=1) is True, (
        "A veil applied after the softmax leaks through the denominator (the future keys are still "
        "in the sum that normalises the visible weights). Your detector missed it."
    )


def test_the_detector_catches_an_unveiled_gaze():
    assert room.leaks_future(_unveiled, B=2, T=8, d=8, seed=1) is True, (
        "Plain bidirectional attention hears everything, and your detector called it causal."
    )


def test_the_detector_catches_a_veil_hung_one_seat_too_far():
    assert room.leaks_future(_off_by_one, B=2, T=8, d=8, seed=1) is True, (
        "tril(..., diagonal=1) lets position t hear t+1. Perturbing positions > t must change out[:, t]; "
        "your detector did not notice."
    )


def test_the_detector_is_deterministic_in_its_seed():
    a = room.leaks_future(_off_by_one, B=1, T=6, d=4, seed=11)
    b = room.leaks_future(_off_by_one, B=1, T=6, d=4, seed=11)
    assert a is b is True, "the same seed must produce the same verdict: draw from torch.Generator().manual_seed(seed)"
