"""TRIAL 6.4 - THE PADDING VEIL

Padded keys weigh nothing, padded batches compute the same thing as unpadded
sequences run alone, and nothing is ever NaN.
"""

import pytest

from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
F = torch.nn.functional

room = load_room(__file__, "room_4_the_padding_veil")
heads = load_room(__file__, "room_3_the_thousand_heads")

D_MODEL, HEADS = 32, 4


def _mha(seed=4):
    torch.manual_seed(seed)
    return heads.MultiHeadAttention(D_MODEL, HEADS).eval()


def _reference_causal(mha, x):
    """Independent causal MHA from mha's weights via torch's own attention (no padding)."""
    B, T, d = x.shape
    H, hd = mha.n_heads, d // mha.n_heads
    q, k, v = F.linear(x, mha.qkv.weight, mha.qkv.bias).split(d, dim=-1)
    split = lambda t: t.view(B, T, H, hd).transpose(1, 2)  # noqa: E731
    out = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=True)
    return F.linear(out.transpose(1, 2).reshape(B, T, d), mha.proj.weight, mha.proj.bias)


# ------------------------------------------------------------------- the masks
def test_the_padding_mask_marks_real_tokens_true_and_blanks_false():
    mask = room.key_padding_mask(torch.tensor([3, 5, 0]), T=5)
    assert mask.dtype == torch.bool, f"the mask must be torch.bool, got {mask.dtype}"
    assert mask.shape == (3, 5), f"key_padding_mask should be (B, T) = (3, 5), got {tuple(mask.shape)}"
    expected = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.equal(mask, expected), (
        f"True where position < length. For lengths [3, 5, 0] and T=5 expected\n{expected.int()}\ngot\n{mask.int()}"
    )


def test_the_combined_veil_has_a_head_axis_of_one_and_hides_both_future_and_blanks():
    lengths = torch.tensor([4, 2])
    T = 4
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    padding = torch.arange(T)[None, :] < lengths[:, None]
    combined = room.combine_masks(causal, padding)
    assert combined.shape == (2, 1, T, T), (
        f"combine_masks should give (B, 1, T, T) = (2, 1, 4, 4), got {tuple(combined.shape)}. "
        "The 1 is the head axis, kept so the mask broadcasts over every head."
    )
    assert combined.dtype == torch.bool
    for b in range(2):
        for i in range(T):
            for j in range(T):
                want = (j <= i) and (j < lengths[b].item())
                assert combined[b, 0, i, j].item() == want, (
                    f"combined[b={b}, 0, query={i}, key={j}] should be {want}: causal[i, j] AND padding[b, j]. "
                    "Padding enters along the KEY axis (the last one)."
                )


# -------------------------------------------------------------- safe softmax
def test_safe_softmax_silences_masked_voices_and_keeps_rows_summing_to_one():
    g = torch.Generator().manual_seed(1)
    scores = torch.randn(2, 3, 5, 6, generator=g)
    mask = torch.rand(2, 1, 5, 6, generator=g) > 0.4
    mask[..., 0] = True
    weights = room.safe_softmax(scores, mask)
    assert weights.shape == scores.shape, f"safe_softmax keeps the shape of scores, got {tuple(weights.shape)}"
    masked = weights[mask.expand_as(weights).logical_not()]
    assert (masked == 0).all(), f"masked entries must be exactly 0, largest is {masked.abs().max().item():.3e}"
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), "rows with at least one allowed key must sum to 1"
    plain = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
    assert torch.allclose(weights, plain, atol=1e-6), "where a row is alive, safe_softmax is the ordinary masked softmax"


def test_safe_softmax_returns_silence_not_nan_for_a_fully_veiled_row():
    g = torch.Generator().manual_seed(2)
    scores = torch.randn(1, 1, 4, 4, generator=g)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))[None, None].clone()
    mask[..., 2, :] = False  # query 2 may hear nobody
    weights = room.safe_softmax(scores, mask)
    assert not torch.isnan(weights).any(), (
        "a row of all -inf goes through softmax as NaN ((-inf) - (-inf)). Detect rows with no True "
        "(mask.any(-1, keepdim=True)) and set them to zero."
    )
    assert (weights[..., 2, :] == 0).all(), "a fully masked row must come out as all zeros"
    other = weights[..., [0, 1, 3], :].sum(dim=-1)
    assert torch.allclose(other, torch.ones_like(other), atol=1e-6), "the live rows still sum to 1"


# ---------------------------------------------------------------- masked_mha
def test_padded_keys_receive_exactly_zero_weight():
    lengths = torch.tensor([5, 3, 1])
    T = 5
    g = torch.Generator().manual_seed(3)
    scores = torch.randn(3, HEADS, T, T, generator=g)
    mask = room.combine_masks(torch.tril(torch.ones(T, T, dtype=torch.bool)), room.key_padding_mask(lengths, T))
    weights = room.safe_softmax(scores, mask)
    for b, L in enumerate(lengths.tolist()):
        leaked = weights[b, :, :, L:]
        assert (leaked == 0).all(), (
            f"sequence {b} has length {L}; keys {L}..{T - 1} are blank parchment and must weigh exactly 0 for every "
            f"query and head. Largest leaked weight: {leaked.abs().max().item():.3e}."
        )
        real = weights[b, :, :L, :].sum(dim=-1)
        assert torch.allclose(real, torch.ones_like(real), atol=1e-6), f"real queries of sequence {b} must still sum to 1"


def test_a_padded_scroll_reads_the_same_as_the_scroll_alone():
    mha = _mha()
    lengths = torch.tensor([8, 5, 2])
    g = torch.Generator().manual_seed(5)
    x = torch.randn(3, 8, D_MODEL, generator=g)
    out = room.masked_mha(mha, x, lengths)
    assert out.shape == x.shape, f"masked_mha returns (B, T, d_model) = {tuple(x.shape)}, got {tuple(out.shape)}"
    for b, L in enumerate(lengths.tolist()):
        alone = _reference_causal(mha, x[b : b + 1, :L])[0]
        err = (out[b, :L] - alone).abs().max().item()
        assert err < 1e-5, (
            f"sequence {b} (length {L}) computes something different inside the padded batch than when run alone, "
            f"unpadded: max difference {err:.2e}. Padding must be invisible to real positions: mask it along the keys."
        )


def test_changing_the_blank_parchment_changes_nothing_real():
    mha = _mha(seed=6)
    lengths = torch.tensor([6, 3, 4])
    g = torch.Generator().manual_seed(7)
    x = torch.randn(3, 6, D_MODEL, generator=g)
    base = room.masked_mha(mha, x, lengths)
    x2 = x.clone()
    padded = ~(torch.arange(6)[None, :] < lengths[:, None])
    x2[padded] = torch.randn(int(padded.sum()), D_MODEL, generator=g) * 10
    out = room.masked_mha(mha, x2, lengths)
    real = ~padded
    assert torch.allclose(out[real], base[real], atol=1e-6), (
        "rewriting the padded positions changed a real position's output. A padded KEY is being heard."
    )


def test_an_empty_scroll_produces_silence_not_nan():
    mha = _mha(seed=8)
    lengths = torch.tensor([4, 0, 2])
    g = torch.Generator().manual_seed(9)
    x = torch.randn(3, 4, D_MODEL, generator=g)
    out = room.masked_mha(mha, x, lengths)
    assert torch.isfinite(out).all(), (
        "a length-0 sequence gives every query a fully masked row. Plain softmax makes that NaN, and NaN in "
        "one sequence poisons the whole batch's gradient. Use safe_softmax."
    )
    alone = _reference_causal(mha, x[0:1, :4])[0]
    assert torch.allclose(out[0], alone, atol=1e-5), "the non-empty sequences must still be computed correctly"
