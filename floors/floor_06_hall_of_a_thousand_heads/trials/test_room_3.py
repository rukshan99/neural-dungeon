"""TRIAL 6.3 - THE THOUSAND HEADS

Shapes through the split and the merge, one head equals the single gaze, and
your weights, copied into torch.nn.MultiheadAttention, must produce the same
output to 1e-5.
"""

import pytest

from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
F = torch.nn.functional
nn = torch.nn

room = load_room(__file__, "room_3_the_thousand_heads")
gaze = load_room(__file__, "room_1_the_single_gaze")

D_MODEL, HEADS, B, T = 32, 4, 2, 8


def _build(d_model=D_MODEL, n_heads=HEADS, seed=3):
    torch.manual_seed(seed)
    mha = room.MultiHeadAttention(d_model, n_heads)
    for name in ("d_model", "n_heads", "head_dim", "qkv", "proj"):
        assert hasattr(mha, name), (
            f"MultiHeadAttention is missing self.{name}. The trial reads these attributes by name; see __init__'s docstring."
        )
    return mha.eval()


def _x(seed=4, d_model=D_MODEL):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, d_model, generator=g)


def _torch_twin(mha):
    """torch's module wearing your weights: in_proj <- qkv, out_proj <- proj."""
    twin = nn.MultiheadAttention(mha.d_model, mha.n_heads, batch_first=True, bias=True)
    with torch.no_grad():
        twin.in_proj_weight.copy_(mha.qkv.weight)
        twin.in_proj_bias.copy_(mha.qkv.bias)
        twin.out_proj.weight.copy_(mha.proj.weight)
        twin.out_proj.bias.copy_(mha.proj.bias)
    return twin.eval()


# ------------------------------------------------------------------ carvings
def test_the_hall_has_exactly_the_right_number_of_carvings():
    mha = _build()
    assert isinstance(mha.qkv, nn.Linear) and tuple(mha.qkv.weight.shape) == (3 * D_MODEL, D_MODEL), (
        f"qkv must be nn.Linear(d_model, 3 * d_model): weight (96, 32), got {tuple(mha.qkv.weight.shape)}"
    )
    assert isinstance(mha.proj, nn.Linear) and tuple(mha.proj.weight.shape) == (D_MODEL, D_MODEL), (
        f"proj must be nn.Linear(d_model, d_model): weight (32, 32), got {tuple(mha.proj.weight.shape)}"
    )
    n = sum(p.numel() for p in mha.parameters())
    assert n == 4 * D_MODEL**2 + 4 * D_MODEL, (
        f"parameter count should be 4*d^2 + 4*d = {4 * D_MODEL**2 + 4 * D_MODEL} (3d^2+3d for qkv, d^2+d for proj); yours has {n}."
    )
    assert mha.head_dim == D_MODEL // HEADS, f"head_dim should be d_model // n_heads = {D_MODEL // HEADS}, got {mha.head_dim}"


def test_a_head_count_that_does_not_divide_the_model_is_refused():
    with pytest.raises(ValueError):
        room.MultiHeadAttention(30, 4)


# ---------------------------------------------------------------- split/merge
def test_split_heads_gives_each_head_its_own_contiguous_slice():
    mha = _build()
    x = _x()
    heads = mha.split_heads(x)
    assert heads.shape == (B, HEADS, T, D_MODEL // HEADS), (
        f"split_heads should give (B, H, T, head_dim) = {(B, HEADS, T, D_MODEL // HEADS)}, got {tuple(heads.shape)}"
    )
    hd = D_MODEL // HEADS
    for h in range(HEADS):
        assert torch.equal(heads[:, h], x[..., h * hd : (h + 1) * hd]), (
            f"head {h} must own columns {h * hd}:{(h + 1) * hd} of the model dim. You reshaped where you "
            "should have viewed-then-transposed: x.view(B, T, H, hd).transpose(1, 2)."
        )


def test_merge_heads_undoes_split_heads_exactly():
    mha = _build()
    x = _x()
    back = mha.merge_heads(mha.split_heads(x))
    assert back.shape == x.shape, f"merge_heads should restore (B, T, d_model) = {tuple(x.shape)}, got {tuple(back.shape)}"
    assert torch.equal(back, x), "merge_heads(split_heads(x)) must be x bit for bit: transpose(1, 2) first, then reshape."


# -------------------------------------------------------------------- forward
def test_the_output_keeps_the_shape_of_the_input_and_the_weights_sum_to_one():
    mha = _build()
    x = _x()
    out = mha(x)
    assert isinstance(out, torch.Tensor) and out.shape == (B, T, D_MODEL), (
        f"forward(x) should return a (B, T, d_model) tensor, got {tuple(out.shape) if isinstance(out, torch.Tensor) else type(out)}"
    )
    out2, weights = mha(x, return_weights=True)
    assert torch.equal(out, out2), "return_weights must not change the output"
    assert weights.shape == (B, HEADS, T, T), f"weights should be (B, H, T, T) = {(B, HEADS, T, T)}, got {tuple(weights.shape)}"
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), "every head's rows must sum to 1"


def test_one_head_is_the_single_gaze_wearing_two_linear_layers():
    mha = _build(n_heads=1, seed=7)
    x = _x(seed=8)
    q, k, v = mha.qkv(x).split(D_MODEL, dim=-1)
    single, _ = gaze.scaled_dot_product_attention(q, k, v)
    expected = mha.proj(single)
    err = (mha(x) - expected).abs().max().item()
    assert err < 1e-5, (
        f"With n_heads=1, MultiHeadAttention is exactly Room 6.1's attention between qkv and proj; "
        f"max difference {err:.2e}. Check the order of q, k, v in the split and that you scale by sqrt(head_dim)."
    )


def test_the_thousand_heads_agree_with_torch_multihead_attention():
    mha = _build()
    x = _x()
    twin = _torch_twin(mha)
    ref, ref_weights = twin(x, x, x, need_weights=True, average_attn_weights=True)
    out, weights = mha(x, return_weights=True)
    err = (out - ref).abs().max().item()
    assert err < 1e-5, (
        f"max |yours - nn.MultiheadAttention with your weights| = {err:.2e}, want < 1e-5. The mapping is "
        "in_proj_weight = qkv.weight, out_proj = proj, heads are contiguous chunks of the model dim. "
        "If shapes are right and values are wrong, the scaling (sqrt(head_dim)) or the merge order is off."
    )
    werr = (weights.mean(dim=1) - ref_weights).abs().max().item()
    assert werr < 1e-5, (
        f"torch returns the weights averaged over heads; your (B, H, T, T).mean(1) differs by {werr:.2e}."
    )


def test_the_veil_passes_through_every_head_and_torch_wants_it_inverted():
    mha = _build(seed=5)
    x = _x(seed=6)
    veil = torch.tril(torch.ones(T, T, dtype=torch.bool))  # True = may attend, our convention
    out, weights = mha(x, mask=veil, return_weights=True)
    future = ~veil
    assert (weights[:, :, future] == 0).all(), (
        "with a causal mask every head's weights above the diagonal must be exactly 0; a (T, T) mask "
        "broadcasts over (B, H, T, T) on its own."
    )
    twin = _torch_twin(mha)
    ref, _ = twin(x, x, x, attn_mask=~veil, need_weights=False)  # torch: True = may NOT attend
    err = (out - ref).abs().max().item()
    assert err < 1e-5, (
        f"masked output differs from torch by {err:.2e}. Note the trial passes ~veil to torch: "
        "nn.MultiheadAttention's boolean attn_mask means True = ignore, the opposite of this floor."
    )


def test_a_per_batch_mask_broadcasts_over_the_heads():
    mha = _build(seed=9)
    x = _x(seed=10)
    g = torch.Generator().manual_seed(11)
    mask = torch.rand(B, 1, T, T, generator=g) > 0.3
    mask[..., torch.arange(T), torch.arange(T)] = True  # everyone hears themselves
    out = mha(x, mask=mask)
    q, k, v = F.linear(x, mha.qkv.weight, mha.qkv.bias).split(D_MODEL, dim=-1)
    hd = D_MODEL // HEADS
    split = lambda t: t.view(B, T, HEADS, hd).transpose(1, 2)  # noqa: E731
    heads = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=mask)
    ref = F.linear(heads.transpose(1, 2).reshape(B, T, D_MODEL), mha.proj.weight, mha.proj.bias)
    err = (out - ref).abs().max().item()
    assert err < 1e-5, (
        f"A (B, 1, T, T) mask must broadcast over the head axis; yours differs from the reference by {err:.2e}."
    )
