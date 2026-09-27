"""TRIAL 6.5 - THE ROTARY GALLERY

The rotation must preserve lengths, do nothing at position 0, agree with
complex multiplication and make the scores depend only on the distance
between seats. Grouped heads must collapse to Room 6.3 with one kv head per
head, to multi-query attention with one, and agree with torch in between.
"""

import math

import pytest

from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
F = torch.nn.functional
nn = torch.nn

room = load_room(__file__, "room_5_the_rotary_gallery")
heads = load_room(__file__, "room_3_the_thousand_heads")

D_MODEL, HEADS, B, T = 32, 4, 2, 8
HD = D_MODEL // HEADS


# ------------------------------------------------------- the trial's own rotary
def _ref_freqs(hd, base=10000.0):
    return torch.tensor([base ** (-2 * i / hd) for i in range(hd // 2)], dtype=torch.float32)


def _ref_rope(x, positions, base=10000.0):
    """RoPE as complex multiplication: pair (x[2i], x[2i+1]) is x[2i] + i*x[2i+1], times exp(i * m * theta_i)."""
    hd = x.shape[-1]
    angles = positions.to(torch.float32)[:, None] * _ref_freqs(hd, base)[None, :]  # (T, hd/2)
    pairs = torch.view_as_complex(x.contiguous().reshape(*x.shape[:-1], hd // 2, 2))
    rotated = pairs * torch.polar(torch.ones_like(angles), angles)  # exp(i * angle) has modulus 1
    return torch.view_as_real(rotated).flatten(-2)


def _x(seed, B=B, T=T, d=D_MODEL):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, d, generator=g)


def _qk(seed, hd=HD, T=T):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, HEADS, T, hd, generator=g), torch.randn(B, HEADS, T, hd, generator=g)


def _leaks(module_fn, x):
    """Perturb positions after t; do positions <= t move? (Room 6.2's test on a module.)"""
    base = module_fn(x)
    g = torch.Generator().manual_seed(99)
    for t in range(x.shape[1] - 1):
        x2 = x.clone()
        x2[:, t + 1 :] = torch.randn(x2[:, t + 1 :].shape, generator=g)
        out = module_fn(x2)
        if not torch.allclose(out[:, : t + 1], base[:, : t + 1], atol=1e-6, rtol=0.0):
            return True
    return False


# ------------------------------------------------------------ the frequency ladder
def test_the_frequencies_start_at_one_and_shrink_geometrically():
    theta = room.rope_frequencies(HD)
    assert theta.shape == (HD // 2,), f"rope_frequencies(head_dim={HD}) should be (head_dim // 2,) = ({HD // 2},), got {tuple(theta.shape)}"
    assert theta.dtype == torch.float32, f"frequencies should be float32, got {theta.dtype}"
    assert theta[0].item() == pytest.approx(1.0), f"theta_0 = base^0 must be exactly 1: pair 0 turns one radian per seat. Got {theta[0].item()}"
    assert torch.allclose(theta, _ref_freqs(HD), rtol=1e-6, atol=0), (
        f"theta_i must be base^(-2i / head_dim) for i = 0..{HD // 2 - 1}. Expected {_ref_freqs(HD).tolist()}, got {theta.tolist()}. "
        "torch.arange(0, head_dim, 2) is 2i; the exponent is -2i / head_dim, not -i / head_dim."
    )
    assert (theta[1:] < theta[:-1]).all(), "the ladder descends: every pair turns more slowly than the one before"
    big = room.rope_frequencies(64, base=1e6)
    assert torch.allclose(big, _ref_freqs(64, 1e6), rtol=1e-6, atol=0), "base is a parameter; rope_frequencies(64, base=1e6) must use it"
    with pytest.raises(ValueError):
        room.rope_frequencies(7)


def test_the_angles_are_position_times_frequency():
    positions = torch.arange(5)
    angles = room.rope_angles(positions, HD)
    assert angles.shape == (5, HD // 2), f"rope_angles should be (T, head_dim // 2) = (5, {HD // 2}), got {tuple(angles.shape)}"
    assert (angles[0] == 0).all(), "position 0 has angle 0 in every pair"
    expected = positions.float()[:, None] * _ref_freqs(HD)[None, :]
    assert torch.allclose(angles, expected, atol=1e-6), (
        "angles[m, i] must be m * theta_i: an outer product positions[:, None] * theta[None, :]. "
        f"Row 3 should be {expected[3].tolist()}, got {angles[3].tolist()}"
    )
    assert angles[3, 0].item() == pytest.approx(3.0), "pair 0 has theta = 1, so at position 3 its angle is 3 radians"


# ------------------------------------------------------------------- the rotation
def test_apply_rope_keeps_the_shape_and_position_zero_is_the_identity():
    q, _ = _qk(seed=1)
    out = room.apply_rope(q, torch.arange(T))
    assert out.shape == q.shape, f"apply_rope returns the shape it was given, {tuple(q.shape)}; got {tuple(out.shape)}"
    assert out.dtype == q.dtype, f"apply_rope keeps the dtype ({q.dtype}), got {out.dtype}"
    still = room.apply_rope(q, torch.zeros(T, dtype=torch.long))
    assert torch.allclose(still, q, atol=1e-6), (
        "at position 0 every angle is 0, cos = 1 and sin = 0: apply_rope must return x unchanged. "
        f"Largest change: {(still - q).abs().max().item():.3e}"
    )


def test_a_rotation_preserves_every_voices_length():
    q, _ = _qk(seed=2)
    out = room.apply_rope(q, torch.arange(T) * 37)  # big angles, still rotations
    assert torch.allclose(out.norm(dim=-1), q.norm(dim=-1), atol=1e-5), (
        "a rotation is orthogonal: the norm of every (b, h, t) vector must be unchanged. "
        f"Largest change in norm: {(out.norm(dim=-1) - q.norm(dim=-1)).abs().max().item():.3e}. "
        "Are you scaling by cos and sin without the cross terms, or rotating the wrong pairs?"
    )
    pair_norm = lambda z: (z[..., 0::2] ** 2 + z[..., 1::2] ** 2).sqrt()  # noqa: E731
    assert torch.allclose(pair_norm(out), pair_norm(q), atol=1e-5), (
        "each PAIR (x[2i], x[2i+1]) is rotated on its own, so every pair's length is preserved, not just the whole vector's."
    )
    assert not torch.allclose(out, q, atol=1e-3), "positions 0, 37, 74, ... must actually rotate something: yours is a no-op"


def test_the_first_pair_turns_by_one_radian_per_seat():
    q, _ = _qk(seed=3)
    m = 3
    out = room.apply_rope(q, torch.arange(T))
    c, s = math.cos(m), math.sin(m)
    x0, x1 = q[..., m, 0], q[..., m, 1]
    expected = torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1)
    assert torch.allclose(out[..., m, :2], expected, atol=1e-5), (
        f"pair 0 (coordinates 0 and 1) at position {m} must be rotated by exactly {m} radians: "
        "(x0 cos - x1 sin, x0 sin + x1 cos). This room uses INTERLEAVED pairs (x[2i], x[2i+1]); if you paired "
        "x[i] with x[i + head_dim/2] you built the rotate_half layout, which is a different column order. "
        "If the sign is off, you rotated the other way."
    )
    theta_last = _ref_freqs(HD)[-1].item()
    c, s = math.cos(m * theta_last), math.sin(m * theta_last)
    x0, x1 = q[..., m, -2], q[..., m, -1]
    expected = torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1)
    assert torch.allclose(out[..., m, -2:], expected, atol=1e-5), (
        f"the LAST pair turns by m * theta_last = {m * theta_last:.4f} radians at position {m}, the slowest ring in the gallery."
    )


def test_the_rotation_agrees_with_complex_multiplication():
    q, _ = _qk(seed=4)
    positions = torch.arange(T) + 5
    ref = _ref_rope(q, positions)
    out = room.apply_rope(q, positions)
    err = (out - ref).abs().max().item()
    assert err < 1e-5, (
        f"max |yours - complex form| = {err:.2e}. The trial treats each pair as the complex number x[2i] + i*x[2i+1] and "
        "multiplies by exp(i * m * theta_i) (torch.view_as_complex, torch.polar). Rotation by an angle IS that multiplication."
    )
    ref = _ref_rope(q, positions, base=500000.0)
    out = room.apply_rope(q, positions, base=500000.0)
    assert torch.allclose(out, ref, atol=1e-5), "apply_rope must pass its base down to the frequencies (Llama 3 uses 500000)"


def test_the_scores_hear_only_the_distance_between_seats():
    q, k = _qk(seed=5)
    pos = torch.arange(T)
    scores_here = room.apply_rope(q, pos) @ room.apply_rope(k, pos).transpose(-2, -1)
    scores_shifted = room.apply_rope(q, pos + 7) @ room.apply_rope(k, pos + 7).transpose(-2, -1)
    err = (scores_here - scores_shifted).abs().max().item()
    assert err < 1e-5, (
        f"shifting EVERY position by 7 changed the scores by up to {err:.2e}. R(m t)^T R(n t) = R((n - m) t): the dot product "
        "of a rotated q and a rotated k depends only on the difference of their positions. Yours depends on where the scroll starts."
    )
    scores_k_only = room.apply_rope(q, pos) @ room.apply_rope(k, pos + 7).transpose(-2, -1)
    assert not torch.allclose(scores_here, scores_k_only, atol=1e-3), (
        "moving only the keys by 7 seats changes the relative distance and MUST change the scores; yours did not rotate anything"
    )
    plain = q @ k.transpose(-2, -1)
    assert not torch.allclose(scores_here, plain, atol=1e-3), "rotated scores must differ from unrotated ones (except on the diagonal)"
    diag = torch.arange(T)
    assert torch.allclose(scores_here[..., diag, diag], plain[..., diag, diag], atol=1e-5), (
        "a query and a key at the SAME position (distance 0) are rotated by the same angle, so their score is the plain q . k"
    )


def test_a_larger_base_turns_the_slow_pairs_more_slowly():
    far = torch.tensor([1024])
    a4 = room.rope_angles(far, 64, base=1e4)[0]
    a6 = room.rope_angles(far, 64, base=1e6)[0]
    assert a4[0].item() == pytest.approx(1024.0) and a6[0].item() == pytest.approx(1024.0), "pair 0 has theta = 1 whatever the base"
    assert (a6[1:] < a4[1:]).all(), (
        "with base 1e6 every theta_i (i >= 1) is smaller than with base 1e4, so position 1024 is turned LESS in every slow pair. "
        "That is how long-context models keep far positions distinguishable: the slowest ring takes longer to come round."
    )
    period = 2 * math.pi / room.rope_frequencies(64)[-1].item()
    assert period == pytest.approx(2 * math.pi / _ref_freqs(64)[-1].item(), rel=1e-5), "the slowest pair of a 64-wide head with base 1e4 needs ~47,000 seats for one turn"
    assert 40_000 < period < 55_000, f"expected the slowest pair's period near 47,000 positions, got {period:.0f}"


# ----------------------------------------------------------------- RoPEAttention
def _gallery(seed=3, causal=True, base=10000.0):
    torch.manual_seed(seed)
    gallery = room.RoPEAttention(D_MODEL, HEADS, base=base, causal=causal)
    for name in ("d_model", "n_heads", "head_dim", "qkv", "proj", "base", "causal"):
        assert hasattr(gallery, name), f"RoPEAttention is missing self.{name}; call super().__init__ and store base and causal"
    return gallery.eval()


def _ref_gallery(gallery, x, positions=None, causal=True):
    """Room 6.3's layout with the trial's own complex RoPE on q and k, torch's attention, no rotation of v."""
    Bn, Tn, d = x.shape
    positions = torch.arange(Tn) if positions is None else positions
    q, k, v = F.linear(x, gallery.qkv.weight, gallery.qkv.bias).split(d, dim=-1)
    split = lambda t: t.view(Bn, Tn, HEADS, HD).transpose(1, 2)  # noqa: E731
    q, k, v = _ref_rope(split(q), positions, gallery.base), _ref_rope(split(k), positions, gallery.base), split(v)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return F.linear(out.transpose(1, 2).reshape(Bn, Tn, d), gallery.proj.weight, gallery.proj.bias)


def test_the_gallery_keeps_room_3s_carvings_and_shape():
    gallery = _gallery()
    assert isinstance(gallery, heads.MultiHeadAttention), "RoPEAttention subclasses Room 6.3's MultiHeadAttention"
    n = sum(p.numel() for p in gallery.parameters())
    assert n == 4 * D_MODEL**2 + 4 * D_MODEL, (
        f"RoPE adds NO parameters: still 4d^2 + 4d = {4 * D_MODEL**2 + 4 * D_MODEL}, got {n}. Positions are angles, not a table."
    )
    out = gallery(_x(seed=4))
    assert isinstance(out, torch.Tensor) and out.shape == (B, T, D_MODEL), (
        f"forward(x) returns a (B, T, d_model) = {(B, T, D_MODEL)} tensor, got {tuple(out.shape) if isinstance(out, torch.Tensor) else type(out)}"
    )


def test_the_gallery_does_not_leak_the_future():
    gallery = _gallery(seed=5)
    assert _leaks(gallery, _x(seed=6)) is False, (
        "perturbing positions after t changed the output at or before t. The veil (causal_mask(T)) must be applied "
        "before the softmax, rotation or no rotation."
    )


def test_the_gallery_is_room_3s_heads_with_rotated_queries_and_keys():
    gallery = _gallery(seed=7)
    x = _x(seed=8)
    err = (gallery(x) - _ref_gallery(gallery, x)).abs().max().item()
    assert err < 1e-5, (
        f"max |yours - reference| = {err:.2e}, want < 1e-5. The reference rotates q and k (complex form, positions 0..T-1), "
        "leaves v alone, then runs causal attention and proj. If you rotated v as well, or rotated before splitting heads "
        "(the pairs must live inside each head's head_dim), this is where it shows."
    )
    gallery = _gallery(seed=7, base=500000.0)
    err = (gallery(x) - _ref_gallery(gallery, x)).abs().max().item()
    assert err < 1e-5, f"with base=500000 the module must use self.base; differs from the reference by {err:.2e}"


def test_with_every_seat_at_position_zero_the_gallery_is_the_thousand_heads():
    gallery = _gallery(seed=9)
    plain = heads.MultiHeadAttention(D_MODEL, HEADS).eval()
    plain.load_state_dict(gallery.state_dict())
    x = _x(seed=10)
    veil = torch.tril(torch.ones(T, T, dtype=torch.bool))
    err = (gallery(x, positions=torch.zeros(T, dtype=torch.long)) - plain(x, mask=veil)).abs().max().item()
    assert err < 1e-5, (
        f"with every position 0 the rotation is the identity and RoPEAttention must equal Room 6.3's causal MHA with the same "
        f"weights; differs by {err:.2e}. The rotation is the ONLY thing you added."
    )


def test_the_gallery_does_not_care_where_the_scroll_starts():
    gallery = _gallery(seed=11)
    x = _x(seed=12)
    here = gallery(x)
    later = gallery(x, positions=torch.arange(T) + 100)
    err = (here - later).abs().max().item()
    assert err < 1e-5, (
        f"positions 100..{100 + T - 1} gave a different output than 0..{T - 1}, by {err:.2e}. Every score depends only on the "
        "distance between seats and v is not rotated, so the whole layer is shift-invariant. This is what makes decoding "
        "against a KV cache correct: the cached keys keep their absolute positions, the new query gets its own, and only "
        "the differences matter."
    )


def test_position_free_heads_are_permutation_equivariant_and_the_gallery_is_not():
    gallery = _gallery(seed=13, causal=False)
    plain = heads.MultiHeadAttention(D_MODEL, HEADS).eval()
    plain.load_state_dict(gallery.state_dict())
    x = _x(seed=14)
    perm = torch.tensor([5, 2, 7, 0, 3, 6, 1, 4])
    err = (plain(x)[:, perm] - plain(x[:, perm])).abs().max().item()
    assert err < 1e-5, (
        f"Room 6.3's attention without positions IS permutation-equivariant: shuffle the tokens and the outputs shuffle the same "
        f"way. It moved by {err:.2e}; Room 6.3 is broken, fix it first."
    )
    moved = (gallery(x)[:, perm] - gallery(x[:, perm])).abs().max().item()
    assert moved > 1e-3, (
        "with RoPE (and no veil) shuffling the tokens must NOT merely shuffle the outputs: each token now carries its seat "
        f"number in q and k. Yours moved by only {moved:.2e}; the rotation is not reaching the scores."
    )


# ---------------------------------------------------------- GroupedQueryAttention
def _gqa(n_kv_heads, seed=20, d_model=D_MODEL, n_heads=HEADS):
    torch.manual_seed(seed)
    gqa = room.GroupedQueryAttention(d_model, n_heads, n_kv_heads)
    for name in ("d_model", "n_heads", "n_kv_heads", "head_dim", "groups", "q_proj", "k_proj", "v_proj", "proj"):
        assert hasattr(gqa, name), f"GroupedQueryAttention is missing self.{name}. The trial reads these by name; see __init__'s docstring."
    return gqa.eval()


def _ref_gqa(gqa, x, mask):
    """torch's own grouped attention (enable_gqa), or repeat_interleave on older torch."""
    Bn, Tn, d = x.shape
    q = F.linear(x, gqa.q_proj.weight, gqa.q_proj.bias).view(Bn, Tn, gqa.n_heads, HD).transpose(1, 2)
    k = F.linear(x, gqa.k_proj.weight, gqa.k_proj.bias).view(Bn, Tn, gqa.n_kv_heads, HD).transpose(1, 2)
    v = F.linear(x, gqa.v_proj.weight, gqa.v_proj.bias).view(Bn, Tn, gqa.n_kv_heads, HD).transpose(1, 2)
    try:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    except TypeError:  # torch < 2.5
        g = gqa.n_heads // gqa.n_kv_heads
        out = F.scaled_dot_product_attention(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), attn_mask=mask)
    return F.linear(out.transpose(1, 2).reshape(Bn, Tn, d), gqa.proj.weight, gqa.proj.bias)


VEIL = torch.tril(torch.ones(T, T, dtype=torch.bool))


def test_the_grouped_gallery_carves_fewer_keys_and_values():
    gqa = _gqa(n_kv_heads=2)
    assert gqa.head_dim == HD and gqa.groups == 2, f"head_dim = d_model // n_heads = {HD}, groups = n_heads // n_kv_heads = 2; got {gqa.head_dim}, {gqa.groups}"
    assert tuple(gqa.q_proj.weight.shape) == (D_MODEL, D_MODEL), f"q_proj is Linear(d_model, d_model): weight (32, 32), got {tuple(gqa.q_proj.weight.shape)}"
    assert tuple(gqa.k_proj.weight.shape) == (2 * HD, D_MODEL), (
        f"k_proj is Linear(d_model, n_kv_heads * head_dim): weight ({2 * HD}, {D_MODEL}), got {tuple(gqa.k_proj.weight.shape)}. Only n_kv_heads heads of keys."
    )
    assert tuple(gqa.v_proj.weight.shape) == (2 * HD, D_MODEL), f"v_proj weight should be ({2 * HD}, {D_MODEL}), got {tuple(gqa.v_proj.weight.shape)}"
    assert tuple(gqa.proj.weight.shape) == (D_MODEL, D_MODEL), "proj is Linear(d_model, d_model)"
    n = sum(p.numel() for p in gqa.parameters())
    expected = 2 * D_MODEL**2 + 2 * D_MODEL + 2 * 2 * HD * (D_MODEL + 1)
    assert n == expected, (
        f"parameters: q_proj d^2 + d, proj d^2 + d, k_proj and v_proj n_kv*hd*(d + 1) each = {expected}; yours has {n}. "
        f"(Room 6.3's MHA has {4 * D_MODEL**2 + 4 * D_MODEL}.)"
    )
    with pytest.raises(ValueError):
        room.GroupedQueryAttention(D_MODEL, 4, 3)  # 4 heads cannot share 3 kv heads evenly
    with pytest.raises(ValueError):
        room.GroupedQueryAttention(30, 4, 2)


def test_every_query_head_in_a_group_hears_the_same_key_head():
    gqa = _gqa(n_kv_heads=2)
    g = torch.Generator().manual_seed(21)
    kv = torch.randn(B, 2, T, HD, generator=g)
    expanded = gqa.expand_kv(kv)
    assert expanded.shape == (B, HEADS, T, HD), f"expand_kv: (B, n_kv_heads, T, hd) -> (B, n_heads, T, hd) = {(B, HEADS, T, HD)}, got {tuple(expanded.shape)}"
    for h in range(HEADS):
        assert torch.equal(expanded[:, h], kv[:, h // gqa.groups]), (
            f"query head {h} must read kv head {h // gqa.groups} (h // groups): heads 0, 1 share kv head 0 and heads 2, 3 share kv head 1. "
            "That is repeat_interleave(groups, dim=1); .repeat() gives 0, 1, 0, 1 and torch's enable_gqa disagrees."
        )
    split = gqa.split_heads(torch.randn(B, T, 2 * HD, generator=g), 2)
    assert split.shape == (B, 2, T, HD), f"split_heads(x, n_heads=2) on (B, T, 2*hd) gives (B, 2, T, hd), got {tuple(split.shape)}"


def test_with_as_many_kv_heads_as_heads_it_is_the_thousand_heads():
    gqa = _gqa(n_kv_heads=HEADS, seed=22)
    mha = heads.MultiHeadAttention(D_MODEL, HEADS).eval()
    with torch.no_grad():
        mha.qkv.weight.copy_(torch.cat([gqa.q_proj.weight, gqa.k_proj.weight, gqa.v_proj.weight], dim=0))
        mha.qkv.bias.copy_(torch.cat([gqa.q_proj.bias, gqa.k_proj.bias, gqa.v_proj.bias], dim=0))
        mha.proj.weight.copy_(gqa.proj.weight)
        mha.proj.bias.copy_(gqa.proj.bias)
    x = _x(seed=23)
    out = gqa(x, mask=VEIL)
    assert out.shape == (B, T, D_MODEL), f"forward returns (B, T, d_model), got {tuple(out.shape)}"
    err = (out - mha(x, mask=VEIL)).abs().max().item()
    assert err < 1e-5, (
        f"with n_kv_heads == n_heads (groups = 1) GQA is Room 6.3's MHA with its fused qkv cut into q_proj, k_proj, v_proj; "
        f"differs by {err:.2e}. Check the head split (view then transpose) and the scale sqrt(head_dim)."
    )


def test_with_one_kv_head_it_is_multi_query_attention():
    gqa = _gqa(n_kv_heads=1, seed=24)
    assert tuple(gqa.k_proj.weight.shape) == (HD, D_MODEL), f"with one kv head, k_proj produces a single head: weight ({HD}, {D_MODEL})"
    x = _x(seed=25)
    q = F.linear(x, gqa.q_proj.weight, gqa.q_proj.bias).view(B, T, HEADS, HD).transpose(1, 2)
    k = F.linear(x, gqa.k_proj.weight, gqa.k_proj.bias)[:, None].expand(B, HEADS, T, HD)  # the ONE key head, seen by all
    v = F.linear(x, gqa.v_proj.weight, gqa.v_proj.bias)[:, None].expand(B, HEADS, T, HD)
    ref = F.linear(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D_MODEL), gqa.proj.weight, gqa.proj.bias)
    err = (gqa(x, mask=VEIL) - ref).abs().max().item()
    assert err < 1e-5, (
        f"multi-query attention: every one of the {HEADS} query heads attends over the same single key/value head; differs by {err:.2e}."
    )


def test_grouped_heads_agree_with_torch():
    gqa = _gqa(n_kv_heads=2, seed=26)
    x = _x(seed=27)
    err = (gqa(x, mask=VEIL) - _ref_gqa(gqa, x, VEIL)).abs().max().item()
    assert err < 1e-5, (
        f"with 4 heads and 2 kv heads, yours differs from torch's scaled_dot_product_attention(enable_gqa=True) by {err:.2e}. "
        "torch pairs query head h with kv head h // groups, exactly repeat_interleave."
    )
    err = (gqa(x) - _ref_gqa(gqa, x, None)).abs().max().item()
    assert err < 1e-5, f"without a mask the grouped gallery is bidirectional and must still match torch; differs by {err:.2e}"


def test_the_grouped_gallery_does_not_leak_behind_the_veil():
    gqa = _gqa(n_kv_heads=2, seed=28)
    assert _leaks(lambda x: gqa(x, mask=VEIL), _x(seed=29)) is False, (
        "with the causal mask, perturbing the future changed the past. The mask goes before the softmax, on every group."
    )


# ---------------------------------------------------------------- the KV cache
def test_kv_cache_bytes_counts_keys_and_values_for_every_layer():
    n = room.kv_cache_bytes(n_layer=32, T=4096, n_kv_heads=32, head_dim=128, bytes_per=2)
    assert type(n) is int, f"kv_cache_bytes returns a Python int, got {type(n).__name__}"
    assert n == 2 * 32 * 4096 * 32 * 128 * 2 == 2**31, (
        f"a 7B-class MHA model (32 layers, 32 kv heads, head_dim 128) at 4096 tokens in bf16 caches exactly 2 GiB = {2**31} bytes "
        f"per sequence; yours says {n}. The factor 2 in front is keys AND values."
    )
    assert room.kv_cache_bytes(1, 1, 1, 1, 1) == 2, "one layer, one token, one head, one dim, one byte: a key and a value = 2 bytes"
    assert room.kv_cache_bytes(4, 128, 4, 32, 4) == 2 * 4 * 128 * 4 * 32 * 4, "the Chronicler's own cache (Floor 7 sizes, fp32) is 524,288 bytes"


def test_fewer_kv_heads_shrink_the_cache_by_the_group_size():
    full = room.kv_cache_bytes(32, 8192, 32, 128)
    grouped = room.kv_cache_bytes(32, 8192, 8, 128)
    assert full == 4 * 2**30 and grouped == 2**30, (
        f"Llama-3-8B-like at 8192 tokens, bf16: 4 GiB with 32 kv heads, 1 GiB with 8. Got {full} and {grouped}."
    )
    assert full // grouped == 4, "the saving is n_heads / n_kv_heads = 32 / 8 = 4: four times as many sequences fit in the same memory"
    assert room.kv_cache_bytes(32, 8192, 1, 128) * 32 == full, "multi-query attention (1 kv head) saves the full factor n_heads"


# --------------------------------------------------------------- the prophecy
PROPHECY_KEYS = [
    "the pair that turns fastest",
    "shift every position by 7, the scores are",
    "raise the base from 1e4 to 1e6, far positions turn",
    "kv-cache saving for 32 heads and 8 kv heads",
]


def _truth(key):
    if key == PROPHECY_KEYS[0]:
        theta = _ref_freqs(64)
        return "first" if theta.argmax().item() == 0 else "last"
    if key == PROPHECY_KEYS[1]:
        q, k = _qk(seed=30)
        pos = torch.arange(T)
        a = _ref_rope(q, pos) @ _ref_rope(k, pos).transpose(-2, -1)
        b = _ref_rope(q, pos + 7) @ _ref_rope(k, pos + 7).transpose(-2, -1)
        return "unchanged" if torch.allclose(a, b, atol=1e-5) else "changed"
    if key == PROPHECY_KEYS[2]:
        far = torch.tensor([1024.0])
        a4 = (far[:, None] * _ref_freqs(64, 1e4)).sum()
        a6 = (far[:, None] * _ref_freqs(64, 1e6)).sum()
        return "less" if a6 < a4 else "more"
    return (2 * 32 * 8192 * 32 * 128 * 2) // (2 * 32 * 8192 * 8 * 128 * 2)


@pytest.mark.parametrize("key", PROPHECY_KEYS)
def test_the_prophecy_of_the_gallery(key):
    prediction = room.GALLERY_PROPHECY.get(key)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied {key!r}. Fill in GALLERY_PROPHECY.")
    truth = _truth(key)
    assert prediction == truth, f"For {key!r}: the measurement says {truth!r}; you prophesied {prediction!r}."


def test_the_prophecy_is_complete():
    assert set(room.GALLERY_PROPHECY) == set(PROPHECY_KEYS), "Do not add, remove or rename the prophecy's cases."
    words = {"first", "last", "unchanged", "changed", "less", "more"}
    for key, value in room.GALLERY_PROPHECY.items():
        if key == PROPHECY_KEYS[3]:
            assert value is None or (isinstance(value, int) and not isinstance(value, bool)), "the kv-cache saving is an int"
        else:
            assert value is None or value in words, f"{key!r}: answers are one of {sorted(words)}"
