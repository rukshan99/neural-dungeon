"""TRIAL 7.2 - THE BLOCK

Causal self-attention is checked for shape, for leaking the future, and for
agreeing with the reference when it wears the reference's weights. The block
is checked for its four named parts, its pre-norm wiring and its parameter
count, and then it too must wear the reference's weights exactly.
"""

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

import torch.nn as nn  # noqa: E402

from dungeon.artifacts import tiny_gpt  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

room = load_room(__file__, "room_2_the_block")

torch.manual_seed(72)

D, H, BLOCK = 32, 4, 16
CFG = tiny_gpt.GPTConfig(vocab_size=8, block_size=BLOCK, n_layer=1, n_head=H, n_embd=D)
REFERENCE = tiny_gpt.build_modules()


def _loud(module: nn.Module, std: float = 0.3) -> nn.Module:
    """Reference modules with large random weights, so that any wiring mistake is visible."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("weight") and p.dim() == 1:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))  # LayerNorm gains near 1
            else:
                p.normal_(std=std)
    return module


def _wear(mine: nn.Module, reference: nn.Module, what: str) -> None:
    try:
        mine.load_state_dict(reference.state_dict(), strict=True)
    except RuntimeError as exc:
        first = str(exc).splitlines()[0]
        pytest.fail(
            f"Your {what} could not wear the reference weights: {first} "
            "The names must match the nanoGPT layout (c_attn, c_proj / ln_1, attn, ln_2, mlp.fc, mlp.proj) "
            "and the causal mask must not be in the state_dict (register it with persistent=False)."
        )


# ------------------------------------------------------------------ attention
def test_attention_keeps_the_shape_of_the_stream():
    attn = room.CausalSelfAttention(D, H, BLOCK)
    x = torch.randn(2, 10, D)
    out = attn(x)
    assert out.shape == (2, 10, D), f"Attention maps (B, T, D) to (B, T, D); got {tuple(out.shape)} from (2, 10, {D})."


def test_attention_has_a_fused_qkv_projection_and_an_output_projection():
    attn = room.CausalSelfAttention(D, H, BLOCK)
    assert isinstance(getattr(attn, "c_attn", None), nn.Linear), "Attention needs an nn.Linear named c_attn."
    assert isinstance(getattr(attn, "c_proj", None), nn.Linear), "Attention needs an nn.Linear named c_proj."
    assert (attn.c_attn.in_features, attn.c_attn.out_features) == (D, 3 * D), (
        f"c_attn is ONE linear layer producing q, k and v at once: D -> 3D = {D} -> {3 * D}, "
        f"got {attn.c_attn.in_features} -> {attn.c_attn.out_features}."
    )
    assert (attn.c_proj.in_features, attn.c_proj.out_features) == (D, D), "c_proj maps D -> D."


def test_attention_refuses_a_head_count_that_does_not_divide_d():
    with pytest.raises((ValueError, AssertionError)):
        room.CausalSelfAttention(D, 5, BLOCK)


def test_the_mask_is_not_saved_in_the_state_dict():
    attn = room.CausalSelfAttention(D, H, BLOCK)
    keys = set(attn.state_dict())
    assert keys == {"c_attn.weight", "c_attn.bias", "c_proj.weight", "c_proj.bias"}, (
        f"state_dict keys are {sorted(keys)}. The causal mask is derived from block_size, not learned, "
        "so it must not be saved: register_buffer('mask', ..., persistent=False). Otherwise the "
        "Chronicler's checkpoint (which has no mask) will refuse to load in room 3."
    )


def test_no_token_can_see_the_future():
    attn = room.CausalSelfAttention(D, H, BLOCK).eval()
    x = torch.randn(2, 12, D)
    with torch.no_grad():
        base = attn(x)
        tampered = x.clone()
        tampered[:, 7:] = torch.randn(2, 5, D) * 10  # rewrite positions 7..11
        out = attn(tampered)
    unchanged = (out[:, :7] - base[:, :7]).abs().max().item()
    assert unchanged < 1e-5, (
        f"Changing positions 7..11 changed the outputs at positions 0..6 by up to {unchanged:.2e}. "
        "That is information flowing backwards in time. Fill scores where key > query with -inf "
        "BEFORE the softmax (tril mask), so those weights become exactly 0."
    )
    changed = (out[:, 7:] - base[:, 7:]).abs().max().item()
    assert changed > 1e-3, "Positions 7..11 were rewritten but their outputs did not change; the attention is not reading its input."


def test_every_token_can_see_the_past():
    attn = room.CausalSelfAttention(D, H, BLOCK).eval()
    x = torch.randn(1, 6, D)
    with torch.no_grad():
        base = attn(x)
        tampered = x.clone()
        tampered[:, 0] += 5.0
        out = attn(tampered)
    moved = (out[:, 1:] - base[:, 1:]).abs().max().item()
    assert moved > 1e-4, (
        "Changing token 0 did not affect any later position. Attention that ignores the past is just "
        "an MLP; check that the mask is lower-triangular (tril), not upper, and not the identity."
    )


def test_attention_wears_the_reference_weights_and_agrees():
    reference = _loud(REFERENCE["CausalSelfAttention"](CFG)).eval()
    mine = room.CausalSelfAttention(D, H, BLOCK).eval()
    _wear(mine, reference, "CausalSelfAttention")
    x = torch.randn(3, BLOCK, D)
    with torch.no_grad():
        diff = (mine(x) - reference(x)).abs().max().item()
    assert diff < 1e-5, (
        f"Same weights, different attention: max difference {diff:.2e}. The usual suspects, in order: "
        "the q/k/v split must be along the LAST dim in the order q, k, v; heads are made with "
        ".view(B, T, H, hd).transpose(1, 2); scores are divided by sqrt(hd) (the HEAD dim, not D); "
        "softmax is over the last dim (the keys); heads are merged with .transpose(1, 2).contiguous().view(B, T, D)."
    )


def test_attention_works_for_sequences_shorter_than_block_size():
    reference = _loud(REFERENCE["CausalSelfAttention"](CFG)).eval()
    mine = room.CausalSelfAttention(D, H, BLOCK).eval()
    _wear(mine, reference, "CausalSelfAttention")
    for T in (1, 5):
        x = torch.randn(2, T, D)
        with torch.no_grad():
            diff = (mine(x) - reference(x)).abs().max().item()
        assert diff < 1e-5, (
            f"With T={T} < block_size={BLOCK} the outputs differ by {diff:.2e}. Slice the mask to [:T, :T]."
        )


# ---------------------------------------------------------------------- block
def test_block_has_its_four_named_parts():
    block = room.Block(D, H, BLOCK)
    for name in ("ln_1", "attn", "ln_2", "mlp"):
        assert isinstance(getattr(block, name, None), nn.Module), f"Block needs a sub-module called {name}."
    assert isinstance(block.attn, room.CausalSelfAttention), "block.attn should be your CausalSelfAttention."
    assert hasattr(block.mlp, "fc") and hasattr(block.mlp, "proj"), "block.mlp should be the room 1 MLP (fc, proj)."
    assert block.ln_1.weight.shape == (D,) and block.ln_2.weight.shape == (D,), "ln_1 and ln_2 normalise over D."


def test_block_param_count_is_12d2_plus_13d():
    n = sum(p.numel() for p in room.Block(D, H, BLOCK).parameters())
    want = 12 * D * D + 13 * D
    assert n == want, (
        f"A block with D={D} has 12D^2 + 13D = {want} parameters: attention 4D^2 + 4D, MLP 8D^2 + 5D, "
        f"two LayerNorms 4D. Yours has {n}."
    )


def test_block_is_the_identity_when_its_sublayers_fall_silent():
    block = room.Block(D, H, BLOCK).eval()
    with torch.no_grad():
        for p in (block.attn.c_proj.weight, block.attn.c_proj.bias, block.mlp.proj.weight, block.mlp.proj.bias):
            p.zero_()
    x = torch.randn(2, 8, D)
    with torch.no_grad():
        out = block(x)
    assert torch.allclose(out, x, atol=1e-6), (
        "With c_proj and mlp.proj zeroed, both sub-layers output zeros, so the block must return x unchanged. "
        "Yours did not: the residual connection is missing (x + f(x), not f(x)) or the norm was applied to the stream itself."
    )


def test_block_is_pre_norm_not_post_norm():
    block = _loud(room.Block(D, H, BLOCK)).eval()
    x = torch.randn(2, 9, D)
    with torch.no_grad():
        out = block(x)
        pre = x + block.attn(block.ln_1(x))
        pre = pre + block.mlp(block.ln_2(pre))
        post = block.ln_1(x + block.attn(x))
        post = block.ln_2(post + block.mlp(post))
    assert torch.allclose(out, pre, atol=1e-5), (
        "The block is not wired as pre-norm. It must be exactly: x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)). "
        + ("You built post-norm (norm AFTER the residual add): that is the original 2017 layout, not GPT-2's." if torch.allclose(out, post, atol=1e-4) else "")
    )


def test_block_wears_the_reference_weights_exactly():
    reference = _loud(REFERENCE["Block"](CFG)).eval()
    mine = room.Block(D, H, BLOCK).eval()
    _wear(mine, reference, "Block")
    x = torch.randn(3, 20 if BLOCK >= 20 else BLOCK, D) * 2
    with torch.no_grad():
        diff = (mine(x) - reference(x)).abs().max().item()
    assert diff < 1e-5, (
        f"Same weights, different block: max difference {diff:.2e}. Your attention and MLP pass on their own, "
        "so look at the wiring: ln_1 feeds attn, ln_2 feeds mlp, and each result is ADDED to the stream."
    )
