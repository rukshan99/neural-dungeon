"""TRIAL 12.3 - THE BATCHER

Eight petitioners in one boat. The sandbags must be deaf and mute, the positions
must start at each petitioner's first word, and every row must decode exactly
as it would have alone.
"""

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

from dungeon.artifacts.tiny_gpt import load_pretrained, read_corpus  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

room = load_room(__file__, "room_3_the_batcher")

torch.manual_seed(123)
torch.set_num_threads(1)
MODEL, TOK, _ = load_pretrained()
CFG = MODEL.cfg
TEXT = read_corpus()
N_NEW = 40  # measured: batched and individual greedy agree on all 40

PROMPTS = [
    "Ruel carried a mirror that showed the validation set. ",
    '"Count the axes," Okonkwo told Tadeo. ',
    "The Learning-Rate Lich",
    "Below them, the Scriptorium of Tokens. Above them, the Forge of Layers.",
]
FOUR = [TOK.encode(p) for p in PROMPTS]


def _reference(prompt_ids, n):
    with torch.no_grad():
        return MODEL.generate(torch.tensor([prompt_ids]), n, temperature=0)[0, len(prompt_ids):].tolist()


# ------------------------------------------------------------------ left_pad
def test_left_pad_lines_the_prompts_up_at_the_right_edge():
    idx, mask, pos = room.left_pad([[5, 6, 7], [8], [1, 2]], pad_id=0)
    assert idx.shape == (3, 3) and idx.dtype == torch.long, f"idx should be (3, 3) int64, got {tuple(idx.shape)} {idx.dtype}"
    assert idx.tolist() == [[5, 6, 7], [0, 0, 8], [0, 1, 2]], f"Pads go on the LEFT: got {idx.tolist()}"
    assert mask.dtype == torch.bool, f"attention_mask must be bool, got {mask.dtype}"
    assert mask.tolist() == [[True, True, True], [False, False, True], [False, True, True]], f"mask: {mask.tolist()}"
    assert pos.tolist() == [[0, 1, 2], [0, 0, 0], [0, 0, 1]], (
        f"position ids should start at 0 at each row's first real token and be 0 on pads; got {pos.tolist()}"
    )


def test_left_pad_refuses_an_empty_petition():
    with pytest.raises(ValueError):
        room.left_pad([[1, 2], []], pad_id=0)


# ------------------------------------------------------------------ masked_attention
def test_masked_attention_with_a_plain_causal_mask_is_the_reference_attention():
    block = MODEL.blocks[0]
    x = torch.randn(2, 7, CFG.n_embd)
    allowed = torch.tril(torch.ones(7, 7, dtype=torch.bool))[None, None]
    with torch.no_grad():
        got = room.masked_attention(block, x, allowed)
        ref = block.attn(x)
    assert torch.allclose(got, ref, atol=1e-5), (
        f"With a plain causal mask your attention should equal block.attn(x); max diff {(got - ref).abs().max():.2e}."
    )


def test_padded_keys_get_exactly_zero_weight_and_padded_queries_stay_finite():
    block = MODEL.blocks[1]
    x = torch.randn(1, 6, CFG.n_embd)
    key_mask = torch.tensor([[False, False, True, True, True, True]])
    allowed = torch.tril(torch.ones(6, 6, dtype=torch.bool))[None, None] & key_mask[:, None, None, :]
    with torch.no_grad():
        got = room.masked_attention(block, x, allowed)
        ref = block.attn(x[:, 2:])  # the real tokens on their own
    assert torch.isfinite(got).all(), (
        "NaN or inf in the output. The two leading pads are queries that see only padded keys: every score is -inf, "
        "the softmax is NaN. nan_to_num after the softmax (or a finite fill value) keeps those rows finite."
    )
    assert torch.allclose(got[:, 2:], ref, atol=1e-5), (
        f"Real positions differ from attending over the real tokens alone by {(got[:, 2:] - ref).abs().max():.2e}. "
        "Padded keys must receive weight exactly 0 (fill -inf BEFORE the softmax)."
    )


# ------------------------------------------------------------------ forward_masked
def test_forward_masked_matches_the_plain_forward_on_every_real_token():
    idx, mask, pos = room.left_pad(FOUR, pad_id=0)
    with torch.no_grad():
        logits = room.forward_masked(MODEL, idx, mask, pos)
    assert logits.shape == (4, idx.shape[1], CFG.vocab_size), f"logits should be (B, T, V), got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "Every logit, even at padded positions, must be finite."
    for b, p in enumerate(FOUR):
        with torch.no_grad():
            ref, _ = MODEL(torch.tensor([p]))
        diff = (logits[b, -len(p):] - ref[0]).abs().max()
        assert diff < 1e-3, (
            f"Row {b} (length {len(p)}): logits at the real positions differ from the unpadded forward by {diff:.2e}. "
            "Check the position ids (0 at the first real token) and the key-padding half of the mask."
        )


def test_the_sandbags_are_deaf_and_mute():
    """Changing the pad token id must not change any real logit."""
    idx_a, mask, pos = room.left_pad(FOUR, pad_id=0)
    idx_b, _, _ = room.left_pad(FOUR, pad_id=41)
    with torch.no_grad():
        la = room.forward_masked(MODEL, idx_a, mask, pos)
        lb = room.forward_masked(MODEL, idx_b, mask, pos)
    for b, p in enumerate(FOUR):
        diff = (la[b, -len(p):] - lb[b, -len(p):]).abs().max()
        assert diff < 1e-4, f"Row {b}: the pad id leaked into the real positions (max change {diff:.2e})."


def test_the_batch_is_bounded_by_block_size():
    with pytest.raises(ValueError):
        room.forward_masked(
            MODEL,
            torch.zeros(1, CFG.block_size + 1, dtype=torch.long),
            torch.ones(1, CFG.block_size + 1, dtype=torch.bool),
            torch.arange(CFG.block_size + 1)[None],
        )


# ------------------------------------------------------------------ batched_greedy_generate
def test_batched_greedy_matches_each_prompt_alone():
    with torch.no_grad():
        batched = room.batched_greedy_generate(MODEL, FOUR, N_NEW)
    assert isinstance(batched, list) and len(batched) == 4, "Return a list with one token list per prompt."
    for b, p in enumerate(FOUR):
        ref = _reference(p, N_NEW)
        got = list(batched[b])
        assert len(got) == N_NEW, f"Row {b}: expected {N_NEW} generated tokens (and only the generated ones), got {len(got)}."
        if got != ref:
            first = next(i for i in range(N_NEW) if got[i] != ref[i])
            pytest.fail(
                f"Row {b} (prompt length {len(p)}) diverges at token {first}: alone it writes {TOK.decode(ref)!r}, "
                f"in the batch {TOK.decode(got)!r}. The padding is influencing the real tokens."
            )


def test_eos_stops_one_row_while_the_others_carry_on():
    refs = [_reference(p, N_NEW) for p in FOUR]
    eos = refs[0][0]  # row 0's very first token: it should come back empty
    with torch.no_grad():
        batched = room.batched_greedy_generate(MODEL, FOUR, N_NEW, eos_id=eos)
    assert list(batched[0]) == [], f"Row 0 emits eos ({TOK.decode([eos])!r}) first, so its output should be []; got {batched[0]}."
    for b in range(1, 4):
        expected = refs[b][:refs[b].index(eos)] if eos in refs[b] else refs[b]
        assert list(batched[b]) == expected, (
            f"Row {b} should stop right before its first eos: expected {TOK.decode(expected)!r}, got {TOK.decode(batched[b])!r}."
        )


def test_the_boat_stops_at_block_size():
    long_prompt = TOK.encode(TEXT[3000:3000 + CFG.block_size - 8])
    with torch.no_grad():
        out = room.batched_greedy_generate(MODEL, [long_prompt, [1, 2, 3]], 20)
    assert len(out[0]) == 8 and len(out[1]) == 8, (
        f"The longest prompt leaves room for 8 tokens before block_size; got {len(out[0])} and {len(out[1])}."
    )
