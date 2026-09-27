"""TRIAL 5.3 - THE EMBEDDING WELL

Two ways down the well must return the same vector and the same gradient; the
padding row must stay dry; the positional tables must follow the formula.
"""

import math

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

from dungeon.scrutiny import is_stub, names_called_in, operators_used_in  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

room = load_room(__file__, "room_3_embedding_well")

V, D = 7, 4


def _table(seed=0):
    torch.manual_seed(seed)
    return torch.randn(V, D)


# --------------------------------------------------------------- lookup vs matmul
def test_the_well_returns_the_row_of_the_id():
    W = _table()
    ids = torch.tensor([[1, 2, 1, 0], [3, 3, 0, 6]])
    out = room.embedding_lookup(W, ids)
    assert tuple(out.shape) == (2, 4, D), f"(B, T) ids into a (V, D) table give (B, T, D); got {tuple(out.shape)}."
    assert torch.equal(out[0, 0], W[1]) and torch.equal(out[1, 3], W[6]), "out[b, t] must be row ids[b, t] of the table."
    assert torch.equal(out[0, 0], out[0, 2]), "The same id always comes back as the same vector."


def test_lookup_indexes_and_matmul_multiplies():
    for name in ("embedding_lookup", "embedding_as_matmul"):
        if is_stub(getattr(room, name)):
            raise NotImplementedError(f"{name}() is unwritten")
    uses_matmul = "MatMult" in operators_used_in(room.embedding_as_matmul) or bool(
        names_called_in(room.embedding_as_matmul) & {"matmul", "mm", "einsum", "bmm"}
    )
    assert uses_matmul, "embedding_as_matmul() must use a matrix product (@, matmul, mm or einsum) on a one-hot."
    lookup_calls = names_called_in(room.embedding_lookup)
    assert not (lookup_calls & {"one_hot", "matmul", "mm", "einsum", "embedding"}), (
        f"embedding_lookup() should be plain indexing, but it calls {sorted(lookup_calls)}."
    )


def test_one_hot_times_table_is_the_same_lookup():
    W = _table(1)
    ids = torch.tensor([[1, 2, 1, 0], [3, 3, 0, 6]])
    a = room.embedding_lookup(W, ids)
    b = room.embedding_as_matmul(W, ids)
    assert tuple(b.shape) == tuple(a.shape), f"Both routes must give (B, T, D); matmul gave {tuple(b.shape)}."
    assert torch.allclose(a, b, atol=1e-6), "one_hot(ids) @ W picks exactly the rows of W. The outputs differ."
    assert b.dtype == W.dtype, f"The one-hot must be cast to the table's dtype ({W.dtype}); got {b.dtype}."


def test_both_routes_carry_the_same_gradient_a_scatter_add():
    W1 = _table(2).requires_grad_(True)
    W2 = W1.detach().clone().requires_grad_(True)
    ids = torch.tensor([[1, 2, 1, 0], [3, 3, 0, 6]])
    torch.manual_seed(3)
    upstream = torch.randn(2, 4, D)
    (room.embedding_lookup(W1, ids) * upstream).sum().backward()
    (room.embedding_as_matmul(W2, ids) * upstream).sum().backward()
    assert W1.grad is not None and W2.grad is not None, "Gradient must reach the table on both routes."
    assert torch.allclose(W1.grad, W2.grad, atol=1e-6), "Indexing and one-hot matmul must have identical gradients."
    assert torch.all(W1.grad[4] == 0) and torch.all(W1.grad[5] == 0), (
        "Rows 4 and 5 were never looked up, so their gradient is zero. An embedding gradient is sparse."
    )
    expected_row1 = upstream[0, 0] + upstream[0, 2]
    assert torch.allclose(W1.grad[1], expected_row1, atol=1e-6), (
        "Row 1 was used twice (positions [0,0] and [0,2]) so its gradient is the SUM of both upstream vectors."
    )


# --------------------------------------------------------------- padding_idx
def test_the_padding_row_starts_dry_and_stays_dry():
    torch.manual_seed(4)
    emb = room.token_embedding(V, D, padding_idx=0)
    assert isinstance(emb, torch.nn.Embedding), f"token_embedding() must return an nn.Embedding, got {type(emb).__name__}."
    assert tuple(emb.weight.shape) == (V, D), f"The table must be (vocab_size, d_model) = {(V, D)}, got {tuple(emb.weight.shape)}."
    assert emb.padding_idx == 0, f"padding_idx should be 0, got {emb.padding_idx}."
    assert torch.all(emb.weight[0] == 0), "nn.Embedding initializes the padding_idx row to zeros."
    ids = torch.tensor([[2, 5, 0, 0], [1, 0, 0, 0]])  # a right-padded batch
    emb(ids).sum().backward()
    assert torch.all(emb.weight.grad[0] == 0), (
        f"The padding row received gradient {emb.weight.grad[0].tolist()}. With padding_idx set it must be all zeros."
    )
    assert torch.any(emb.weight.grad[2] != 0) and torch.any(emb.weight.grad[5] != 0), "Real rows still get gradient."
    plain = torch.nn.Embedding(V, D)
    plain(ids).sum().backward()
    assert torch.any(plain.weight.grad[0] != 0), "(sanity) without padding_idx the pad row DOES learn; that is the bug we avoid."


def test_padding_idx_is_honoured_wherever_it_is_placed():
    emb = room.token_embedding(V, D, padding_idx=3)
    assert emb.padding_idx == 3 and torch.all(emb.weight[3] == 0)


# --------------------------------------------------------------- sinusoidal
def _reference_pe(max_len, d_model):
    pe = [[0.0] * d_model for _ in range(max_len)]
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            pe[pos][i] = math.sin(angle)
            pe[pos][i + 1] = math.cos(angle)
    return torch.tensor(pe, dtype=torch.float32)


def test_sinusoidal_table_matches_the_formula():
    pe = room.sinusoidal_positional_encoding(50, 16)
    assert tuple(pe.shape) == (50, 16), f"Expected (max_len, d_model) = (50, 16), got {tuple(pe.shape)}."
    assert pe.dtype == torch.float32, f"The table should be float32, got {pe.dtype}."
    ref = _reference_pe(50, 16)
    worst = (pe - ref).abs().max().item()
    assert torch.allclose(pe, ref, atol=1e-4), (
        f"Largest deviation from the formula is {worst:.2e}. Check: even columns sin, odd columns cos, "
        "and column pair i uses 10000^(2i/d_model), where 2i is the EVEN column index."
    )


def test_position_zero_is_zeros_and_ones():
    pe = room.sinusoidal_positional_encoding(8, 6)
    assert torch.allclose(pe[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])), (
        f"At pos 0 every angle is 0, so PE[0] = [sin 0, cos 0, ...] = [0, 1, 0, 1, ...]; got {pe[0].tolist()}."
    )
    assert pe.min() >= -1.0 and pe.max() <= 1.0, "Sines and cosines live in [-1, 1]."


def test_the_first_column_is_a_plain_sine_of_the_position():
    pe = room.sinusoidal_positional_encoding(10, 4)
    expected = torch.sin(torch.arange(10, dtype=torch.float32))
    assert torch.allclose(pe[:, 0], expected, atol=1e-5), "Column 0 has rate 1/10000^0 = 1: PE[pos, 0] = sin(pos)."


def test_dot_products_depend_on_the_offset_not_the_position():
    pe = room.sinusoidal_positional_encoding(64, 32)
    for k in (1, 5, 17):
        dots = (pe[:-k] * pe[k:]).sum(dim=1)  # PE[p] . PE[p + k] for every p
        assert torch.allclose(dots, dots[0].expand_as(dots), atol=1e-3), (
            f"PE[p].PE[p+{k}] should be the same for every p (it equals sum_i cos(w_i * {k})); "
            f"yours ranges from {dots.min():.4f} to {dots.max():.4f}."
        )


def test_an_odd_width_is_refused():
    with pytest.raises(ValueError):
        room.sinusoidal_positional_encoding(4, 5)


# --------------------------------------------------------------- learned
def test_learned_positions_add_a_row_per_position():
    torch.manual_seed(5)
    pos_emb = room.LearnedPositionalEmbedding(max_len=12, d_model=D)
    assert isinstance(pos_emb, torch.nn.Module)
    n_params = sum(p.numel() for p in pos_emb.parameters())
    assert n_params == 12 * D, f"The module should own exactly max_len * d_model = {12 * D} parameters, it owns {n_params}."
    x = torch.zeros(2, 5, D)
    out = pos_emb(x)
    assert tuple(out.shape) == (2, 5, D), f"(B, T, D) in, (B, T, D) out; got {tuple(out.shape)}."
    assert torch.allclose(out[0], out[1]), "With zero input, every batch element sees the same position vectors."
    assert not torch.allclose(out[0, 0], out[0, 1]), "Different positions must receive different vectors."
    table = next(p for p in pos_emb.parameters() if tuple(p.shape) == (12, D))
    assert torch.allclose(out[0, 3], table[3]), "Position t receives row t of the table."


def test_learned_positions_receive_gradient_only_where_used():
    pos_emb = room.LearnedPositionalEmbedding(max_len=12, d_model=D)
    pos_emb(torch.zeros(3, 4, D)).sum().backward()
    table = next(p for p in pos_emb.parameters() if tuple(p.shape) == (12, D))
    assert torch.all(table.grad[:4] != 0), "Positions 0..3 were used by 3 batch elements each and must get gradient."
    assert torch.all(table.grad[4:] == 0), "Positions 4..11 were never used; their gradient is zero."


def test_a_sequence_longer_than_the_table_has_nowhere_to_look():
    pos_emb = room.LearnedPositionalEmbedding(max_len=4, d_model=D)
    with pytest.raises(ValueError):
        pos_emb(torch.zeros(1, 5, D))
