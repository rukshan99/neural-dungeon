"""BOSS FIGHT - THE ORACLE WHO PEEKS

Phase 1: your detectors name every (query, key) pair through which each
         oracle hears the future, and therefore every compromised position.
Phase 2: your padding detector catches the oracle that veils the wrong axis.
Phase 3: your FixedOracle passes both detectors and matches torch.
Phase 4: the prophecy - your verdicts against the trial's own detectors.
"""

import pytest

from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
F = torch.nn.functional

boss = load_room(__file__, "boss_oracle_who_peeks")

pytestmark = pytest.mark.boss

T, D, SEED = 8, 8, 0
ORACLE_NAMES = ["oracle_honest", "oracle_mask_after_softmax", "oracle_off_by_one", "oracle_padding_on_keys", "oracle_padding_on_queries"]
CAUSAL_ORACLES = ORACLE_NAMES[:3]
PADDING_ORACLES = ORACLE_NAMES[3:]


# ------------------------------------------------- the trial's own detectors
def _ref_leak_pairs(attn_fn, T, d, seed):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(1, T, d, generator=g) for _ in range(3))
    base = attn_fn(q, k, v)
    pairs = []
    for j in range(1, T):
        q2, k2, v2 = q.clone(), k.clone(), v.clone()
        for tensor in (q2, k2, v2):
            tensor[:, j] = torch.randn(1, d, generator=g)
        out = attn_fn(q2, k2, v2)
        pairs.extend((t, j) for t in range(j) if not torch.allclose(out[:, t], base[:, t], atol=1e-6, rtol=0.0))
    return sorted(pairs)


def _ref_padding_leaks(attn_fn, lengths, d=8, seed=0):
    lengths = torch.as_tensor(lengths)
    B, Tmax = int(lengths.shape[0]), int(lengths.max())
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, Tmax, d, generator=g) for _ in range(3))
    base = attn_fn(q, k, v, lengths)
    real = torch.arange(Tmax)[None, :] < lengths[:, None]
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    for tensor in (q2, k2, v2):
        tensor[~real] = torch.randn(int((~real).sum()), d, generator=g)
    out = attn_fn(q2, k2, v2, lengths)
    return not torch.allclose(out[real], base[real], atol=1e-6, rtol=0.0)


def _peeks_at_one_seat(q, k, v):
    """Honest everywhere except that seat 3 hears seat 5. For checking the report is exact."""
    T = q.shape[1]
    veil = torch.tril(torch.ones(T, T, dtype=torch.bool))
    veil[3, 5] = True
    scores = (q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5).masked_fill(~veil, float("-inf"))
    return scores.softmax(dim=-1) @ v


ALL_FUTURE_PAIRS = [(t, j) for t in range(T) for j in range(t + 1, T)]
ONE_STEP_PAIRS = [(t, t + 1) for t in range(T - 1)]


# --------------------------------------------------------------------- phase 1
def test_phase_1_the_honest_oracle_leaks_nowhere():
    assert boss.leak_pairs(boss.oracle_honest, T, D, SEED) == [], "oracle_honest is honest; your detector reports leaks."
    assert boss.detect_leaks(boss.oracle_honest, T, D, SEED) == [], "no position of an honest oracle is compromised"


def test_phase_1_the_veil_after_softmax_leaks_from_every_future_seat():
    pairs = boss.leak_pairs(boss.oracle_mask_after_softmax, T, D, SEED)
    assert [tuple(p) for p in pairs] == ALL_FUTURE_PAIRS, (
        f"Masking after the softmax leaves every future key in the denominator, so EVERY (t, j) with j > t leaks: "
        f"{len(ALL_FUTURE_PAIRS)} pairs. Your report has {len(pairs)}: {pairs}"
    )
    positions = boss.detect_leaks(boss.oracle_mask_after_softmax, T, D, SEED)
    assert list(positions) == list(range(T - 1)), (
        f"every position except the last ({T - 1}, which has no future) is compromised: expected {list(range(T - 1))}, got {positions}"
    )


def test_phase_1_the_off_by_one_oracle_hears_exactly_one_seat_ahead():
    pairs = boss.leak_pairs(boss.oracle_off_by_one, T, D, SEED)
    assert [tuple(p) for p in pairs] == ONE_STEP_PAIRS, (
        f"tril(diagonal=1) lets t hear t+1 and nothing further: expected exactly {ONE_STEP_PAIRS}, got {pairs}. "
        "Perturb ONE future position at a time or you cannot tell this oracle from the last one."
    )
    positions = boss.detect_leaks(boss.oracle_off_by_one, T, D, SEED)
    assert list(positions) == list(range(T - 1)), f"positions 0..{T - 2} are compromised; got {positions}"


def test_phase_1_the_report_is_exact_down_to_a_single_seat():
    pairs = boss.leak_pairs(_peeks_at_one_seat, T, D, SEED)
    assert [tuple(p) for p in pairs] == [(3, 5)], (
        f"this oracle is honest except that seat 3 hears seat 5. Expected [(3, 5)], got {pairs}."
    )
    assert list(boss.detect_leaks(_peeks_at_one_seat, T, D, SEED)) == [3]


def test_phase_1_the_report_is_made_of_plain_sorted_ints():
    positions = boss.detect_leaks(boss.oracle_off_by_one, T, D, SEED)
    assert isinstance(positions, list) and all(type(p) is int for p in positions), "detect_leaks returns a list of Python ints"
    assert positions == sorted(set(positions)), "sorted, no duplicates"
    pairs = boss.leak_pairs(boss.oracle_off_by_one, T, D, SEED)
    assert all(type(t) is int and type(j) is int and j > t for t, j in pairs), "leak_pairs holds (t, j) int pairs with j > t"


# --------------------------------------------------------------------- phase 2
def test_phase_2_padding_on_the_keys_is_honest():
    verdict = boss.detect_padding_leak(boss.oracle_padding_on_keys, [5, 3, 1])
    assert isinstance(verdict, bool), f"detect_padding_leak returns a Python bool, got {type(verdict).__name__}"
    assert verdict is False, "oracle_padding_on_keys masks the blanks along the keys. It is honest; your detector accuses it."


def test_phase_2_padding_on_the_queries_lets_real_seats_hear_the_blanks():
    assert boss.detect_padding_leak(boss.oracle_padding_on_queries, [5, 3, 1]) is True, (
        "veiling padded QUERIES silences rows nobody reads and leaves every real query hearing the padded keys. "
        "Rewrite the padding and the real outputs move. Your detector missed it."
    )


def test_phase_2_a_batch_without_padding_cannot_leak():
    assert boss.detect_padding_leak(boss.oracle_padding_on_queries, [4, 4, 4]) is False, (
        "when every scroll is full there is no padding to perturb; the verdict must be False."
    )


# --------------------------------------------------------------------- phase 3
def _fixed(seed=1, d_model=32, n_heads=4):
    torch.manual_seed(seed)
    return boss.FixedOracle(d_model, n_heads).eval()


def _reference(oracle, x, lengths):
    B, Tn, d = x.shape
    H, hd = oracle.n_heads, d // oracle.n_heads
    q, k, v = F.linear(x, oracle.qkv.weight, oracle.qkv.bias).split(d, dim=-1)
    split = lambda t: t.view(B, Tn, H, hd).transpose(1, 2)  # noqa: E731
    real = torch.arange(Tn)[None, :] < lengths[:, None]
    mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool))[None, None] & real[:, None, None, :]
    out = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=mask)
    return F.linear(out.transpose(1, 2).reshape(B, Tn, d), oracle.proj.weight, oracle.proj.bias)


def test_phase_3_the_fixed_oracle_has_the_shape_and_the_carvings():
    oracle = _fixed()
    n = sum(p.numel() for p in oracle.parameters())
    assert n == 4 * 32**2 + 4 * 32, f"FixedOracle keeps Room 6.3's layers: 4*d^2 + 4*d = {4 * 32**2 + 4 * 32} parameters, got {n}"
    x = torch.randn(3, T, 32, generator=torch.Generator().manual_seed(2))
    out = oracle(x, torch.tensor([8, 5, 2]))
    assert out.shape == x.shape, f"forward returns (B, T, d_model) = {tuple(x.shape)}, got {tuple(out.shape)}"


def test_phase_3_the_fixed_oracle_passes_the_leak_detector():
    oracle = _fixed()
    pairs = _ref_leak_pairs(lambda q, k, v: oracle(q), T, 32, SEED)
    assert pairs == [], f"the trial's own detector found leaks in FixedOracle: {pairs}. The veil is torn."


def test_phase_3_the_fixed_oracle_passes_the_padding_detector():
    oracle = _fixed()
    assert _ref_padding_leaks(lambda q, k, v, lengths: oracle(q, lengths), [6, 3, 1], d=32) is False, (
        "rewriting the padded positions changed a real position of FixedOracle. Mask the padding along the keys."
    )


def test_phase_3_the_fixed_oracle_matches_torch_at_every_real_seat():
    oracle = _fixed(seed=3)
    lengths = torch.tensor([8, 6, 3, 0])
    x = torch.randn(4, T, 32, generator=torch.Generator().manual_seed(4))
    out = oracle(x, lengths)
    assert torch.isfinite(out).all(), "FixedOracle produced NaN or inf (a length-0 scroll gives fully masked rows: use safe_softmax)"
    ref = _reference(oracle, x, lengths)
    for b, L in enumerate(lengths.tolist()):
        if L == 0:
            continue
        err = (out[b, :L] - ref[b, :L]).abs().max().item()
        assert err < 1e-5, f"sequence {b} (length {L}) differs from torch's attention with the combined mask by {err:.2e}"


def test_phase_3_without_lengths_every_seat_is_real():
    oracle = _fixed(seed=5)
    x = torch.randn(2, T, 32, generator=torch.Generator().manual_seed(6))
    a = oracle(x)
    b = oracle(x, torch.tensor([T, T]))
    assert torch.allclose(a, b, atol=1e-6), "lengths=None must mean 'no padding', i.e. lengths = [T] * B"


# --------------------------------------------------------------------- phase 4
def _truth(name):
    fn = getattr(boss, name)
    if name in PADDING_ORACLES:
        return "leaks" if _ref_padding_leaks(fn, [5, 3, 1]) else "honest"
    return "leaks" if _ref_leak_pairs(fn, T, D, SEED) else "honest"


def test_phase_4_the_prophecy_names_every_oracle():
    assert set(boss.ORACLE_PROPHECY) == set(ORACLE_NAMES), "Do not add or remove oracles from the prophecy; judge them."
    assert all(v in (None, "leaks", "honest") for v in boss.ORACLE_PROPHECY.values()), 'Verdicts are "leaks" or "honest".'


@pytest.mark.parametrize("name", ORACLE_NAMES)
def test_phase_4_the_prophecy_is_judged(name):
    prediction = boss.ORACLE_PROPHECY.get(name)
    if prediction is None:
        raise NotImplementedError(f"You have not judged {name}. Fill in ORACLE_PROPHECY.")
    truth = _truth(name)
    assert prediction == truth, f"You said {name} {prediction!r}; the detectors say it {truth!r}."


def test_the_oracles_remain_as_cursed_as_they_were():
    """The fixtures are the monsters. Fixing them is not defeating them."""
    assert _ref_leak_pairs(boss.oracle_mask_after_softmax, T, D, SEED) == ALL_FUTURE_PAIRS, (
        "oracle_mask_after_softmax has been edited. Put the curse back; write detectors instead."
    )
    assert _ref_leak_pairs(boss.oracle_off_by_one, T, D, SEED) == ONE_STEP_PAIRS, "oracle_off_by_one has been edited. Restore it."
    assert _ref_padding_leaks(boss.oracle_padding_on_queries, [5, 3, 1]) is True, "oracle_padding_on_queries has been edited. Restore it."
