"""SECRET - THE TILED GAZE

Exact attention that never builds the (Tq, Tk) matrix. The trial checks the
numbers against plain attention for several block sizes, then watches every
tensor operation you perform: a full score matrix, or a call to torch's own
attention, and the slit closes.
"""

import pytest

from dungeon.scrutiny import is_stub, names_called_in, python_loops_in
from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
F = torch.nn.functional
TorchFunctionMode = torch.overrides.TorchFunctionMode

slit = load_room(__file__, "secret_the_tiled_gaze")

pytestmark = pytest.mark.secret

B, TQ, TK, D = 2, 12, 40, 16
BLOCK_SIZES = [1, 3, 7, 8, 16, 40, 64]  # 3 and 7 do not divide 40; 40 and 64 are one block


def _draw(seed=13, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, TQ, D, generator=g) * scale
    k = torch.randn(B, TK, D, generator=g)
    v = torch.randn(B, TK, D, generator=g)
    return q, k, v


class _Watcher(TorchFunctionMode):
    """Records the shape of every tensor any torch operation returns, and every function called."""

    def __init__(self):
        super().__init__()
        self.shapes: set[tuple[int, ...]] = set()
        self.funcs: set[str] = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.funcs.add(getattr(func, "__name__", str(func)))
        out = func(*args, **(kwargs or {}))
        self._record(out)
        return out

    def _record(self, obj):
        if isinstance(obj, torch.Tensor):
            self.shapes.add(tuple(obj.shape))
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                self._record(item)


# ------------------------------------------------------------------ warm-up
@pytest.mark.parametrize("block_size", [1, 4, 5, 8, 32])
def test_the_running_max_and_sum_reproduce_logsumexp(block_size):
    g = torch.Generator().manual_seed(21)
    x = torch.randn(3, 5, 20, generator=g) * 5
    out = slit.blockwise_logsumexp(x, block_size)
    ref = torch.logsumexp(x, dim=-1)
    assert out.shape == ref.shape, f"blockwise_logsumexp should reduce the last axis: expected {tuple(ref.shape)}, got {tuple(out.shape)}"
    err = (out - ref).abs().max().item()
    assert err < 1e-5, (
        f"block_size={block_size}: max |yours - torch.logsumexp| = {err:.2e}. When a block raises the max, "
        "rescale the old sum by exp(m_old - m_new) before adding the new block's exp(x - m_new)."
    )


# ---------------------------------------------------------------- the gaze
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_the_tiled_gaze_sees_exactly_what_the_whole_gaze_sees(block_size):
    q, k, v = _draw()
    out = slit.online_softmax_attention(q, k, v, block_size)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == ref.shape, f"out should be (B, Tq, dv) = {tuple(ref.shape)}, got {tuple(out.shape)}"
    err = (out - ref).abs().max().item()
    assert err < 1e-5, (
        f"block_size={block_size} (Tk={TK}): max |tiled - plain| = {err:.2e}. "
        + ("The last block is shorter than block_size; slicing k[:, s:s+block_size] handles that on its own. "
           if TK % block_size else "")
        + "Check that acc and l are both rescaled by alpha = exp(m_old - m_new) every block, and that you divide acc by l at the end."
    )


def test_the_tiled_gaze_survives_scores_in_the_hundreds():
    q, k, v = _draw(seed=14, scale=100.0)
    out = slit.online_softmax_attention(q, k, v, 8)
    assert torch.isfinite(out).all(), "scores of magnitude ~100 overflowed exp(). The running max m keeps every exponent <= 0."
    ref = F.scaled_dot_product_attention(q, k, v)
    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"with huge scores the tiled gaze differs from torch by {err:.2e}"


def test_the_slit_admits_no_full_score_matrix():
    fn = slit.online_softmax_attention
    if is_stub(fn):
        raise NotImplementedError("online_softmax_attention() is unwritten")
    called = names_called_in(fn)
    assert "scaled_dot_product_attention" not in called, "calling torch's attention is not walking through the slit."
    assert python_loops_in(fn), "the tiled gaze walks the keys block by block; there must be a loop over blocks."
    q, k, v = _draw()
    with _Watcher() as watcher:
        fn(q, k, v, 8)
    forbidden = {(TQ, TK), (TK, TQ)}
    carved = sorted(s for s in watcher.shapes if len(s) >= 2 and s[-2:] in forbidden)
    assert not carved, (
        f"a tensor with trailing shape (Tq, Tk) = {(TQ, TK)} was created: {carved}. The whole point is that the score "
        f"matrix is never materialised; the largest tile may be (B, Tq, block_size) = {(B, TQ, 8)}."
    )
    assert not any("scaled_dot_product_attention" in f for f in watcher.funcs), (
        "torch's scaled_dot_product_attention was called somewhere inside. Do the recurrence yourself."
    )
