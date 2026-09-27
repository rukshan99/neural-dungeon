"""SECRET - THE CUSTOM WHISPER

A hand-written backward is judged three ways: against the composed op,
against finite differences (gradcheck, float64), and by reading the source
for the shape of a proper autograd.Function.
"""

import pytest

from dungeon.scrutiny import is_stub, names_called_in
from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

whisper = load_room(__file__, "secret_custom_whisper")

pytestmark = pytest.mark.secret

LAM = 0.5


def _away_from_the_kinks(shape, seed):
    """float64 inputs whose |x| is at least 0.05 from LAM, so finite differences are honest."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g, dtype=torch.float64)
    near = (x.abs() - LAM).abs() < 0.05
    return torch.where(near, x + 0.1 * torch.sign(x), x)


def test_the_whisper_is_an_autograd_function():
    assert issubclass(whisper.SoftThreshold, torch.autograd.Function), "SoftThreshold must subclass torch.autograd.Function."
    if is_stub(whisper.SoftThreshold.forward) or is_stub(whisper.SoftThreshold.backward):
        raise NotImplementedError("SoftThreshold.forward()/backward() are unwritten")
    assert "save_for_backward" in names_called_in(whisper.SoftThreshold.forward), (
        "forward() never calls ctx.save_for_backward(...). That is how the mask reaches backward()."
    )
    called = names_called_in(whisper.SoftThreshold.backward)
    assert not ({"backward", "grad", "gradcheck"} & called), (
        "backward() must be written by hand: no autograd calls inside it."
    )
    assert "apply" in names_called_in(whisper.soft_threshold), "soft_threshold() must go through SoftThreshold.apply(...)."


def test_forward_matches_the_composed_op():
    x = _away_from_the_kinks((6, 5), seed=1).float()
    got = whisper.soft_threshold(x, LAM)
    expected = whisper.soft_threshold_composed(x, LAM)
    assert got.shape == x.shape and got.dtype == x.dtype, f"Output should match the input's shape and dtype; got {tuple(got.shape)} {got.dtype}."
    assert torch.allclose(got, expected, atol=1e-6), "soft_threshold(x) must equal sign(x) * max(|x| - lam, 0)."
    inside = torch.tensor([-0.4, -0.1, 0.0, 0.2, 0.49])
    assert bool((whisper.soft_threshold(inside, LAM) == 0).all()), "Inside the band |x| <= lam the output is exactly 0."
    assert whisper.soft_threshold(torch.tensor([2.0, -2.0]), LAM).tolist() == [1.5, -1.5], "Outside the band the output is shrunk towards 0 by lam."


def test_backward_matches_autograd_on_the_composed_op():
    x = _away_from_the_kinks((8, 4), seed=2).requires_grad_(True)
    upstream = torch.randn(8, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    (whisper.soft_threshold(x, LAM) * upstream).sum().backward()
    got = x.grad.clone()
    x.grad = None
    (whisper.soft_threshold_composed(x, LAM) * upstream).sum().backward()
    assert torch.allclose(got, x.grad), (
        "Your backward disagrees with autograd's differentiation of the composed op. "
        "grad_x = grad_output where |x| > lam, else 0; multiply, do not just return the mask."
    )


def test_the_gradient_is_one_outside_the_band_and_zero_inside():
    x = torch.tensor([-2.0, -0.2, 0.0, 0.3, 2.0], requires_grad=True)
    whisper.soft_threshold(x, LAM).sum().backward()
    assert x.grad.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0], f"Expected gradient [1, 0, 0, 0, 1] for lam=0.5; got {x.grad.tolist()}."


def test_gradcheck_in_float64_hears_no_lie():
    x = _away_from_the_kinks((5, 7), seed=4).requires_grad_(True)
    ok = torch.autograd.gradcheck(lambda t: whisper.soft_threshold(t, LAM), (x,), eps=1e-6, atol=1e-5, raise_exception=False)
    assert ok, "gradcheck failed: finite differences of your forward do not match your hand-written backward."


def test_lam_needs_no_gradient():
    x = _away_from_the_kinks((3,), seed=5).requires_grad_(True)
    out = whisper.soft_threshold(x, LAM)
    assert out.requires_grad and out.grad_fn is not None, "The output must be attached to the graph through your Function."
    out.sum().backward()  # a backward that returns the wrong number of values raises here
    assert x.grad is not None
