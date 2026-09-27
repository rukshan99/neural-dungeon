"""TRIAL 7.1 - THE NORM AND THE NONLINEARITY

Four bricks are weighed against PyTorch's own: LayerNorm and RMSNorm
(outputs and gradients), GELU (exact and tanh-approximate) and the MLP
(shapes, names, parameter count, and whether it can wear the reference's
weights).
"""

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dungeon.artifacts import tiny_gpt  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

room = load_room(__file__, "room_1_norm_and_nonlinearity")

torch.manual_seed(71)


def _paired_layernorms(ndim, eps=1e-5, bias=True, dtype=torch.float64):
    """The learner's LayerNorm and nn.LayerNorm with identical random weight/bias."""
    mine = room.LayerNorm(ndim, eps=eps, bias=bias).to(dtype)
    ref = nn.LayerNorm(ndim, eps=eps, bias=bias).to(dtype)
    with torch.no_grad():
        mine.weight.copy_(torch.randn(ndim, dtype=dtype))
        if bias:
            mine.bias.copy_(torch.randn(ndim, dtype=dtype))
    ref.load_state_dict(mine.state_dict())
    return mine, ref


# ------------------------------------------------------------------- LayerNorm
def test_layernorm_is_a_module_with_a_weight_and_a_bias():
    ln = room.LayerNorm(16)
    assert isinstance(ln, nn.Module), "LayerNorm must subclass nn.Module."
    names = dict(ln.named_parameters())
    assert set(names) == {"weight", "bias"}, (
        f"LayerNorm should own exactly two parameters named weight and bias, found {sorted(names)}. "
        "The checkpoint you will load in room 3 uses those names."
    )
    assert names["weight"].shape == (16,) and names["bias"].shape == (16,), "weight and bias are both (ndim,)."
    assert torch.all(names["weight"] == 1.0) and torch.all(names["bias"] == 0.0), (
        "A fresh LayerNorm is the identity on normalised inputs: weight starts at ones, bias at zeros."
    )


@pytest.mark.parametrize("shape", [(4, 32), (2, 5, 32), (3, 7, 1, 8)], ids=["2d", "3d", "4d"])
def test_layernorm_matches_torch_to_a_millionth(shape):
    mine, ref = _paired_layernorms(shape[-1])
    x = torch.randn(*shape, dtype=torch.float64) * 3 + 1
    diff = (mine(x) - ref(x)).abs().max().item()
    assert diff < 1e-6, (
        f"LayerNorm output differs from nn.LayerNorm by {diff:.2e} on input shape {shape}. "
        "Normalise over the LAST dim only, use the biased variance (divide by D, not D-1), "
        "and put eps inside the square root: (x - mean) / sqrt(var + eps) * weight + bias."
    )


def test_layernorm_gradients_match_torch_too():
    mine, ref = _paired_layernorms(24)
    x = torch.randn(3, 6, 24, dtype=torch.float64)
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    upstream = torch.randn(3, 6, 24, dtype=torch.float64)
    (mine(x1) * upstream).sum().backward()
    (ref(x2) * upstream).sum().backward()
    for name, got, want in [
        ("input", x1.grad, x2.grad),
        ("weight", mine.weight.grad, ref.weight.grad),
        ("bias", mine.bias.grad, ref.bias.grad),
    ]:
        diff = (got - want).abs().max().item()
        assert diff < 1e-6, (
            f"The gradient with respect to the {name} differs from nn.LayerNorm's by {diff:.2e}. "
            "If the forward matches but the backward does not, you probably detached something "
            "(.detach(), .data, torch.no_grad) or used an in-place op autograd cannot see through."
        )


def test_layernorm_normalises_each_position_on_its_own():
    ln = room.LayerNorm(8)
    x = torch.randn(2, 3, 8) * 5 + 7  # wildly un-normalised
    out = ln(x)
    means = out.mean(dim=-1)
    variances = out.var(dim=-1, unbiased=False)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5), (
        f"Each (batch, position) vector should have mean 0 after LayerNorm; got means {means.flatten().tolist()}. "
        "Reduce over dim=-1 with keepdim=True, not over the batch."
    )
    assert torch.allclose(variances, torch.ones_like(variances), atol=1e-3), (
        f"Each vector should have variance 1 after LayerNorm; got {variances.flatten().tolist()}."
    )


def test_layernorm_respects_eps_when_the_variance_is_tiny():
    mine, ref = _paired_layernorms(8, eps=0.1)
    x = torch.randn(4, 8, dtype=torch.float64) * 1e-2
    diff = (mine(x) - ref(x)).abs().max().item()
    assert diff < 1e-6, (
        f"With eps=0.1 and a nearly constant input, your output differs from torch's by {diff:.2e}. "
        "eps is added to the variance BEFORE the square root, and it must be the eps you were given."
    )


def test_layernorm_can_go_without_a_bias():
    mine, ref = _paired_layernorms(8, bias=False)
    assert mine.bias is None, "With bias=False the bias attribute should be None."
    assert set(mine.state_dict()) == {"weight"}, "With bias=False the state_dict holds only 'weight'."
    x = torch.randn(5, 8, dtype=torch.float64)
    assert torch.allclose(mine(x), ref(x), atol=1e-6)


def test_layernorm_works_in_float32_where_the_tower_lives():
    mine, ref = _paired_layernorms(128, dtype=torch.float32)
    x = torch.randn(4, 16, 128)
    diff = (mine(x) - ref(x)).abs().max().item()
    assert diff < 1e-5, f"float32 LayerNorm differs from torch by {diff:.2e}; expected < 1e-5."


# --------------------------------------------------------------------- RMSNorm
def _paired_rmsnorms(ndim, eps=1e-6, dtype=torch.float64):
    """The learner's RMSNorm and nn.RMSNorm with the same random weight."""
    mine = room.RMSNorm(ndim, eps=eps).to(dtype)
    ref = nn.RMSNorm(ndim, eps=eps).to(dtype)
    with torch.no_grad():
        mine.weight.copy_(torch.randn(ndim, dtype=dtype))
    ref.load_state_dict(mine.state_dict())
    return mine, ref


def test_rmsnorm_owns_exactly_ndim_parameters_and_no_bias():
    rms = room.RMSNorm(16)
    assert isinstance(rms, nn.Module), "RMSNorm must subclass nn.Module."
    names = dict(rms.named_parameters())
    assert set(names) == {"weight"}, (
        f"RMSNorm owns exactly one parameter, `weight`; found {sorted(names)}. "
        "There is no bias and nothing to centre with: that is the point of it."
    )
    assert names["weight"].shape == (16,), f"weight is (ndim,) = (16,), got {tuple(names['weight'].shape)}."
    assert torch.all(names["weight"] == 1.0), "A fresh RMSNorm only rescales: weight starts at ones."
    n = sum(p.numel() for p in room.RMSNorm(48).parameters())
    assert n == 48, f"RMSNorm(48) has ndim = 48 parameters (LayerNorm has 2 * ndim); yours has {n}."


@pytest.mark.parametrize("shape", [(4, 32), (2, 5, 32), (3, 7, 1, 8)], ids=["2d", "3d", "4d"])
def test_rmsnorm_matches_torch_to_a_millionth(shape):
    mine, ref = _paired_rmsnorms(shape[-1])
    x = torch.randn(*shape, dtype=torch.float64) * 3 + 1
    diff = (mine(x) - ref(x)).abs().max().item()
    assert diff < 1e-6, (
        f"RMSNorm output differs from nn.RMSNorm by {diff:.2e} on input shape {shape}. "
        "Do NOT subtract the mean: y = x * rsqrt(mean(x^2, -1, keepdim=True) + eps) * weight. "
        "Reduce over the LAST dim only, and eps goes inside the root."
    )


def test_rmsnorm_gradients_match_torch_too():
    mine, ref = _paired_rmsnorms(24)
    x = torch.randn(3, 6, 24, dtype=torch.float64)
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    upstream = torch.randn(3, 6, 24, dtype=torch.float64)
    (mine(x1) * upstream).sum().backward()
    (ref(x2) * upstream).sum().backward()
    for name, got, want in [("input", x1.grad, x2.grad), ("weight", mine.weight.grad, ref.weight.grad)]:
        diff = (got - want).abs().max().item()
        assert diff < 1e-6, (
            f"The gradient with respect to the {name} differs from nn.RMSNorm's by {diff:.2e}. "
            "If the forward matches but the backward does not, you detached something or wrote in place."
        )


def test_rmsnorm_respects_the_eps_it_is_given():
    mine, ref = _paired_rmsnorms(8, eps=0.1)
    x = torch.randn(4, 8, dtype=torch.float64) * 1e-2
    diff = (mine(x) - ref(x)).abs().max().item()
    assert diff < 1e-6, (
        f"With eps=0.1 and a tiny input, your RMSNorm differs from torch's by {diff:.2e}. "
        "Use the eps you were given, added to mean(x^2) BEFORE the square root."
    )


@pytest.mark.parametrize("c", [2.5, 40.0])
def test_rmsnorm_does_not_care_how_loud_the_input_is(c):
    rms = room.RMSNorm(32).to(torch.float64)
    x = torch.randn(3, 5, 32, dtype=torch.float64)
    diff = (rms(c * x) - rms(x)).abs().max().item()
    assert diff < 1e-5, (
        f"RMSNorm(c * x) should equal RMSNorm(x) for c = {c} (up to eps); they differ by {diff:.2e}. "
        "Dividing by the root mean square removes the scale of the input: that is the norm's job."
    )


def test_rmsnorm_is_not_layernorm_in_disguise():
    rms = room.RMSNorm(16).to(torch.float64)
    ln = nn.LayerNorm(16, eps=1e-6, bias=False).to(torch.float64)
    x = torch.randn(4, 16, dtype=torch.float64) + 3.0  # a healthy non-zero mean per row
    gap = (rms(x) - ln(x)).abs().max().item()
    assert gap > 0.1, (
        f"On rows with mean 3 your RMSNorm and a bias-free LayerNorm agree to {gap:.2e}. "
        "RMSNorm does not subtract the mean; a vector's mean survives into its output."
    )
    centred = x - x.mean(dim=-1, keepdim=True)
    agree = (rms(centred) - ln(centred)).abs().max().item()
    assert agree < 1e-6, (
        f"On rows that already have mean 0 the two norms are the same function, yet they differ by {agree:.2e}. "
        "With mean 0, var = mean(x^2): check your mean square and your eps."
    )


def test_rmsnorm_works_in_float32_where_the_tower_lives():
    mine, ref = _paired_rmsnorms(128, dtype=torch.float32)
    x = torch.randn(4, 16, 128)
    diff = (mine(x) - ref(x)).abs().max().item()
    assert diff < 1e-5, f"float32 RMSNorm differs from torch by {diff:.2e}; expected < 1e-5."


# ------------------------------------------------------------------------ GELU
def test_gelu_is_exact_to_a_millionth():
    x = torch.linspace(-8, 8, 4001)
    diff = (room.gelu(x) - F.gelu(x)).abs().max().item()
    assert diff < 1e-6, (
        f"gelu differs from F.gelu by {diff:.2e}. The exact form is 0.5 * x * (1 + erf(x / sqrt(2))). "
        "Did you use tanh here? That is gelu_tanh, the other function."
    )


def test_gelu_is_not_relu_wearing_a_hat():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    out = room.gelu(x)
    assert out[1] < 0, f"gelu(-1) should be about -0.159, a small NEGATIVE number, not {out[1].item():.3f}. GELU lets a little of the negative side through."
    assert abs(out[2].item()) < 1e-7, "gelu(0) is exactly 0."
    assert torch.allclose(out, torch.tensor([-0.0040, -0.1587, 0.0, 0.8413, 2.9960]), atol=1e-3)


def test_gelu_tanh_approximates_within_a_thousandth_and_is_the_real_approximation():
    x = torch.linspace(-8, 8, 4001)
    to_exact = (room.gelu_tanh(x) - F.gelu(x)).abs().max().item()
    assert to_exact < 1e-3, (
        f"gelu_tanh strays {to_exact:.2e} from the exact GELU; the tanh approximation stays within about 5e-4. "
        "0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."
    )
    to_tanh = (room.gelu_tanh(x) - F.gelu(x, approximate="tanh")).abs().max().item()
    assert to_tanh < 1e-6, (
        f"gelu_tanh differs from torch's tanh approximation by {to_tanh:.2e}. "
        "If it matches the exact GELU instead, you returned gelu() and called it an approximation."
    )


# ------------------------------------------------------------------------- MLP
def test_mlp_has_fc_and_proj_with_the_right_shapes():
    mlp = room.MLP(32)
    assert isinstance(mlp, nn.Module)
    assert isinstance(getattr(mlp, "fc", None), nn.Linear), "MLP needs an nn.Linear called `fc` (the widening layer)."
    assert isinstance(getattr(mlp, "proj", None), nn.Linear), "MLP needs an nn.Linear called `proj` (the narrowing layer)."
    assert (mlp.fc.in_features, mlp.fc.out_features) == (32, 128), (
        f"fc should map D -> 4D, i.e. 32 -> 128, got {mlp.fc.in_features} -> {mlp.fc.out_features}."
    )
    assert (mlp.proj.in_features, mlp.proj.out_features) == (128, 32), (
        f"proj should map 4D -> D, i.e. 128 -> 32, got {mlp.proj.in_features} -> {mlp.proj.out_features}."
    )
    x = torch.randn(2, 5, 32)
    assert mlp(x).shape == (2, 5, 32), f"The MLP must keep the shape (B, T, D); got {tuple(mlp(x).shape)}."


@pytest.mark.parametrize("d", [16, 48])
def test_mlp_param_count_is_8d2_plus_5d(d):
    n = sum(p.numel() for p in room.MLP(d).parameters())
    assert n == 8 * d * d + 5 * d, (
        f"An MLP with D={d} should have 8D^2 + 5D = {8 * d * d + 5 * d} parameters "
        f"(fc: 4D*D + 4D, proj: D*4D + D); yours has {n}. Are both layers biased? Is the hidden width 4D?"
    )


def test_mlp_bends_with_gelu_not_relu():
    d = 6
    mlp = room.MLP(d)
    with torch.no_grad():
        mlp.fc.weight.zero_()
        mlp.fc.bias.zero_()
        mlp.proj.weight.zero_()
        mlp.proj.bias.zero_()
        mlp.fc.weight[:d, :d] = torch.eye(d)  # first D hidden units copy the input
        mlp.proj.weight[:, :d] = torch.eye(d)  # and are copied straight back out
    x = torch.linspace(-3, 3, d).view(1, d)
    out = mlp(x)
    assert torch.allclose(out, F.gelu(x), atol=1e-6), (
        f"With identity weights the MLP should reduce to gelu(x) = {F.gelu(x).flatten().tolist()}, "
        f"but produced {out.flatten().tolist()}. Negative inputs that come out as exactly 0 mean ReLU snuck in."
    )


def test_mlp_wears_the_reference_weights_and_agrees():
    cfg = tiny_gpt.GPTConfig(vocab_size=8, n_embd=32)
    reference = tiny_gpt.build_modules()["MLP"](cfg)
    with torch.no_grad():
        for p in reference.parameters():
            p.normal_(std=0.3)
    mine = room.MLP(32)
    try:
        mine.load_state_dict(reference.state_dict(), strict=True)
    except RuntimeError as exc:
        pytest.fail(
            "Your MLP could not wear the reference weights: "
            + str(exc).splitlines()[0]
            + " The parameter names must be fc.weight, fc.bias, proj.weight, proj.bias and nothing else."
        )
    x = torch.randn(3, 7, 32)
    diff = (mine(x) - reference(x)).abs().max().item()
    assert diff < 1e-5, (
        f"Same weights, different answers: max difference {diff:.2e}. The reference is proj(gelu(fc(x))). "
        "Check the order of the layers and that the nonlinearity is the exact GELU."
    )
