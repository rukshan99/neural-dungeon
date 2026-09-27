"""TRIAL 4.2 - THE MODULE FORGE

An MLP is poured into the nn.Module mould and inspected: registration, shape,
activation placement, parameter count, initialisation, a hand-made forward
with the forge cold, a round trip through bytes, and a frozen layer.
"""

import io

import pytest

from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

room = load_room(__file__, "room_2_module_forge")

nn = torch.nn
F = torch.nn.functional


class ColdIron(RuntimeError):
    """Raised when manual_forward leans on the forge instead of the parameters."""


def _cold(name):
    def cold(*args, **kwargs):
        raise ColdIron(f"{name} is cold iron in this trial. Use the weights and matmul yourself.")

    return cold


# ------------------------------------------------------------------ structure
def test_the_forge_stamps_logits_of_the_right_shape():
    torch.manual_seed(0)
    model = room.MLP([64, 32, 10])
    out = model(torch.rand(5, 8, 8))
    assert out.shape == (5, 10), f"(5, 8, 8) runes through MLP([64, 32, 10]) should give (5, 10) logits, got {tuple(out.shape)}."
    assert model(torch.rand(5, 64)).shape == (5, 10), "Already-flat input must work too: flatten from dim 1."


def test_every_layer_is_registered_with_the_module():
    model = room.MLP([64, 32, 10])
    names = dict(model.named_parameters())
    expected = {"layers.0.weight": (32, 64), "layers.0.bias": (32,), "layers.1.weight": (10, 32), "layers.1.bias": (10,)}
    assert set(names) == set(expected), (
        f"named_parameters() gives {sorted(names)}; expected {sorted(expected)}. "
        "A plain Python list of layers is invisible to the module: use nn.ModuleList and call it `layers`."
    )
    for name, shape in expected.items():
        assert tuple(names[name].shape) == shape, f"{name} has shape {tuple(names[name].shape)}, expected {shape}. nn.Linear(in, out).weight is (out, in)."


def test_the_last_layer_speaks_raw_logits():
    torch.manual_seed(0)
    model = room.MLP([64, 32, 10], activation=nn.Sigmoid)
    out = model(torch.rand(64, 8, 8))
    assert bool(((out < 0) | (out > 1)).any()), (
        "Every output lies in (0, 1): you applied the activation after the last layer too. "
        "The MLP returns logits; the loss function does the squashing."
    )


@pytest.mark.parametrize("activation", [nn.ReLU, nn.Tanh])
def test_the_activation_is_the_one_you_asked_for(activation):
    torch.manual_seed(1)
    model = room.MLP([4, 3, 2], activation=activation)
    assert any(isinstance(m, activation) for m in model.modules()), f"MLP(..., activation={activation.__name__}) contains no {activation.__name__}."
    x = torch.randn(6, 4)
    expected = model.layers[1](activation()(model.layers[0](x)))
    assert torch.allclose(model(x), expected, atol=1e-6), (
        f"With {activation.__name__} between the two layers the output should be layers[1](act(layers[0](x))). It is not."
    )


# ------------------------------------------------------------------ counting
def test_count_parameters_counts_only_what_will_train():
    assert room.count_parameters(room.MLP([64, 32, 10])) == 2410, "64*32 + 32 + 32*10 + 10 = 2410."
    small = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    assert room.count_parameters(small) == 11, "3*2 + 2 + 2*1 + 1 = 11."
    small[0].weight.requires_grad_(False)
    small[0].bias.requires_grad_(False)
    got = room.count_parameters(small)
    assert got == 3, f"With the first layer frozen only 2 + 1 = 3 parameters train; you counted {got}. Filter on requires_grad."
    assert type(got) is int, "Return a Python int."


# ------------------------------------------------------------------ init
def test_init_weights_forges_kaiming_weights_and_zero_biases():
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Sequential(nn.Linear(500, 20)))
    before = model[0].weight.detach().clone()
    room.init_weights(model)
    for name, param in model.named_parameters():
        if name.endswith("bias"):
            assert bool((param == 0).all()), f"{name} is not all zeros. Biases start at zero; nn.init.zeros_."
    assert not torch.equal(model[0].weight.detach(), before), "The weights were not touched at all."
    ratio = model[0].weight.detach().std().item() * 1000**0.5
    assert 1.30 < ratio < 1.53, (
        f"std(weight) * sqrt(fan_in) is {ratio:.3f}. Kaiming for ReLU gives sqrt(2) = 1.414 "
        "(the default Linear init gives 0.577). nn.init.kaiming_normal_(w, nonlinearity='relu')."
    )
    assert abs(model[0].weight.detach().mean().item()) < 0.01, "Kaiming weights are centred on zero."
    nested = model[2][0]
    assert bool((nested.bias == 0).all()), "The nested Linear was missed. model.apply(fn) visits every submodule, however deep."


# ------------------------------------------------------------------ manual forward
def test_manual_forward_matches_the_module_with_the_forge_cold(monkeypatch):
    torch.manual_seed(3)
    model = room.MLP([64, 32, 16, 10], activation=nn.Tanh)
    x = torch.randn(7, 8, 8)
    expected = model(x)
    monkeypatch.setattr(nn.Linear, "forward", _cold("nn.Linear.forward"))
    monkeypatch.setattr(F, "linear", _cold("F.linear"))
    try:
        got = room.manual_forward(model, x)
    except ColdIron as exc:
        pytest.fail(str(exc))
    assert got.shape == expected.shape, f"manual_forward gives {tuple(got.shape)}, the module gives {tuple(expected.shape)}."
    assert torch.allclose(got, expected, atol=1e-5), (
        "manual_forward disagrees with model(x). Each layer is h @ weight.T + bias (weight is (out, in)), "
        "with the activation between layers and not after the last."
    )


# ------------------------------------------------------------------ bytes round trip
def test_the_weights_survive_a_trip_through_bytes():
    torch.manual_seed(4)
    original = room.MLP([64, 32, 10])
    blob = room.serialize(original)
    assert isinstance(blob, bytes), f"serialize must return bytes, not {type(blob).__name__}. buffer.getvalue()."
    decoded = torch.load(io.BytesIO(blob))
    assert isinstance(decoded, dict) and set(decoded) == set(original.state_dict()), (
        "The blob should decode to the state_dict (a dict of name -> tensor), not the whole module."
    )
    fresh = room.MLP([64, 32, 10])
    x = torch.rand(5, 8, 8)
    assert not torch.equal(original(x), fresh(x)), "Two freshly built MLPs should differ (random init)."
    returned = room.deserialize_into(fresh, blob)
    assert returned is fresh, "deserialize_into loads IN PLACE and returns the same module."
    assert torch.equal(original(x), fresh(x)), "After loading, the two models must agree exactly on every output."


def test_strict_loading_refuses_the_wrong_mould():
    blob = room.serialize(room.MLP([64, 32, 10]))
    with pytest.raises(RuntimeError):
        room.deserialize_into(room.MLP([64, 8, 10]), blob)


# ------------------------------------------------------------------ freeze
def test_freeze_stops_the_gradient_at_the_prefix():
    torch.manual_seed(5)
    model = room.MLP([64, 32, 10])
    frozen = room.freeze(model, "layers.0")
    assert frozen == 2, f"'layers.0' matches layers.0.weight and layers.0.bias: 2 tensors, you froze {frozen}."
    assert room.count_parameters(model) == 330, "With the first layer frozen 32*10 + 10 = 330 parameters remain trainable."
    loss = model(torch.rand(4, 8, 8)).sum()
    loss.backward()
    assert model.layers[0].weight.grad is None, "A frozen parameter must receive no gradient."
    assert model.layers[1].weight.grad is not None, "The unfrozen layer must still receive its gradient."
    assert room.freeze(model, "nothing_here") == 0, "A prefix that matches nothing freezes nothing."
