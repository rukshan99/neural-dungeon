"""TRIAL 4.6 - THE HALF-LIGHT

Three floating-point formats are read off torch.finfo and checked against
each other, the Half-Light Prophecy is run, a sum is burned to infinity and
rescued, autocast is watched from inside a model, a model is halved for
inference, and a hand-written loss scaler is driven with injected infinities.
"""

import pytest

from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

room = load_room(__file__, "room_6_the_half_light")

nn = torch.nn
F = torch.nn.functional

FORMATS = {torch.float32: (8, 23), torch.float16: (5, 10), torch.bfloat16: (8, 7)}
REPORT_KEYS = {"bits", "exponent_bits", "mantissa_bits", "max", "eps", "tiny"}


def _mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))


def _relative_error(got, expected):
    return ((got.float() - expected.float()).norm() / expected.float().norm()).item()


# ------------------------------------------------------------------ dtype_report
@pytest.mark.parametrize("dtype", list(FORMATS))
def test_the_three_formats_are_read_off_finfo(dtype):
    report = room.dtype_report(dtype)
    assert isinstance(report, dict) and set(report) == REPORT_KEYS, (
        f"dtype_report({dtype}) must return a dict with exactly the keys {sorted(REPORT_KEYS)}; got {sorted(report) if isinstance(report, dict) else type(report).__name__}."
    )
    info = torch.finfo(dtype)
    exponent_bits, mantissa_bits = FORMATS[dtype]
    assert report["bits"] == info.bits, f"{dtype} is {info.bits} bits wide, not {report['bits']}."
    assert report["exponent_bits"] == exponent_bits, f"{dtype} has {exponent_bits} exponent bits, not {report['exponent_bits']}."
    assert report["mantissa_bits"] == mantissa_bits, f"{dtype} has {mantissa_bits} mantissa bits, not {report['mantissa_bits']}."
    for key in ("max", "eps", "tiny"):
        assert report[key] == getattr(info, key), f"{dtype}: {key} should be torch.finfo({dtype}).{key} = {getattr(info, key)!r}; you reported {report[key]!r}."
    for key in ("bits", "exponent_bits", "mantissa_bits"):
        assert type(report[key]) is int, f"{key} should be a Python int, not {type(report[key]).__name__}."
    for key in ("max", "eps", "tiny"):
        assert type(report[key]) is float, f"{key} should be a Python float, not {type(report[key]).__name__}."


@pytest.mark.parametrize("dtype", list(FORMATS))
def test_the_columns_of_the_table_agree_with_each_other(dtype):
    r = room.dtype_report(dtype)
    e, m = r["exponent_bits"], r["mantissa_bits"]
    assert r["bits"] == 1 + e + m, f"{dtype}: 1 sign + {e} exponent + {m} mantissa bits should add up to {r['bits']}."
    assert r["eps"] == 2.0**-m, (
        f"{dtype}: eps is the value of the last mantissa bit, 2**-{m} = {2.0**-m}; finfo says {r['eps']}. "
        "If these disagree, the mantissa width in your table is wrong."
    )
    assert r["tiny"] == 2.0 ** (2 - 2 ** (e - 1)), (
        f"{dtype}: the smallest normal number is 2**(2 - 2**({e} - 1)) = {2.0 ** (2 - 2 ** (e - 1))}; finfo says {r['tiny']}. "
        "If these disagree, the exponent width in your table is wrong."
    )
    assert r["max"] == (2 - r["eps"]) * 2.0 ** (2 ** (e - 1) - 1), (
        f"{dtype}: the largest finite value is (2 - eps) * 2**(2**({e} - 1) - 1); finfo says {r['max']}."
    )


def test_an_integer_is_not_a_floating_point_format():
    with pytest.raises(ValueError):
        room.dtype_report(torch.int64)


# ------------------------------------------------------------------ the Prophecy
SNIPPETS = {
    "70000_in_float16": """
x = torch.tensor(70000.0).to(torch.float16)
result = "inf" if torch.isinf(x) else "finite"
""",
    "70000_in_bfloat16": """
x = torch.tensor(70000.0).to(torch.bfloat16)
result = "inf" if torch.isinf(x) else "finite"
""",
    "1_plus_1e-3_equals_1_in_float16": """
one = torch.tensor(1.0, dtype=torch.float16)
small = torch.tensor(1e-3, dtype=torch.float16)
result = bool((one + small) == one)
""",
    "1_plus_1e-3_equals_1_in_bfloat16": """
one = torch.tensor(1.0, dtype=torch.bfloat16)
small = torch.tensor(1e-3, dtype=torch.bfloat16)
result = bool((one + small) == one)
""",
    "1e-4_squared_in_float16": """
x = torch.tensor(1e-4, dtype=torch.float16)
result = "zero" if (x * x) == 0 else "nonzero"
""",
    "1e-4_squared_in_bfloat16": """
x = torch.tensor(1e-4, dtype=torch.bfloat16)
result = "zero" if (x * x) == 0 else "nonzero"
""",
    "a_gradient_of_1e-8_in_float16": """
g = torch.tensor(1e-8).to(torch.float16)
result = "zero" if g == 0 else "nonzero"
""",
    "the_same_gradient_scaled_by_2_to_16": """
g = (torch.tensor(1e-8) * 2.0 ** 16).to(torch.float16)
result = "zero" if g == 0 else "nonzero"
""",
}


def _run(snippet):
    namespace = {"torch": torch}
    exec(snippet, namespace)  # noqa: S102 - the snippets are ours
    return namespace["result"]


def test_the_prophecy_snippets_are_untouched():
    assert dict(room.HALF_LIGHT_SNIPPETS) == SNIPPETS, "Do not edit the snippets; predict them."
    assert set(room.HALF_LIGHT_PROPHECY) == set(SNIPPETS), "Do not add or remove prophecies; fill them in."


@pytest.mark.parametrize("name", list(SNIPPETS))
def test_the_half_light_prophecy_holds(name):
    prediction = room.HALF_LIGHT_PROPHECY.get(name)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied {name!r}. Fill in HALF_LIGHT_PROPHECY.")
    actual = _run(SNIPPETS[name])
    assert type(prediction) is type(actual), (
        f"{name}: the snippet produces a {type(actual).__name__} ({actual!r}); you predicted a {type(prediction).__name__} ({prediction!r})."
    )
    assert prediction == actual, f"{name}: the prophecy said {prediction!r}, the machine produced {actual!r}."


# ------------------------------------------------------------------ overflow
def test_the_naive_sum_of_squares_burns_out_to_infinity():
    thousand_tens = torch.full((1000,), 10.0, dtype=torch.float16)
    got = room.overflow_demo(thousand_tens)
    assert torch.is_tensor(got) and got.dtype == torch.float16 and got.ndim == 0, (
        f"overflow_demo returns a 0-d float16 tensor, computed naively; got {type(got).__name__} {getattr(got, 'dtype', '')} {tuple(getattr(got, 'shape', ()))}. Do not upcast: this one is meant to break."
    )
    assert torch.isinf(got), (
        f"1000 squares of 10.0 are each a harmless 100, but their sum, 100000, is past 65504 and should come back inf in float16. You got {got.item()}: something was upcast."
    )
    assert torch.isinf(room.overflow_demo(torch.tensor([300.0, 1.0], dtype=torch.float16))), (
        "300.0 ** 2 = 90000 is past 65504 on its own: the naive float16 sum should be inf."
    )


def test_the_safe_sum_survives_in_float32():
    thousand_tens = torch.full((1000,), 10.0, dtype=torch.float16)
    got = room.safe_sum_of_squares(thousand_tens)
    assert torch.is_tensor(got) and got.dtype == torch.float32 and got.ndim == 0, (
        f"safe_sum_of_squares returns a 0-d float32 tensor; got {type(got).__name__} {getattr(got, 'dtype', '')} {tuple(getattr(got, 'shape', ()))}."
    )
    assert torch.isfinite(got) and got.item() == pytest.approx(100000.0, rel=1e-3), (
        f"1000 squares of 10.0 sum to 100000; accumulated in float32 that is finite. You got {got.item()}."
    )
    got = room.safe_sum_of_squares(torch.tensor([300.0, 1.0], dtype=torch.float16))
    assert torch.isfinite(got) and got.item() == pytest.approx(90001.0, rel=1e-3), (
        f"300**2 + 1**2 = 90001, but you got {got.item()}. The cast must happen BEFORE the square: "
        "sum(dtype=torch.float32) still squares in float16, where 300**2 is already inf."
    )
    benign = torch.tensor([0.5, -1.5, 2.0, 3.0], dtype=torch.float16)
    naive, safe = room.overflow_demo(benign), room.safe_sum_of_squares(benign)
    assert safe.item() == pytest.approx(15.5, rel=1e-3) and naive.item() == pytest.approx(15.5, rel=1e-2), (
        "On a vector that does not overflow the two sums agree: 0.25 + 2.25 + 4 + 9 = 15.5."
    )


# ------------------------------------------------------------------ autocast
class AutocastSpy(nn.Module):
    """Writes down whether autocast was on, and in which dtype, when it was called."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(64, 10)
        self.seen_enabled = []
        self.seen_dtype = []

    def forward(self, x):
        self.seen_enabled.append(torch.is_autocast_enabled("cpu"))
        self.seen_dtype.append(torch.get_autocast_dtype("cpu"))
        return self.lin(x.flatten(1))


class WithLoss(nn.Module):
    """Returns (logits, loss) so the trial can see what dtype the loss comes out in."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        self.targets = torch.arange(16) % 10

    def forward(self, x):
        logits = self.body(x)
        return logits, F.cross_entropy(logits, self.targets)


def test_autocast_runs_the_matmuls_in_bfloat16_and_leaves_the_weights_alone():
    model = _mlp(0)
    x = torch.rand(16, 8, 8)
    expected = model(x)
    got = room.forward_autocast_bf16(model, x)
    assert torch.is_tensor(got) and got.shape == expected.shape, f"forward_autocast_bf16 returns model(x): a {tuple(expected.shape)} tensor."
    assert got.dtype == torch.bfloat16, (
        f"Under autocast(dtype=torch.bfloat16) nn.Linear produces bfloat16 logits; yours are {got.dtype}. Is the forward actually inside the with-block?"
    )
    assert all(p.dtype == torch.float32 for p in model.parameters()), (
        "The parameters are no longer float32. autocast casts copies on the way into each op; it never touches the model. Do not call model.to(bfloat16) here."
    )
    err = _relative_error(got, expected)
    assert err < 2e-2, f"The bfloat16 forward differs from the float32 forward by {err:.3%} (relative); 8 significant bits should land within about 1%."


def test_autocast_is_on_inside_the_block_and_off_afterwards():
    spy = AutocastSpy()
    room.forward_autocast_bf16(spy, torch.rand(4, 8, 8))
    assert spy.seen_enabled == [True], "The model was called with autocast switched off. Wrap the call in torch.autocast(device_type='cpu', dtype=torch.bfloat16)."
    assert spy.seen_dtype == [torch.bfloat16], f"autocast was on, but in {spy.seen_dtype[0]}. This room asks for torch.bfloat16."
    assert not torch.is_autocast_enabled("cpu"), "autocast is still on after the call returned. Use the context manager so it restores the previous state."


def test_the_loss_comes_out_in_float32_under_autocast():
    torch.manual_seed(1)
    model = WithLoss()
    x = torch.rand(16, 8, 8)
    logits32, loss32 = model(x)
    logits, loss = room.forward_autocast_bf16(model, x)
    assert logits.dtype == torch.bfloat16, f"The logits should be bfloat16 (they come out of a Linear); got {logits.dtype}."
    assert loss.dtype == torch.float32, (
        f"F.cross_entropy is on autocast's float32 list: the loss should come out float32 even from bfloat16 logits; got {loss.dtype}."
    )
    assert loss.item() == pytest.approx(loss32.item(), rel=2e-2), f"Loss under autocast {loss.item():.4f} vs float32 {loss32.item():.4f}: too far apart."


# ------------------------------------------------------------------ cast_for_inference
def test_the_inference_copy_weighs_half_and_answers_nearly_the_same():
    model = _mlp(2)
    x = torch.rand(16, 8, 8)
    expected = model(x)
    full_bytes = room.parameter_bytes(model)
    assert full_bytes == 2410 * 4, f"2410 float32 parameters occupy 9640 bytes; parameter_bytes says {full_bytes}."
    half = room.cast_for_inference(model, torch.bfloat16)
    assert isinstance(half, nn.Module) and half is not model, "cast_for_inference returns a NEW module (copy.deepcopy), not the original."
    assert all(p.dtype == torch.bfloat16 for p in half.parameters()), "Every parameter of the copy should be bfloat16: module.to(dtype)."
    assert room.parameter_bytes(half) == full_bytes // 2, f"The bfloat16 copy should weigh {full_bytes // 2} bytes, half of {full_bytes}; parameter_bytes says {room.parameter_bytes(half)}."
    assert all(p.dtype == torch.float32 for p in model.parameters()) and model.training, (
        "The original was modified. deepcopy first; the caller's float32 model is the master and stays as it was."
    )
    assert not half.training, "The copy is for inference: put it in eval mode."
    for (name, a), b in zip(half.named_parameters(), model.parameters()):
        assert a.data_ptr() != b.data_ptr(), f"{name} of the copy shares storage with the original."
    got = half(x.to(torch.bfloat16))
    assert got.dtype == torch.bfloat16 and got.shape == expected.shape
    err = _relative_error(got, expected)
    assert err < 2e-2, f"The bfloat16 copy differs from the float32 model by {err:.3%} (relative); expected about 1% or less."


def test_parameter_bytes_counts_element_sizes_not_assumptions():
    mixed = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1).to(torch.bfloat16))
    got = room.parameter_bytes(mixed)
    assert type(got) is int, f"Return a Python int, not {type(got).__name__}."
    assert got == 8 * 4 + 3 * 2, f"8 float32 parameters (32 bytes) plus 3 bfloat16 parameters (6 bytes) = 38; you counted {got}. Use p.element_size(), not 4."


# ------------------------------------------------------------------ LossScaler
def _param(*values):
    return nn.Parameter(torch.tensor(list(values)))


def test_scale_multiplies_the_loss_and_every_gradient_with_it():
    scaler = room.LossScaler(init_scale=2.0**16)
    assert scaler.get_scale() == 2.0**16, f"A fresh scaler reports init_scale; yours says {scaler.get_scale()}."
    p = _param(1.0, 2.0)
    loss = (p**2).sum()
    scaled = scaler.scale(loss)
    assert torch.is_tensor(scaled) and scaled.requires_grad, "scale(loss) returns loss * S as a tensor still in the graph."
    assert scaled.item() == pytest.approx(5.0 * 2.0**16), f"scale(5.0) with S = 2**16 is {5.0 * 2.0**16}; got {scaled.item()}."
    scaled.backward()
    assert torch.equal(p.grad, torch.tensor([2.0, 4.0]) * 2.0**16), (
        f"After scaled.backward() every gradient carries the factor S: expected {(torch.tensor([2.0, 4.0]) * 2.0**16).tolist()}, got {p.grad.tolist()}."
    )


def test_unscale_divides_in_place_and_reports_infinities():
    scaler = room.LossScaler(init_scale=8.0)
    g = torch.tensor([8.0, 16.0])
    found = scaler.unscale_([g, None])
    assert found is False, f"unscale_ returns False when every gradient is finite; got {found!r}."
    assert torch.equal(g, torch.tensor([1.0, 2.0])), f"unscale_ divides IN PLACE by S = 8: the tensor you passed should now read [1.0, 2.0], not {g.tolist()}."
    assert scaler.unscale_([torch.tensor([1.0, float("inf")])]) is True, "An inf anywhere means found_inf = True."
    assert scaler.unscale_([torch.tensor([1.0]), torch.tensor([float("nan")])]) is True, "A nan counts too: torch.isfinite is False for both."


def test_step_takes_the_step_when_the_gradients_are_finite():
    p = _param(1.0, 2.0)
    optimizer = torch.optim.SGD([p], lr=0.1)
    scaler = room.LossScaler(init_scale=16.0, growth_interval=3)
    p.grad = torch.tensor([1.0, 1.0]) * 16.0  # what a scaled backward would leave behind
    took = scaler.step(optimizer, [p])
    assert took is True, "With finite gradients step() takes the step and returns True."
    assert torch.allclose(p.detach(), torch.tensor([0.9, 1.9])), (
        f"After unscaling, grad = [1, 1] and SGD(lr=0.1) should give [0.9, 1.9]; got {p.tolist()}. "
        "If you see [-0.6, 0.4] the gradients were never divided by S before optimizer.step()."
    )
    assert scaler.get_scale() == 16.0, "One clean step out of a growth_interval of 3 does not grow the scale yet."


def test_step_skips_and_backs_off_on_an_infinite_gradient():
    p = _param(1.0, 2.0)
    optimizer = torch.optim.SGD([p], lr=0.1)
    scaler = room.LossScaler(init_scale=16.0, backoff_factor=0.5, growth_interval=3)
    p.grad = torch.tensor([1.0, float("inf")])
    took = scaler.step(optimizer, [p])
    assert took is False, "An inf gradient means the step is skipped: return False."
    assert torch.equal(p.detach(), torch.tensor([1.0, 2.0])), f"The parameters moved to {p.tolist()}. A skipped step leaves them exactly as they were."
    assert scaler.get_scale() == 8.0, f"After an overflow the scale backs off by backoff_factor: 16 * 0.5 = 8, not {scaler.get_scale()}."


def test_the_scale_grows_only_after_consecutive_clean_steps():
    p = _param(0.0, 0.0)
    optimizer = torch.optim.SGD([p], lr=0.01)
    scaler = room.LossScaler(init_scale=16.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=3)
    schedule = []
    for kind in ("good", "good", "inf", "good", "good", "good"):
        value = float("inf") if kind == "inf" else 1.0
        p.grad = torch.tensor([value, 1.0]) * scaler.get_scale()
        scaler.step(optimizer, [p])
        schedule.append(scaler.get_scale())
    assert schedule == [16.0, 16.0, 8.0, 8.0, 8.0, 16.0], (
        f"Scale after each of good, good, inf, good, good, good should read [16, 16, 8, 8, 8, 16]; yours reads {schedule}. "
        "The overflow halves the scale AND resets the count of consecutive clean steps; only 3 clean steps in a row double it."
    )


def test_the_small_gradient_survives_the_half_light_only_when_scaled():
    def half_light_loss(w):
        h = w.half()                    # the float16 copy autocast would make
        y = (h * h).sum().float()       # float16 arithmetic, then back to float32 for the loss
        return y * 1e-8                 # a tiny loss, like a network that is nearly trained

    w = torch.tensor([1.0, -2.0], requires_grad=True)
    half_light_loss(w).backward()
    assert torch.equal(w.grad, torch.zeros(2)), "Sanity: without scaling the 1e-8 gradient underflows to zero on its way through float16."

    w = torch.tensor([1.0, -2.0], requires_grad=True)
    scaler = room.LossScaler(init_scale=2.0**16)
    scaler.scale(half_light_loss(w)).backward()
    assert bool((w.grad != 0).all()), "Scaled by 2**16 the gradient is 6.6e-4 in float16, a normal number: it should survive."
    found = scaler.unscale_([w.grad])
    assert found is False
    expected = 2 * torch.tensor([1.0, -2.0]) * 1e-8
    assert torch.allclose(w.grad, expected, rtol=1e-2, atol=0.0), (
        f"After unscaling, the gradient should be d/dw of 1e-8 * w**2 = {expected.tolist()}; got {w.grad.tolist()}."
    )
