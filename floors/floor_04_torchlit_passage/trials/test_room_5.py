"""TRIAL 4.5 - THE DEVICE FERRY

Everything runs on the CPU. Where a second bank is needed the trial uses the
`meta` device, which exists on every build and holds shapes but no data.
"""

import numpy as np
import pytest

from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

room = load_room(__file__, "room_5_device_ferry")

nn = torch.nn
META = torch.device("meta")


# ------------------------------------------------------------------ pick_device
def test_the_ferryman_honours_an_override(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    got = room.pick_device("cpu")
    assert isinstance(got, torch.device), f"Return a torch.device, not {type(got).__name__}."
    assert got == torch.device("cpu"), "prefer='cpu' must win even when cuda claims to be available."
    assert room.pick_device("cuda:1") == torch.device("cuda:1"), "An override is trusted as given, index included."


def test_the_ferryman_prefers_the_bright_bank(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert room.pick_device().type == "cuda", "With cuda available and no override, pick cuda."


def test_then_the_orchard_bank(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert room.pick_device().type == "mps", "Without cuda but with mps available, pick mps."


def test_then_home(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert room.pick_device() == torch.device("cpu"), "With nothing else available, cpu."


# ------------------------------------------------------------------ to_device
def test_nested_cargo_reaches_the_far_bank():
    marker = object()
    cargo = {
        "x": torch.zeros(2, 3),
        "pair": (torch.ones(4), torch.arange(3)),
        "list": [torch.zeros(1), {"deep": torch.zeros(2, 2, dtype=torch.float64)}],
        "name": "rune",
        "n": 7,
        "nothing": None,
        "thing": marker,
    }
    out = room.to_device(cargo, META)
    assert isinstance(out, dict) and set(out) == set(cargo), "A dict must come back as a dict with the same keys."
    assert out["x"].device == META and tuple(out["x"].shape) == (2, 3), f"cargo['x'] ended up on {out['x'].device}."
    assert isinstance(out["pair"], tuple) and all(t.device == META for t in out["pair"]), "A tuple must stay a tuple, with every tensor moved."
    assert out["pair"][1].dtype == torch.int64, "Moving a tensor must not change its dtype."
    assert isinstance(out["list"], list) and out["list"][0].device == META, "A list must stay a list, with every tensor moved."
    assert out["list"][1]["deep"].device == META and out["list"][1]["deep"].dtype == torch.float64, "Nesting goes all the way down; dtype is preserved."
    assert out["name"] == "rune" and out["n"] == 7 and out["nothing"] is None and out["thing"] is marker, (
        "Non-tensor cargo must pass through untouched: the very same objects."
    )
    assert cargo["x"].device.type == "cpu", "to_device returns new tensors; it must not mutate the input structure."


def test_a_bare_tensor_and_a_string_device_are_fine_too():
    t = torch.zeros(3)
    out = room.to_device(t, "meta")
    assert torch.is_tensor(out) and out.device.type == "meta", "A bare tensor is cargo too, and 'meta' as a string must work."
    same = room.to_device(t, "cpu")
    assert torch.is_tensor(same) and same.device.type == "cpu"


# ------------------------------------------------------------------ ensure_float32
def test_the_float64_that_numpy_smuggled_in_is_cured():
    rng = np.random.default_rng(0)
    x64 = torch.from_numpy(rng.random((4, 8, 8)))  # numpy's default: float64
    labels = torch.tensor([1, 2, 3, 4])
    assert x64.dtype == torch.float64
    x, y = room.ensure_float32((x64, labels))
    assert x.dtype == torch.float32, f"Images should be float32 after the cure, got {x.dtype}."
    assert y.dtype == torch.int64 and torch.equal(y, labels), "Integer labels must stay int64. Only floating tensors change."
    out = nn.Linear(64, 10)(x.flatten(1))
    assert out.shape == (4, 10), "The cured batch must go through a float32 nn.Linear without complaint."
    with pytest.raises(RuntimeError):
        nn.Linear(64, 10)(x64.flatten(1))  # the uncured one does not


def test_numpy_arrays_become_tensors_and_containers_survive():
    rng = np.random.default_rng(1)
    batch = {"img": rng.random((2, 8, 8)), "label": np.array([3, 4]), "flag": True, "pair": [np.float64(0.5), torch.zeros(2, dtype=torch.float16)]}
    out = room.ensure_float32(batch)
    assert torch.is_tensor(out["img"]) and out["img"].dtype == torch.float32, "A float64 numpy array becomes a float32 tensor."
    assert torch.is_tensor(out["label"]) and out["label"].dtype == torch.int64, "An int64 numpy array becomes an int64 tensor."
    assert out["flag"] is True, "Non-tensor, non-array values pass through."
    assert isinstance(out["pair"], list) and out["pair"][1].dtype == torch.float32, "float16 is floating too: it becomes float32. Lists stay lists."


# ------------------------------------------------------------------ model_device
def test_a_model_knows_where_it_lives():
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 10))
    got = room.model_device(model)
    assert isinstance(got, torch.device) and got.type == "cpu", f"A fresh model lives on the cpu; you said {got!r}."
    model.to(META)
    assert room.model_device(model).type == "meta", "After model.to(device) the model reports the new device. Read it from the parameters."


def test_a_model_with_only_buffers_still_has_an_address_and_an_empty_one_has_none():
    class OnlyBuffers(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running", torch.zeros(3))

    assert room.model_device(OnlyBuffers()).type == "cpu", "No parameters, but a buffer: the buffer's device is the answer."
    with pytest.raises(ValueError):
        room.model_device(nn.Identity())
