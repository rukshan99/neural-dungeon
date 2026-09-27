"""BOSS FIGHT - THE REPRODUCIBILITY REVENANT

Phase 1: every die is seeded by one call.
Phase 2: two runs from one seed come back bitwise identical.
Phase 3: a run is interrupted, everything is rebuilt from a checkpoint, and
         the resumed run matches an uninterrupted one to the bit.
"""

import random

import numpy as np
import pytest

from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

boss = load_room(__file__, "boss_reproducibility_revenant")

pytestmark = pytest.mark.boss

nn = torch.nn


def _first_difference(sd_a, sd_b):
    """Name and max abs difference of the first tensor that differs, or None if all equal."""
    if set(sd_a) != set(sd_b):
        return "keys", float("nan")
    for key in sd_a:
        if not torch.equal(sd_a[key], sd_b[key]):
            return key, (sd_a[key].float() - sd_b[key].float()).abs().max().item()
    return None


# ---------------------------------------------------------------------- phase 1
def _roll_the_dice():
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_phase_1_every_die_rolls_the_same_after_seeding():
    boss.seed_everything(7)
    first = _roll_the_dice()
    boss.seed_everything(7)
    second = _roll_the_dice()
    py, npy, th = (a == b for a, b in zip(first, second))
    assert py, "Python's random module was not seeded (random.seed)."
    assert npy, "numpy's global RNG was not seeded (np.random.seed)."
    assert th, "torch was not seeded (torch.manual_seed)."
    boss.seed_everything(8)
    assert _roll_the_dice() != second, "A different seed must roll differently."


def test_phase_1_torch_remembers_the_seed_it_was_given():
    boss.seed_everything(4242)
    assert torch.initial_seed() == 4242, f"torch.initial_seed() is {torch.initial_seed()}, expected 4242."


def test_phase_1_the_deterministic_flag_does_not_break_the_cpu():
    try:
        boss.seed_everything(1, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled(), "deterministic=True should switch torch.use_deterministic_algorithms on."
        model = nn.Sequential(nn.Flatten(), nn.Linear(64, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 10))
        loss = nn.functional.cross_entropy(model(torch.rand(8, 8, 8)), torch.randint(0, 10, (8,)))
        loss.backward()
        boss.seed_everything(1, deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled(), "deterministic=False must switch the flag back off; do not let it leak."
    finally:
        torch.use_deterministic_algorithms(False)


# ---------------------------------------------------------------------- phase 2
def test_phase_2_the_revenant_comes_back_identical():
    model_a, losses_a = boss.train_deterministically(11, epochs=2)
    model_b, losses_b = boss.train_deterministically(11, epochs=2)
    assert len(losses_a) == 2 and all(isinstance(v, float) for v in losses_a), f"Return one Python float per epoch; got {losses_a!r}."
    assert losses_a == losses_b, (
        f"Same seed, different losses: {losses_a} vs {losses_b}. Some die was not seeded: init, shuffle (give the loader a generator) or dropout."
    )
    diff = _first_difference(model_a.state_dict(), model_b.state_dict())
    assert diff is None, (
        f"Same seed, different weights: {diff[0]} differs by up to {diff[1]:.3e}. "
        "Bitwise identity is the bar. Seed before building the model, and give the DataLoader its own seeded generator."
    )


def test_phase_2_a_different_seed_is_a_different_revenant():
    model_a, losses_a = boss.train_deterministically(11, epochs=1)
    model_b, losses_b = boss.train_deterministically(12, epochs=1)
    assert losses_a != losses_b and _first_difference(model_a.state_dict(), model_b.state_dict()) is not None, (
        "Seeds 11 and 12 gave the same run. Is the seed reaching the model init and the loader?"
    )


def test_phase_2_it_learned_and_it_has_a_dropout_layer():
    model, losses = boss.train_deterministically(3, epochs=2)
    assert losses[1] < losses[0] < 2.4, f"The Revenant should learn: {losses}."
    assert losses[1] < 1.0, f"After two epochs the mean loss should be well under 1.0; got {losses[1]:.3f}."
    assert any(isinstance(m, nn.Dropout) for m in model.modules()), (
        "The model must contain nn.Dropout(0.1): the dropout mask is the die the Revenant hides behind."
    )


# ---------------------------------------------------------------------- phase 3
def test_phase_3_a_checkpoint_holds_every_piece_of_state(tmp_path):
    path = tmp_path / "phase3.pt"
    model, loader, optimizer = boss.build_run(3)
    boss.run_epoch(model, loader, optimizer)
    rng_state = {"torch": torch.get_rng_state(), "loader": loader.generator.get_state()}
    boss.save_checkpoint(path, model, optimizer, epoch=1, rng_state=rng_state)
    assert path.is_file(), "save_checkpoint wrote nothing at the given path."

    raw = torch.load(path, weights_only=True)
    assert {"model", "optimizer", "epoch", "rng_state"} <= set(raw), f"Checkpoint keys are {sorted(raw)}; expected model, optimizer, epoch, rng_state."
    checkpoint = boss.load_checkpoint(path)
    assert checkpoint["epoch"] == 1

    fresh_model, fresh_loader, fresh_optimizer = boss.build_run(99)
    fresh_model.load_state_dict(checkpoint["model"])
    fresh_optimizer.load_state_dict(checkpoint["optimizer"])
    x = torch.rand(4, 8, 8)
    fresh_model.eval()
    model.eval()
    assert torch.equal(fresh_model(x), model(x)), "The loaded model must reproduce the saved one exactly."
    saved_buffers = [s["momentum_buffer"] for s in optimizer.state_dict()["state"].values()]
    loaded_buffers = [s["momentum_buffer"] for s in fresh_optimizer.state_dict()["state"].values()]
    assert len(loaded_buffers) == len(saved_buffers) and all(torch.equal(a, b) for a, b in zip(saved_buffers, loaded_buffers)), (
        "The optimizer's momentum buffers did not survive. Save optimizer.state_dict() too."
    )
    assert torch.equal(checkpoint["rng_state"]["torch"], rng_state["torch"]), "The torch RNG state must come back exactly as saved."
    assert torch.equal(checkpoint["rng_state"]["loader"], rng_state["loader"]), "The loader generator's state must come back exactly as saved."


def test_phase_3_the_interrupted_run_matches_the_straight_run_to_the_bit(tmp_path):
    path = tmp_path / "revenant.pt"
    straight, straight_losses = boss.train_deterministically(5, epochs=3)
    resumed, resumed_losses = boss.train_with_resume(5, epochs=3, stop_after=2, path=path)
    assert path.is_file(), "train_with_resume must write its checkpoint at `path`."
    assert len(resumed_losses) == 3, f"Three epochs were requested; {len(resumed_losses)} losses came back."
    assert resumed_losses[:2] == straight_losses[:2], (
        f"The first two epochs already differ: {resumed_losses[:2]} vs {straight_losses[:2]}. Phase 2 is not solid yet."
    )
    assert resumed_losses[2] == straight_losses[2], (
        f"The resumed third epoch drifted: {resumed_losses[2]!r} vs {straight_losses[2]!r}. After rebuilding, restore ALL of: "
        "model state, optimizer state (momentum), torch.set_rng_state (dropout masks) and loader.generator.set_state (shuffle order)."
    )
    diff = _first_difference(straight.state_dict(), resumed.state_dict())
    assert diff is None, (
        f"The final weights differ at {diff[0]} by up to {diff[1]:.3e}. A resumed run must be indistinguishable from an uninterrupted one."
    )
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["epoch"] == 2, f"The checkpoint should record epoch=2 (stop_after); it says {checkpoint.get('epoch')!r}."
    assert checkpoint.get("rng_state"), "The checkpoint carries no rng_state. Without it the Revenant returns."
