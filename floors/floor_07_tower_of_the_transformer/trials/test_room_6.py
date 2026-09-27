"""TRIAL 7.6 - THE FOUNDRY LINE

Shards must be equal cuts of the batch; the mean of the per-shard gradients
must equal the full-batch gradient exactly when the shards are equal, and
only when weighted otherwise; then two real processes over gloo, by hand
and through DDP, must agree with one process to 1e-5.
"""

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

import torch.distributed as dist  # noqa: E402

from dungeon import scrutiny  # noqa: E402
from dungeon.artifacts import tiny_gpt  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

room = load_room(__file__, "room_6_the_foundry_line")

torch.manual_seed(76)

CFG = tiny_gpt.GPTConfig(vocab_size=17, block_size=16, n_layer=1, n_head=2, n_embd=32)
VOCAB, T = CFG.vocab_size, CFG.block_size

needs_distributed = pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed is not available in this build of PyTorch, so the real-process trials cannot run",
)


def _make_model(seed=0):
    torch.manual_seed(seed)
    return tiny_gpt.GPT(CFG)


def _batch(batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, VOCAB, (batch_size, T), generator=g)
    y = torch.randint(0, VOCAB, (batch_size, T), generator=g)
    return x, y


def _full_batch_grads(make_model, x, y):
    """The one-process answer every parallel scheme must reproduce."""
    model = make_model()
    model.train()
    _, loss = model(x, y)
    loss.backward()
    return {name: p.grad.detach().clone() for name, p in model.named_parameters()}


def _grads_of_slice(make_model, x, y):
    return _full_batch_grads(make_model, x, y)


def _assert_same_gradients(got, want, tol, where):
    assert isinstance(got, dict), f"{where} should return a dict of {{parameter name: gradient}}, got {type(got).__name__}."
    assert set(got) == set(want), (
        f"{where} returned gradients for {sorted(got)}, but the model's parameters are {sorted(want)}. "
        "Key the dict by model.named_parameters() names (the tied lm_head.weight appears once, as wte.weight)."
    )
    for name, want_g in want.items():
        got_g = got[name]
        assert got_g.shape == want_g.shape, f"Gradient of {name} has shape {tuple(got_g.shape)}, expected {tuple(want_g.shape)}."
        diff = (got_g - want_g).abs().max().item()
        scale = want_g.abs().max().item() + 1e-12
        ratio = got_g.abs().max().item() / scale
        assert diff < tol, (
            f"{where}: the gradient of {name} differs from the full-batch gradient by {diff:.2e} (tolerance {tol:.0e}). "
            + (f"It is about {ratio:.1f}x too large: you summed across ranks but never divided by world_size."
               if ratio > 1.5 else
               f"It is about {1 / max(ratio, 1e-12):.1f}x too small: did every rank contribute, and was the loss a mean?"
               if ratio < 0.6 else
               "Same size, different direction: a shard was wrong, a rank reused stale gradients, or the ranks did not start from the same weights.")
        )


# ------------------------------------------------------------------ shard_batch
def test_shard_batch_cuts_equal_contiguous_slices():
    x = torch.arange(8).view(8, 1).expand(8, 3).clone()
    y = -x
    for rank in range(4):
        xs, ys = room.shard_batch(x, y, 4, rank)
        assert xs.shape == (2, 3) and ys.shape == (2, 3), f"8 rows over 4 ranks is 2 rows each; rank {rank} got {tuple(xs.shape)}."
        assert xs[:, 0].tolist() == [2 * rank, 2 * rank + 1], (
            f"Rank {rank} should hold rows {[2 * rank, 2 * rank + 1]} of the batch, got rows {xs[:, 0].tolist()}. "
            "Shard r is the contiguous slice [r * n, (r + 1) * n) along dim 0."
        )
        assert torch.equal(ys, -xs), "x and y must be cut with the SAME slice, or targets stop matching their inputs."
    rebuilt = torch.cat([room.shard_batch(x, y, 4, r)[0] for r in range(4)])
    assert torch.equal(rebuilt, x), "Concatenating the four shards in rank order must give the original batch back."


def test_shard_batch_refuses_an_uneven_cut():
    x, y = _batch(10)
    with pytest.raises(ValueError):
        room.shard_batch(x, y, 4, 0)
    x, y = _batch(8)
    with pytest.raises(ValueError):
        room.shard_batch(x, y, 3, 1)


# ------------------------------------------------------------ averaging by hand
def test_average_gradients_is_the_elementwise_mean():
    a = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([[10.0]])}
    b = {"w": torch.tensor([3.0, 6.0]), "b": torch.tensor([[-10.0]])}
    c = {"w": torch.tensor([5.0, 1.0]), "b": torch.tensor([[3.0]])}
    got = room.average_gradients([a, b, c])
    assert set(got) == {"w", "b"}, f"The averaged dict keeps every parameter name; got {sorted(got)}."
    assert torch.allclose(got["w"], torch.tensor([3.0, 3.0])), f"mean([1,2],[3,6],[5,1]) = [3, 3]; got {got['w'].tolist()}."
    assert torch.allclose(got["b"], torch.tensor([[1.0]])), f"mean(10, -10, 3) = 1; got {got['b'].tolist()}."
    assert a["w"].tolist() == [1.0, 2.0], "average_gradients modified its input. Build new tensors."
    with pytest.raises(ValueError):
        room.average_gradients([])


def test_weighted_average_weights_each_rank_by_its_shard_size():
    small = {"w": torch.tensor([0.0, 8.0])}
    big = {"w": torch.tensor([4.0, 0.0])}
    got = room.weighted_average_gradients([small, big], sizes=[2, 6])
    assert torch.allclose(got["w"], torch.tensor([3.0, 2.0])), (
        f"(2 * [0, 8] + 6 * [4, 0]) / 8 = [3, 2]; got {got['w'].tolist()}."
    )
    equal = room.weighted_average_gradients([small, big], sizes=[5, 5])
    assert torch.allclose(equal["w"], room.average_gradients([small, big])["w"]), "With equal sizes the weighted mean IS the plain mean."
    with pytest.raises(ValueError):
        room.weighted_average_gradients([small, big], sizes=[1])


# ------------------------------------------------------------------ the theorem
@pytest.mark.parametrize("world_size", [2, 4])
def test_the_mean_of_the_shard_gradients_is_the_full_batch_gradient(world_size):
    x, y = _batch(8)
    want = _full_batch_grads(_make_model, x, y)
    got = room.simulated_data_parallel_step(_make_model, x, y, world_size)
    _assert_same_gradients(got, want, 1e-6, f"simulated_data_parallel_step with world_size={world_size}")


def test_the_simulation_builds_one_model_and_starts_every_rank_clean():
    x, y = _batch(8)
    builds = []

    def counting_make_model():
        builds.append(1)
        return _make_model()

    got = room.simulated_data_parallel_step(counting_make_model, x, y, 4)
    assert len(builds) == 1, (
        f"make_model was called {len(builds)} times. Build ONE model: the ranks share the same weights, and a "
        "differently seeded copy per rank would differentiate a different function."
    )
    # If gradients were not zeroed between ranks, rank r would carry the sum of ranks 0..r.
    want = _full_batch_grads(_make_model, x, y)
    for name in want:
        ratio = got[name].abs().max().item() / (want[name].abs().max().item() + 1e-12)
        assert ratio < 1.5, (
            f"The gradient of {name} is {ratio:.1f}x the full-batch gradient. Gradients accumulated from one rank "
            "into the next: zero_grad(set_to_none=True) before every rank's backward."
        )


def test_unequal_shards_break_the_plain_mean_and_the_weighted_mean_repairs_it():
    x, y = _batch(8)
    want = _full_batch_grads(_make_model, x, y)
    per_shard = [_grads_of_slice(_make_model, x[:2], y[:2]), _grads_of_slice(_make_model, x[2:], y[2:])]
    plain = room.average_gradients(per_shard)
    worst_plain = max((plain[n] - want[n]).abs().max().item() / (want[n].abs().max().item() + 1e-12) for n in want)
    assert worst_plain > 1e-3, (
        "Dungeon self-check: with shards of 2 and 6 the plain mean of shard gradients should NOT be the full-batch "
        f"gradient, yet the worst relative error is {worst_plain:.2e}."
    )
    weighted = room.weighted_average_gradients(per_shard, sizes=[2, 6])
    _assert_same_gradients(weighted, want, 1e-6, "weighted_average_gradients with sizes [2, 6]")


# -------------------------------------------------------------- real processes
def _guard_stub(fn, name):
    if scrutiny.is_stub(fn):
        raise NotImplementedError(f"{name}() is unwritten")


@needs_distributed
def test_two_real_processes_all_reduce_to_the_full_batch_gradient(tmp_path):
    _guard_stub(room.worker, "worker")
    x, y = _batch(8, seed=1)
    out_dir = tmp_path / "by_hand"
    got = room.launch_data_parallel(2, x, y, out_dir)
    assert (out_dir / room.REDUCED_GRADS_FILE).is_file(), (
        f"Rank 0 should torch.save its reduced gradients to {out_dir / room.REDUCED_GRADS_FILE}; nothing is there."
    )
    want = _full_batch_grads(room.build_foundry_model, x, y)
    _assert_same_gradients(got, want, 1e-5, "two gloo processes with all_reduce(SUM) / world_size")


@needs_distributed
def test_distributed_data_parallel_averages_for_you(tmp_path):
    _guard_stub(room.ddp_worker, "ddp_worker")
    x, y = _batch(8, seed=2)
    out_dir = tmp_path / "ddp"
    got = room.launch_data_parallel(2, x, y, out_dir, use_ddp=True)
    want = _full_batch_grads(room.build_foundry_model, x, y)
    assert not any(k.startswith("module.") for k in got), (
        "Save the bare model's parameter names (ddp.module.named_parameters()), not DDP's 'module.'-prefixed ones."
    )
    _assert_same_gradients(got, want, 1e-5, "DistributedDataParallel over two gloo processes")
