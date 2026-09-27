"""ROOM 7.6 - THE FOUNDRY LINE

    One mason with one chisel built the tower. This is how it would be built
    in a hurry: the same blueprint pinned up in every workshop, a different
    heap of stone at each bench, and a runner who averages every mason's
    notes before anyone is allowed to strike again. The blueprints never
    drift apart, because every workshop applies the same averaged note.

DATA PARALLELISM. Every worker ("rank", numbered 0 .. world_size-1) holds an
identical copy of the model. The batch is cut into world_size equal shards
along the batch axis; each rank computes the gradient of the MEAN loss on
its own shard; the ranks average their gradients; every rank applies the
same averaged gradient with the same optimizer state. The copies stay
identical for the whole run. Nothing about the model changes; only where
the examples go.

THE THEOREM that makes it exact. A mean over N = world_size * n examples is
the mean of world_size shard-means, so with equal shards

    grad( mean_loss(full batch) ) = (1 / world_size) * sum_r grad( mean_loss(shard r) )

to floating-point precision. The trial asks for 1e-6. It fails when the
shards are unequal: a shard of 2 and a shard of 6 must be weighted 2:6, not
1:1 (``weighted_average_gradients``). In real training this bites when the
loss is a mean over *tokens* and ranks hold different numbers of unpadded
tokens: torch's DDP averages the per-rank means and does not know.

THE COLLECTIVE. ``torch.distributed.all_reduce(t, op=ReduceOp.SUM)`` leaves
the SUM over every rank's ``t`` in every rank's ``t``, in place. Divide by
world_size for the mean. Every rank must call it the same number of times
with tensors of the same shape, in the same order, or the line deadlocks.

REAL PROCESSES. ``launch_data_parallel`` spawns world_size Python processes
with ``torch.multiprocessing.spawn`` (the "spawn" start method: fresh
interpreters, nothing inherited but the arguments). Each runs ``worker``,
which joins the group with the gloo backend, rendezvousing through a
shared file:

    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=world_size)

then loads the batch from disk, takes its shard, builds the SAME model
(``build_foundry_model`` seeds torch before constructing it, so every rank
starts from identical weights), computes its shard's gradients, all-reduces
each one, divides by world_size, and rank 0 saves the result to
``out_dir / REDUCED_GRADS_FILE``. Every rank ends with
``dist.destroy_process_group()``. Call ``torch.set_num_threads(1)`` first
thing in the worker: several processes fighting over the same cores is
slower than one, and the trial has a clock.

``ddp_worker`` is the same job with the reduction delegated to
``torch.nn.parallel.DistributedDataParallel``, which hooks every parameter,
buckets the gradients and all-reduces each bucket as soon as it is ready,
overlapping communication with the rest of backward. After
``loss.backward()`` every ``.grad`` is already the average across ranks.

Run:  dungeon trial 7 room_6
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist  # noqa: F401  (used by worker and ddp_worker)
import torch.multiprocessing as mp  # noqa: F401  (used by launch_data_parallel)
import torch.nn as nn

from dungeon.artifacts import tiny_gpt

# The model every rank builds. Provided: the trial is about where the examples
# go, not about the tower, so a known-good tiny GPT stands in for yours.
FOUNDRY_CFG = tiny_gpt.GPTConfig(vocab_size=17, block_size=16, n_layer=1, n_head=2, n_embd=32)
FOUNDRY_SEED = 760
REDUCED_GRADS_FILE = "reduced_grads.pt"  # what rank 0 writes into out_dir


def build_foundry_model() -> nn.Module:
    """The one model every rank constructs, identically: seed first, then build. (Provided.)"""
    torch.manual_seed(FOUNDRY_SEED)
    return tiny_gpt.GPT(FOUNDRY_CFG)


def shard_batch(x: torch.Tensor, y: torch.Tensor, world_size: int, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank ``rank``'s contiguous slice of the batch axis (dim 0) of both ``x`` and ``y``.

    Shards are equal: ``x.shape[0]`` must be divisible by ``world_size``;
    raise ``ValueError`` otherwise. Rank r gets rows ``[r * n, (r + 1) * n)``
    with ``n = x.shape[0] // world_size``. Views are fine; no copy needed.
    """
    raise NotImplementedError("shard_batch() is unwritten")


def average_gradients(grad_dicts: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """The elementwise mean over a list of ``{param_name: grad}`` dicts, one per rank.

    Every dict has the same keys and shapes. Return a NEW dict of new
    tensors; do not modify the inputs. Raise ``ValueError`` on an empty list.
    """
    raise NotImplementedError("average_gradients() is unwritten")


def weighted_average_gradients(
    grad_dicts: Sequence[dict[str, torch.Tensor]], sizes: Sequence[int]
) -> dict[str, torch.Tensor]:
    """The shard-size-weighted mean: ``sum_r sizes[r] * grad_r / sum(sizes)``.

    This is what makes a mean-loss gradient exact when the shards are NOT
    equal. With equal sizes it is ``average_gradients``. Raise ``ValueError``
    if the lengths differ or the list is empty.
    """
    raise NotImplementedError("weighted_average_gradients() is unwritten")


def simulated_data_parallel_step(
    make_model: Callable[[], nn.Module], x: torch.Tensor, y: torch.Tensor, world_size: int
) -> dict[str, torch.Tensor]:
    """Data parallelism in one process: what ``world_size`` ranks would compute, then average.

    Build ONE model with ``make_model()`` (call it exactly once) and put it
    in ``train()`` mode. For each rank in turn: ``zero_grad(set_to_none=True)``,
    forward its shard (``_, loss = model(xs, ys)``; the loss is already a
    mean), ``loss.backward()``, and record ``{name: p.grad.detach().clone()}``
    over ``model.named_parameters()``. Return ``average_gradients`` of the
    per-rank dicts. The parameters never change, so every rank differentiates
    the same function at the same point, exactly as real replicas do.
    """
    raise NotImplementedError("simulated_data_parallel_step() is unwritten")


def worker(rank: int, world_size: int, init_file: str, x_path: str, y_path: str, out_dir: str) -> None:
    """One rank of the foundry line, run in its own process by ``launch_data_parallel``.

    1. ``torch.set_num_threads(1)``.
    2. ``dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)``.
       A ``timeout=datetime.timedelta(seconds=...)`` keyword keeps a lost rank from hanging the rest forever.
    3. ``x = torch.load(x_path)``, ``y = torch.load(y_path)``; take this rank's shard with ``shard_batch``.
    4. ``model = build_foundry_model()`` in ``train()`` mode; forward the shard; ``loss.backward()``.
    5. For every ``name, p`` in ``model.named_parameters()``: clone ``p.grad``,
       ``dist.all_reduce(g, op=dist.ReduceOp.SUM)``, divide by ``world_size``.
    6. Rank 0 only: ``torch.save(reduced, os.path.join(out_dir, REDUCED_GRADS_FILE))``.
    7. ``dist.destroy_process_group()``.
    """
    raise NotImplementedError("worker() is unwritten")


def ddp_worker(rank: int, world_size: int, init_file: str, x_path: str, y_path: str, out_dir: str) -> None:
    """Like ``worker``, but let ``torch.nn.parallel.DistributedDataParallel`` do the reducing.

    Same steps 1-4, except the model is wrapped: ``ddp = DistributedDataParallel(model)``
    (no ``device_ids`` on CPU), and the forward goes through ``ddp``. After
    ``loss.backward()`` DDP has already all-reduced and averaged every
    gradient, so save ``{name: p.grad.detach().clone()}`` over
    ``ddp.module.named_parameters()`` (the bare model's names, no ``module.``
    prefix) from rank 0, then destroy the group.
    """
    raise NotImplementedError("ddp_worker() is unwritten")


def launch_data_parallel(
    world_size: int, x: torch.Tensor, y: torch.Tensor, tmp_dir, use_ddp: bool = False, timeout_s: float = 120.0
) -> dict[str, torch.Tensor]:
    """Run ``world_size`` real processes over the batch and return rank 0's reduced gradients.

    Save ``x`` and ``y`` under ``tmp_dir`` with ``torch.save`` (child
    processes get paths, not tensors), pick ``init_file = tmp_dir / "rendezvous"``
    (delete it if it exists: a stale file store confuses the newcomers), then

        mp.spawn(fn, args=(world_size, init_file, x_path, y_path, out_dir), nprocs=world_size, join=True)

    with ``fn = ddp_worker if use_ddp else worker``. ``join=True`` waits for
    every rank and re-raises a rank's exception here. To bound the wait,
    use ``join=False`` and loop ``ctx.join(timeout=...)`` until it returns
    True, killing the processes and raising ``TimeoutError`` past ``timeout_s``.
    Finally ``torch.load`` and return ``out_dir / REDUCED_GRADS_FILE``.
    """
    raise NotImplementedError("launch_data_parallel() is unwritten")
