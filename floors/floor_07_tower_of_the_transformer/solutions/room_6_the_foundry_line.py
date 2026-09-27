"""ROOM 7.6 - THE FOUNDRY LINE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Data parallelism: equal shards of the batch, one model per rank, the mean
of the per-rank gradients. First simulated in one process, then for real
in world_size processes talking over gloo, by hand and then through DDP.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from dungeon.artifacts import tiny_gpt

FOUNDRY_CFG = tiny_gpt.GPTConfig(vocab_size=17, block_size=16, n_layer=1, n_head=2, n_embd=32)
FOUNDRY_SEED = 760
REDUCED_GRADS_FILE = "reduced_grads.pt"

# How long a rank waits for the others at the rendezvous and in a collective.
# Generous for a laptop, finite so that a crashed rank cannot hang the rest forever.
GROUP_TIMEOUT = timedelta(seconds=120)


def build_foundry_model() -> nn.Module:
    """Seed, then build: every rank that calls this gets bit-identical weights."""
    torch.manual_seed(FOUNDRY_SEED)
    return tiny_gpt.GPT(FOUNDRY_CFG)


# ------------------------------------------------------------------ the theorem
def shard_batch(x: torch.Tensor, y: torch.Tensor, world_size: int, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank ``rank``'s contiguous, equal slice of the batch axis."""
    n_total = x.shape[0]
    if n_total % world_size != 0:
        raise ValueError(f"a batch of {n_total} does not cut into {world_size} equal shards")
    n = n_total // world_size
    return x[rank * n : (rank + 1) * n], y[rank * n : (rank + 1) * n]


def average_gradients(grad_dicts: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Elementwise mean of the per-rank gradient dicts, as new tensors."""
    if len(grad_dicts) == 0:
        raise ValueError("no gradients to average")
    return {name: torch.stack([g[name] for g in grad_dicts]).mean(dim=0) for name in grad_dicts[0]}


def weighted_average_gradients(
    grad_dicts: Sequence[dict[str, torch.Tensor]], sizes: Sequence[int]
) -> dict[str, torch.Tensor]:
    """sum_r sizes[r] * grad_r / sum(sizes): exact for unequal shards of a mean loss."""
    if len(grad_dicts) == 0 or len(grad_dicts) != len(sizes):
        raise ValueError(f"{len(grad_dicts)} gradient dicts but {len(sizes)} sizes")
    total = float(sum(sizes))
    return {
        name: sum(size * g[name] for size, g in zip(sizes, grad_dicts)) / total
        for name in grad_dicts[0]
    }


def simulated_data_parallel_step(
    make_model: Callable[[], nn.Module], x: torch.Tensor, y: torch.Tensor, world_size: int
) -> dict[str, torch.Tensor]:
    """What ``world_size`` replicas would compute, computed one after another on one copy."""
    model = make_model()
    model.train()
    per_rank: list[dict[str, torch.Tensor]] = []
    for rank in range(world_size):
        xs, ys = shard_batch(x, y, world_size, rank)
        model.zero_grad(set_to_none=True)  # a fresh rank has no gradient yet
        _, loss = model(xs, ys)  # the loss is a MEAN over this shard's tokens
        loss.backward()
        per_rank.append({name: p.grad.detach().clone() for name, p in model.named_parameters()})
    return average_gradients(per_rank)


# -------------------------------------------------------------- real processes
def _load_shard(rank: int, world_size: int, x_path: str, y_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.load(x_path)
    y = torch.load(y_path)
    return shard_batch(x, y, world_size, rank)


def worker(rank: int, world_size: int, init_file: str, x_path: str, y_path: str, out_dir: str) -> None:
    """One rank: join the group, differentiate the shard, all-reduce by hand, rank 0 saves."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size, timeout=GROUP_TIMEOUT
    )
    try:
        xs, ys = _load_shard(rank, world_size, x_path, y_path)
        model = build_foundry_model()
        model.train()
        _, loss = model(xs, ys)
        loss.backward()
        reduced: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            g = p.grad.detach().clone()
            dist.all_reduce(g, op=dist.ReduceOp.SUM)  # every rank now holds the SUM
            reduced[name] = g / world_size  # ... and divides it into the mean
        if rank == 0:
            torch.save(reduced, os.path.join(out_dir, REDUCED_GRADS_FILE))
    finally:
        dist.destroy_process_group()


def ddp_worker(rank: int, world_size: int, init_file: str, x_path: str, y_path: str, out_dir: str) -> None:
    """The same rank with DistributedDataParallel doing the reducing during backward."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size, timeout=GROUP_TIMEOUT
    )
    try:
        xs, ys = _load_shard(rank, world_size, x_path, y_path)
        model = build_foundry_model()
        model.train()
        # DDP broadcasts rank 0's parameters to everyone at construction, registers a hook on
        # every parameter, and all-reduces bucketed gradients as backward produces them.
        ddp = nn.parallel.DistributedDataParallel(model)
        _, loss = ddp(xs, ys)
        loss.backward()  # when this returns, every .grad is already the mean across ranks
        if rank == 0:
            reduced = {name: p.grad.detach().clone() for name, p in ddp.module.named_parameters()}
            torch.save(reduced, os.path.join(out_dir, REDUCED_GRADS_FILE))
    finally:
        dist.destroy_process_group()


def launch_data_parallel(
    world_size: int, x: torch.Tensor, y: torch.Tensor, tmp_dir, use_ddp: bool = False, timeout_s: float = 120.0
) -> dict[str, torch.Tensor]:
    """Spawn ``world_size`` processes, wait for all of them, return rank 0's reduced gradients."""
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    x_path, y_path = tmp / "x.pt", tmp / "y.pt"
    torch.save(x, x_path)
    torch.save(y, y_path)
    init_file = tmp / "rendezvous"
    init_file.unlink(missing_ok=True)  # a stale file store would confuse the newcomers

    fn = ddp_worker if use_ddp else worker
    ctx = mp.spawn(
        fn, args=(world_size, str(init_file), str(x_path), str(y_path), str(tmp)), nprocs=world_size, join=False
    )
    # ctx.join(timeout) returns True once every rank is done, False while some are still
    # running (it wakes whenever any one finishes), and re-raises a rank's exception.
    deadline = time.monotonic() + timeout_s
    while not ctx.join(timeout=max(0.0, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            for proc in ctx.processes:
                if proc.is_alive():
                    proc.kill()
            raise TimeoutError(f"the {world_size} data-parallel workers did not finish within {timeout_s:.0f} s")
    return torch.load(tmp / REDUCED_GRADS_FILE)
