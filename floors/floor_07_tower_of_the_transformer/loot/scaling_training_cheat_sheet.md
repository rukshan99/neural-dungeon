# Scaling Training Cheat Sheet

*Loot from Floor 7. One worker built the tower. This page is for when one is not enough: the memory arithmetic, the four ways to split a training run, and the rules that keep the parallel answer equal to the single-worker one.*

## The memory arithmetic

Per parameter, mixed-precision training with Adam keeps:

| What | Bytes | Why |
|---|---|---|
| bf16 / fp16 weight | 2 | what forward and backward compute with |
| bf16 / fp16 gradient | 2 | what backward produces |
| fp32 master weight | 4 | the update is applied here, then rounded back down |
| fp32 Adam first moment `m` | 4 | |
| fp32 Adam second moment `v` | 4 | |
| **total** | **16** | before any activation |

`params × 16 bytes` is the floor: 819K (the Chronicler) is 13 MB; 1B is 16 GB; 7B is 112 GB; 70B is 1.1 TB. Activations come on top and scale with `batch × seq × layers × width` — plus a `(B, H, T, T)` attention map per layer unless a fused attention kernel recomputes it in backward. Inference is far cheaper: 2 bytes per parameter in bf16 plus the KV cache (Floor 12).

Levers before you parallelise: a smaller per-rank batch with gradient accumulation (exact; Floor 7's secret room); activation checkpointing (recompute instead of store, about a third more compute); 8-bit optimizer states; a fused attention kernel.

## The four splits

| Scheme | What is split | What every rank holds | What crosses the wire, and when | Where it lives |
|---|---|---|---|---|
| **Data parallel (DDP)** | the batch | the whole model and optimizer | gradients, once per step, by all-reduce (about 2x the gradient size per rank) | anywhere; the default |
| **ZeRO-1 / ZeRO-2 / ZeRO-3 (FSDP)** | optimizer state / plus gradients / plus parameters | `1/N` of whatever is sharded | gradients by reduce-scatter; ZeRO-3 all-gathers each layer's weights just before use (about 1.5x DDP's traffic) | anywhere; when the replica does not fit |
| **Tensor parallel** | the matrices inside a layer: columns of `fc`, rows of `proj`, attention by heads | `1/N` of each large matrix | activations `(B, T, D)`: two all-reduces per block forward, two backward, every micro-step | inside a node, over its fastest links (2-8 ranks) |
| **Pipeline parallel** | the depth: consecutive blocks per stage | its stage's blocks | boundary activations only, once per micro-batch | across nodes; cheapest in bandwidth, pays a bubble |

Bubble fraction for `p` stages and `m` micro-batches is about `(p - 1) / (m + p - 1)`: more micro-batches, smaller bubble. 1F1B scheduling caps the activation memory the in-flight micro-batches need.

3D parallelism: tensor within a node, pipeline across nodes, data across everything, usually with ZeRO-1 sharding the optimizer state on top.

## Data parallelism: the rules

1. **Equal shards.** `grad(mean over the batch) = mean over ranks of grad(mean over the shard)` only when every shard has the same size. `DistributedSampler` gives every rank the same count (it pads by default or drops with `drop_last=True`); drop the last partial batch.
2. **Watch the denominator.** A loss that is a mean over *tokens* with padding masked out gives every rank a different count, and DDP averages the per-rank means anyway. Either make every rank's token count equal (pack sequences), or sum the loss per rank, all-reduce the token count, and divide.
3. **Effective batch = `world_size x per_rank_batch`** (times accumulation steps), so fewer optimizer steps per epoch. Learning rate: linear scaling with warmup for SGD (Goyal et al., 2017); square-root scaling is a common, gentler choice for Adam-family optimisers; LLM recipes fix the global batch in tokens and tune once for it.
4. **Seed the model identically, the data differently.** DDP broadcasts rank 0's weights at construction anyway; dropout masks and data order should differ per rank (`base_seed + rank`), the weights must not.
5. **Every rank runs every collective, in the same order.** A rank that skips a `backward` (empty shard, an early `continue`) or calls one extra `all_reduce` hangs everyone else until the timeout. Give `init_process_group` a finite `timeout`.
6. **Gradient accumulation under DDP:** wrap the non-final micro-batches in `with ddp.no_sync():`, or you pay one all-reduce per micro-batch.
7. **Clip after the reduction.** `clip_grad_norm_` on the averaged gradient, which DDP has already put in `.grad` when `backward()` returns. Under FSDP use the wrapper's own `clip_grad_norm_`: the norm needs every shard.
8. **Save from one rank, load on all.** `ddp.module.state_dict()` on rank 0 (bare names, no `module.` prefix), a `barrier()`, then everyone loads.

## A minimal CPU-only DDP skeleton

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def worker(rank, world_size, init_file):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    torch.manual_seed(0)                                     # identical weights on every rank ...
    model = DDP(build_model())                               # ... and DDP broadcasts rank 0's anyway
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)
    for epoch in range(epochs):
        sampler.set_epoch(epoch)                             # a new shuffle each epoch, the same on every rank
        for x, y in loader:
            _, loss = model(x, y)                            # (logits, loss), as on Floor 7
            opt.zero_grad(set_to_none=True)
            loss.backward()                                  # the all-reduce happens in here, bucket by bucket
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    if rank == 0:
        torch.save(model.module.state_dict(), "ckpt.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2, "/tmp/rendezvous"), nprocs=2, join=True)
```

On GPUs: backend `"nccl"`, `torch.cuda.set_device(local_rank)`, `DDP(model, device_ids=[local_rank])`, and launch with `torchrun --nproc_per_node=N train.py`, which sets `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR` and `MASTER_PORT` in the environment so that `init_process_group("nccl")` (default `init_method="env://"`) needs no file.

## The all-reduce

`all_reduce(t, op=SUM)`: after the call, every rank's `t` holds the elementwise sum over all ranks, in place. The ring implementation is a reduce-scatter (each rank ends up owning the fully reduced `1/N` of the tensor) followed by an all-gather (everyone collects everyone's piece). Each phase moves `(N - 1)/N` of the tensor through each rank, so the total per rank is `2 (N - 1)/N x size`: bounded by twice the tensor however many ranks there are. That is why data parallelism scales.

Related collectives: `broadcast` (one to all), `reduce_scatter` (half of an all-reduce; ZeRO-2's gradient step), `all_gather` (the other half; ZeRO-3's weight step), `barrier` (wait for everyone).

## Reading a hang

- One rank raised and died; the others wait in a collective until the timeout. Read the *first* traceback, not the timeout.
- Mismatched collectives: one rank ran an extra forward/backward (uneven data, a conditional `continue`), so its `all_reduce` count differs. Equal shards, `drop_last=True`.
- A stale rendezvous file (`file://`) from the previous run: delete it, or use a fresh path.
- A port in use (`tcp://`, `env://`): change `MASTER_PORT`.
- Thread oversubscription on CPU: several ranks times many threads each is slower than one rank. `torch.set_num_threads(1)` per rank.
