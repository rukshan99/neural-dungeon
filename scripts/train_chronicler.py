#!/usr/bin/env python
"""Train the Chronicler checkpoint shipped in dungeon/artifacts/chronicler.pt.

Deterministic, CPU-friendly, a few minutes. Run from the repo root:

    python scripts/train_chronicler.py            # default: 3000 steps
    python scripts/train_chronicler.py --steps 500 --out /tmp/test.pt

The tokenizer vocabulary is built from BOTH corpora (chronicles + goblin ledger)
so that Floor 8 can fine-tune on the ledger without unknown characters.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dungeon.artifacts.tiny_gpt import (  # noqa: E402
    CHECKPOINT,
    GPT,
    CharTokenizer,
    GPTConfig,
    read_corpus,
    save_checkpoint,
)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, gen: torch.Generator):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, gen, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size, gen)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=CHECKPOINT)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    chronicles = read_corpus("chronicles")
    ledger = read_corpus("ledger")
    tok = CharTokenizer.from_text(chronicles + ledger)
    data = torch.tensor(tok.encode(chronicles), dtype=torch.long)
    n_val = len(data) // 10
    train, val = data[:-n_val], data[-n_val:]

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.1)
    model = GPT(cfg)
    print(f"vocab {tok.vocab_size}  params {model.num_params():,}  train chars {len(train):,}  val chars {len(val):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = 100

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train, cfg.block_size, args.batch, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            vl = estimate_loss(model, val, cfg.block_size, args.batch, gen)
            print(f"step {step:5d}  train {loss.item():.3f}  val {vl:.3f}  lr {lr_at(step):.2e}  {time.time() - t0:6.1f}s")

    model.eval()
    val_loss = estimate_loss(model, val, cfg.block_size, args.batch, gen, iters=50)
    ledger_data = torch.tensor(tok.encode(ledger), dtype=torch.long)
    ledger_loss = estimate_loss(model, ledger_data, cfg.block_size, args.batch, gen, iters=50)
    print(f"final val loss {val_loss:.4f}   goblin-ledger loss {ledger_loss:.4f}")

    sample = model.generate(torch.tensor([tok.encode("\n")], dtype=torch.long), 300, temperature=0.8, top_k=40)
    print("--- sample ---")
    print(tok.decode(sample[0].tolist()))
    print("--------------")

    save_checkpoint(
        model,
        tok,
        args.out,
        extra={"steps": args.steps, "val_loss": round(val_loss, 4), "ledger_loss": round(ledger_loss, 4), "seed": args.seed},
    )
    size = Path(args.out).stat().st_size / 1e6
    print(f"saved {args.out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
