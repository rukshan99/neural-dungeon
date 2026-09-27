"""The GPT Skeleton  (loot from Floor 7)

One file, no dependencies beyond torch, that you can drop into a new project:
a GPT, a training loop and a sampler, with the reasoning written next to the
code. It mirrors dungeon/artifacts/tiny_gpt.py (nanoGPT's layout and names),
so anything you read about GPT-2 internals maps onto these attributes.

    python gpt_skeleton.py path/to/some_text.txt --steps 500

Shapes, throughout:  B batch, T sequence length (<= block_size), D n_embd,
H n_head, hd = D / H head size, V vocab_size.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- config


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128  # the longest sequence the model can see: wpe has this many rows
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128  # D; must be divisible by n_head
    dropout: float = 0.0  # 0.1 for small datasets you will over-fit; 0 for big ones
    bias: bool = True  # GPT-2 has biases; many newer models drop them


# ---------------------------------------------------------------------------- bricks


class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * weight + bias, over the LAST dim.

    Every token position is normalised on its own, so the scale of the residual
    stream cannot drift as depth grows. nn.LayerNorm does the same; this is here
    so you can see it.
    """

    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        y = (x - mean) / torch.sqrt(var + self.eps) * self.weight
        return y + self.bias if self.bias is not None else y


class CausalSelfAttention(nn.Module):
    """Every position mixes information from itself and earlier positions.

    c_attn projects the stream to q, k, v in one matmul (D -> 3D). Heads are
    made by reshaping D into (H, hd): each head attends independently over its
    own hd-dim slice, then the slices are concatenated and mixed by c_proj.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular: row t may look at columns <= t. Derived, not learned,
        # so persistent=False keeps it out of checkpoints.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        hd = D // self.n_head
        q, k, v = self.c_attn(x).split(D, dim=2)  # 3 x (B, T, D)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        # Scores: how much query t wants key s. Dividing by sqrt(hd) keeps the
        # dot products O(1) so the softmax is not saturated at initialisation.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B, H, T, T)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))  # future -> -inf -> weight 0
        att = F.softmax(att, dim=-1)  # each query's weights over keys sum to 1
        att = self.attn_dropout(att)
        y = att @ v  # (B, H, T, hd): weighted sum of values
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # concatenate the heads
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise: the same two-layer network applied to every token vector.

    Widen 4x, GELU, narrow back. This is where most of the parameters (and,
    it is thought, most of the memorised facts) live.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm residual block: x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)).

    The stream x is only ever ADDED to. Each sub-layer reads a normalised copy.
    Pre-norm (norm before the sub-layer) trains stably without warmup tricks;
    the original 2017 post-norm layout is harder to train deep.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ----------------------------------------------------------------------------- model


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token -> vector
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # position -> vector
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: the matrix that embeds tokens also scores them. One
        # Parameter, two uses. Saves V*D parameters and helps small models.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # 2*n_layer sub-layers add into the residual stream. Shrinking the
        # projections that write into it keeps the stream's variance ~constant
        # with depth at init (GPT-2's trick).
        for name, p in self.named_parameters():
            if name.endswith("attn.c_proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())  # the tied weight is yielded once
        return n - self.wpe.weight.numel() if non_embedding else n

    def forward(self, idx, targets=None):
        """idx (B, T) int64 -> logits (B, T, V), and the mean next-token loss if targets are given."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))  # (B, T, D) + (T, D) -> broadcast
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            # B*T independent V-way classifications, averaged. targets[b, t] is
            # the token that actually followed idx[b, t].
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# --------------------------------------------------------------------------- decoding


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """(B, V) logits -> (B,) token ids. temperature <= 0 is greedy."""
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        s, idx = torch.sort(logits, descending=True)
        p = F.softmax(s, -1)
        remove = (p.cumsum(-1) - p) >= top_p  # mass before this token already reached p
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, s.masked_fill(remove, float("-inf")))
    return torch.multinomial(F.softmax(logits, -1), 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(model, idx, max_new_tokens, **sampling):
    """Append tokens one at a time. Crops to block_size; re-runs the prefix each step (no cache)."""
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size :])
        nxt = sample_next(logits[:, -1, :], **sampling)
        idx = torch.cat([idx, nxt[:, None]], dim=1)
    return idx


# --------------------------------------------------------------------------- training


def get_batch(data, block_size, batch_size, generator):
    """Random windows of the token stream; y is x shifted one step right."""
    ix = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


def lr_schedule(step, warmup, total, lr_max, lr_min):
    """Linear warmup (protects fresh Adam statistics), then cosine decay to lr_min."""
    if step < warmup:
        return lr_max * (step + 1) / warmup
    if step >= total:
        return lr_min
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, data, batches, generator, batch_size=32):
    was_training = model.training
    model.eval()
    total = sum(model(*get_batch(data, model.cfg.block_size, batch_size, generator))[1].item() for _ in range(batches))
    model.train(was_training)
    return total / batches


def train(model, data, steps, batch_size, lr, generator, grad_clip=1.0, log_every=100):
    # AdamW: Adam with decoupled weight decay. betas (0.9, 0.95) is the GPT default;
    # weight decay 0.1 gently shrinks weights (it should skip biases and norms in
    # a serious setup; fine here).
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = max(1, steps // 10)
    model.train()
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_schedule(step, warmup, steps, lr, lr / 10)
        x, y = get_batch(data, model.cfg.block_size, batch_size, generator)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Rescale the whole gradient so its norm is at most grad_clip: one bad
        # batch cannot throw the weights across the landscape.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:5d}  loss {loss.item():.3f}  lr {opt.param_groups[0]['lr']:.2e}  {time.time() - t0:5.1f}s")


# -------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", type=Path, help="a UTF-8 text file to learn from")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    text = args.text.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = len(data) // 10
    train_data, val_data = data[:-n_val], data[-n_val:]

    cfg = GPTConfig(vocab_size=len(chars), block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.1)
    model = GPT(cfg)
    print(f"vocab {cfg.vocab_size}  params {model.num_params():,}  chars {len(data):,}")

    train(model, train_data, args.steps, args.batch, args.lr, gen)
    print(f"val loss {estimate_loss(model, val_data, 20, gen):.3f}")

    model.eval()
    prompt = torch.tensor([[stoi[text[0]]]])
    out = generate(model, prompt, 300, temperature=0.8, top_p=0.9, generator=gen)
    print("".join(chars[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
