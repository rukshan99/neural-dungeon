"""The Chronicler: a tiny character-level GPT shared by the deeper floors.

Floor 7 has you build this yourself. Floors 8 (LoRA and fine-tuning) and 12
(KV cache, quantization, serving) need a *known-good* model with pretrained
weights, so that your work on those floors is judged against a fixed base, not
against whatever your Floor 7 tower happened to learn.

Naming follows the widely used nanoGPT convention so that anything you read
about GPT-2 internals maps directly onto these attributes:

    GPT
    ├── wte   token embedding      (vocab_size, n_embd)
    ├── wpe   position embedding   (block_size, n_embd)
    ├── blocks[i]
    │   ├── ln_1  LayerNorm
    │   ├── attn  CausalSelfAttention
    │   │   ├── c_attn  Linear(n_embd, 3 * n_embd)   fused q, k, v projection
    │   │   └── c_proj  Linear(n_embd, n_embd)
    │   ├── ln_2  LayerNorm
    │   └── mlp   MLP
    │       ├── fc    Linear(n_embd, 4 * n_embd)
    │       └── proj  Linear(4 * n_embd, n_embd)
    ├── ln_f  LayerNorm
    └── lm_head Linear(n_embd, vocab_size, bias=False)   weight tied to wte

Pretrained weights: ``chronicler.pt`` (trained by scripts/train_chronicler.py on
``chronicles.txt``). Load them with :func:`load_pretrained`.

This module imports torch lazily so that the numpy-only floors never pay for it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent
CHRONICLES = ARTIFACTS / "chronicles.txt"
GOBLIN_LEDGER = ARTIFACTS / "goblin_ledger.txt"
CHECKPOINT = ARTIFACTS / "chronicler.pt"


# --------------------------------------------------------------------------- text

def read_corpus(name: str = "chronicles") -> str:
    path = {"chronicles": CHRONICLES, "ledger": GOBLIN_LEDGER}[name]
    return path.read_text(encoding="utf-8")


class CharTokenizer:
    """Every distinct character is a token. Fixed vocabulary, lossless round trip.

    The vocabulary is the sorted set of characters in the training text. Unknown
    characters at encode time map to ``unk_id`` (the id of the newline, chosen so
    that decoding stays printable) rather than raising.
    """

    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.unk_id = self.stoi.get("\n", 0)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# -------------------------------------------------------------------------- config

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- model

def _torch():
    import torch  # noqa: WPS433 - lazy on purpose

    return torch


def build_modules():
    """Define the nn.Module classes. Called lazily so importing this file needs no torch."""
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalSelfAttention(nn.Module):
        def __init__(self, cfg: GPTConfig):
            super().__init__()
            assert cfg.n_embd % cfg.n_head == 0
            self.n_head = cfg.n_head
            self.n_embd = cfg.n_embd
            self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
            self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
            self.attn_dropout = nn.Dropout(cfg.dropout)
            self.resid_dropout = nn.Dropout(cfg.dropout)
            mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
            self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            hd = C // self.n_head
            q = q.view(B, T, self.n_head, hd).transpose(1, 2)  # (B, H, T, hd)
            k = k.view(B, T, self.n_head, hd).transpose(1, 2)
            v = v.view(B, T, self.n_head, hd).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B, H, T, T)
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, H, T, hd)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_dropout(self.c_proj(y))

    class MLP(nn.Module):
        def __init__(self, cfg: GPTConfig):
            super().__init__()
            self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
            self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(self, x):
            return self.dropout(self.proj(F.gelu(self.fc(x))))

    class Block(nn.Module):
        def __init__(self, cfg: GPTConfig):
            super().__init__()
            self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
            self.attn = CausalSelfAttention(cfg)
            self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
            self.mlp = MLP(cfg)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))  # pre-norm residual
            x = x + self.mlp(self.ln_2(x))
            return x

    class GPT(nn.Module):
        def __init__(self, cfg: GPTConfig):
            super().__init__()
            self.cfg = cfg
            self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.drop = nn.Dropout(cfg.dropout)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
            self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight  # weight tying
            self.apply(self._init_weights)
            for name, p in self.named_parameters():
                if name.endswith("c_proj.weight") or name.endswith("mlp.proj.weight"):
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        @staticmethod
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def num_params(self, non_embedding: bool = True) -> int:
            n = sum(p.numel() for p in self.parameters())
            if non_embedding:
                n -= self.wpe.weight.numel()
            return n

        def forward(self, idx, targets=None):
            """idx: (B, T) int64 token ids. Returns (logits (B, T, V), loss or None)."""
            B, T = idx.shape
            if T > self.cfg.block_size:
                raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = self.drop(self.wte(idx) + self.wpe(pos))
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
            """Plain autoregressive sampling with no cache: every step re-runs the whole prefix."""
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                if temperature <= 0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float("-inf")
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
            return idx

    return {"CausalSelfAttention": CausalSelfAttention, "MLP": MLP, "Block": Block, "GPT": GPT}


def GPT(cfg: GPTConfig):
    """Construct a fresh, randomly initialised GPT."""
    return build_modules()["GPT"](cfg)


# ---------------------------------------------------------------------- checkpoint

def save_checkpoint(model, tokenizer: CharTokenizer, path: Path = CHECKPOINT, extra: dict | None = None) -> None:
    torch = _torch()
    payload = {
        "config": model.cfg.to_dict(),
        "chars": tokenizer.chars,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_pretrained(path: Path = CHECKPOINT, map_location: str = "cpu"):
    """Return ``(model, tokenizer, extra)`` for the shipped Chronicler checkpoint.

    The model is returned in ``eval()`` mode with gradients enabled on every
    parameter (Floor 8 decides what to freeze).
    """
    torch = _torch()
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Regenerate it with: python scripts/train_chronicler.py"
        )
    payload = torch.load(path, map_location=map_location)
    cfg = GPTConfig(**payload["config"])
    model = GPT(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tokenizer = CharTokenizer(payload["chars"])
    return model, tokenizer, payload.get("extra", {})
