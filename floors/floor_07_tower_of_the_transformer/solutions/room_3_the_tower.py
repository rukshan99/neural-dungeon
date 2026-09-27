"""ROOM 7.3 - THE TOWER  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The full GPT: token + position embeddings in, a stack of identical blocks,
a final LayerNorm, and a language-model head whose weight *is* the token
embedding matrix.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .room_1_norm_and_nonlinearity import LayerNorm
from .room_2_the_block import Block


class GPT(nn.Module):
    """A decoder-only transformer language model.

    ``cfg`` needs the fields vocab_size, block_size, n_layer, n_head, n_embd
    and optionally dropout (default 0.0) and bias (default True). The
    dungeon's ``GPTConfig`` has all of them; any object with those attributes
    will do.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dropout = getattr(cfg, "dropout", 0.0)
        bias = getattr(cfg, "bias", True)

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg.n_embd, cfg.n_head, cfg.block_size, bias=bias, dropout=dropout) for _ in range(cfg.n_layer)]
        )
        self.ln_f = LayerNorm(cfg.n_embd, bias=bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: the output projection reuses the input embedding
        # matrix. One tensor, two roles, vocab_size * n_embd fewer parameters.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # The residual stream receives n_layer * 2 sub-layer outputs. Scaling
        # the projections that write into it by 1/sqrt(2 n_layer) keeps its
        # variance from growing with depth at initialisation (GPT-2's trick).
        for name, p in self.named_parameters():
            if name.endswith("attn.c_proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Trainable parameters. ``parameters()`` yields the tied weight once."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) int64 token ids -> (logits (B, T, V), loss or None)."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        x = self.drop(self.wte(idx) + self.wpe(pos))  # (B, T, D) + (T, D) broadcasts
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
