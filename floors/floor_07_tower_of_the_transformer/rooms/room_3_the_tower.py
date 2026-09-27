"""ROOM 7.3 - THE TOWER

    Stack the blocks. Lay a foundation of embeddings under them and a roof
    of one last norm over them, and let the roof speak with the foundation's
    voice. When you are done, the Chronicler's own weights must fit your
    tower exactly, and it must read the chronicles with the same loss.

THE LAYOUT (nanoGPT's names; the checkpoint uses them):

    wte      nn.Embedding(vocab_size, n_embd)      what each token means
    wpe      nn.Embedding(block_size, n_embd)      where each position is
    drop     nn.Dropout(dropout)
    blocks   nn.ModuleList of n_layer Blocks
    ln_f     LayerNorm(n_embd)                     the final norm
    lm_head  nn.Linear(n_embd, vocab_size, bias=False)

WEIGHT TYING: ``self.lm_head.weight = self.wte.weight``. The same Parameter
object plays both roles: it turns a token into a vector on the way in and
scores every vector against every token on the way out. It saves
vocab_size * n_embd parameters and it is why ``parameters()`` yields that
tensor once.

INITIALISATION (GPT-2's recipe): every Linear and Embedding weight is
N(0, 0.02); every Linear bias is zero; then the weights that write into the
residual stream (every ``attn.c_proj.weight`` and ``mlp.proj.weight``) are
re-drawn with std 0.02 / sqrt(2 * n_layer), because 2 * n_layer sub-layers
add into the stream and their variances would otherwise pile up.
``self.apply(fn)`` visits every sub-module; ``self.named_parameters()``
lets you find the residual projections by name.

FORWARD:

    B, T = idx.shape;  raise ValueError if T > block_size
    pos = torch.arange(T)                                  (T,)
    x = drop(wte(idx) + wpe(pos))                         (B, T, D) + (T, D) broadcasts
    for block in blocks: x = block(x)
    logits = lm_head(ln_f(x))                             (B, T, vocab_size)
    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1)) if targets given else None

The loss averages over every (batch, position) at once: B*T predictions,
each a V-way classification of the next token.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .room_1_norm_and_nonlinearity import LayerNorm  # noqa: F401
from .room_2_the_block import Block  # noqa: F401


class GPT(nn.Module):
    """A decoder-only transformer language model.

    ``cfg`` has attributes vocab_size, block_size, n_layer, n_head, n_embd,
    and optionally dropout (default 0.0) and bias (default True). The
    dungeon's ``GPTConfig`` fits; so must any plain object with those
    attributes. Keep ``self.cfg = cfg``: later rooms read ``model.cfg.block_size``.
    """

    def __init__(self, cfg):
        raise NotImplementedError("GPT() is unwritten")

    def num_params(self, non_embedding: bool = True) -> int:
        """Total parameter count, each tensor counted once. With ``non_embedding=True`` subtract wpe."""
        raise NotImplementedError("GPT.num_params() is unwritten")

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) int64 -> (logits (B, T, vocab_size), loss or None). ValueError if T > block_size."""
        raise NotImplementedError("GPT.forward() is unwritten")
