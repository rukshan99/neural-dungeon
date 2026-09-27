"""ROOM 4.2 - THE MODULE FORGE

    An anvil, a quench trough and a rack of identical hammers. On Floor 3 you
    forged an MLP from raw matrices and carried the weights around in a dict.
    Here the forge has a mould: ``nn.Module``. Pour your layers into it and it
    keeps track of every parameter for you. That is the whole trick, and it is
    worth knowing exactly how the trick works.

``nn.Module`` is a container with bookkeeping. When you assign a Module or an
``nn.Parameter`` to ``self.something`` inside ``__init__`` (after calling
``super().__init__()``), it is *registered*: it shows up in ``parameters()``,
``named_parameters()``, ``state_dict()``, and it follows the module through
``.to(device)``, ``.train()``, ``.eval()``. A plain Python list of layers is
NOT registered; use ``nn.ModuleList`` or ``nn.Sequential``.

You call a module, not its ``forward``: ``model(x)`` runs hooks and then
``forward``. ``nn.Linear(a, b)`` stores ``weight`` of shape (b, a) and ``bias``
of shape (b,), and computes ``x @ weight.T + bias``.

``state_dict()`` is an ordered dict name -> tensor. ``torch.save`` writes any
picklable object to a file or a file-like buffer; ``torch.load`` reads it back
(since 2.6 the default ``weights_only=True`` accepts tensors and plain Python
containers only, which a state_dict is). ``load_state_dict`` is strict by
default: keys and shapes must match exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """A multi-layer perceptron over flattened inputs, returning logits.

    ``sizes`` is ``[in_features, hidden_1, ..., out_features]``; ``activation`` is a
    Module *class* such as ``nn.ReLU`` or ``nn.Tanh``.

    Required structure (the trial and ``manual_forward`` rely on these names):

        self.layers   an nn.ModuleList of nn.Linear(sizes[i], sizes[i + 1]),
                      one per consecutive pair, in order
        self.act      one instance of ``activation()``, shared between layers

    ``forward`` flattens ``(B, ...)`` to ``(B, in_features)``, then applies
    Linear, act, Linear, act, ..., Linear. No activation after the last layer:
    the output is raw logits for a loss like CrossEntropyLoss to consume.
    """

    def __init__(self, sizes: list[int], activation: type[nn.Module] = nn.ReLU) -> None:
        raise NotImplementedError("MLP.__init__() is unwritten")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("MLP.forward() is unwritten")


def count_parameters(model: nn.Module) -> int:
    """Number of scalar parameters that will be TRAINED (``requires_grad`` is True).

    Return a Python int. Frozen parameters do not count.
    """
    raise NotImplementedError("count_parameters() is unwritten")


def init_weights(model: nn.Module) -> None:
    """Re-initialise every ``nn.Linear`` in ``model``, however deeply nested.

    Weights: Kaiming (He) initialisation for ReLU, ``nn.init.kaiming_normal_``
    with ``nonlinearity="relu"`` (std = sqrt(2 / fan_in)). Biases: zeros.
    Leave non-Linear modules alone. ``model.apply(fn)`` visits every submodule.
    The init functions end in ``_`` because they work in place; they already
    run under no_grad, so you do not need to.
    """
    raise NotImplementedError("init_weights() is unwritten")


def manual_forward(model: MLP, x: torch.Tensor) -> torch.Tensor:
    """Reproduce ``model(x)`` using only the parameters and raw tensor arithmetic.

    Flatten ``x`` to ``(B, in_features)``. For each ``layer`` in ``model.layers``:
    ``h = h @ layer.weight.T + layer.bias``, then ``model.act(h)`` between layers
    (not after the last). During the trial ``nn.Linear.forward`` and
    ``F.linear`` are cold iron: calling them raises. Matmul is allowed.
    """
    raise NotImplementedError("manual_forward() is unwritten")


def serialize(model: nn.Module) -> bytes:
    """The model's ``state_dict`` as bytes, written with ``torch.save`` into an ``io.BytesIO``.

    Return the buffer's contents (``buffer.getvalue()``). Save the state_dict,
    not the module: pickled modules break when the class moves; state_dicts
    do not.
    """
    raise NotImplementedError("serialize() is unwritten")


def deserialize_into(model: nn.Module, blob: bytes) -> nn.Module:
    """Load a ``serialize`` blob into ``model`` in place and return ``model``.

    Wrap the bytes in ``io.BytesIO`` for ``torch.load``. Use the strict default
    of ``load_state_dict``: a blob from a different architecture must raise
    RuntimeError, not silently half-load.
    """
    raise NotImplementedError("deserialize_into() is unwritten")


def freeze(model: nn.Module, prefix: str) -> int:
    """Set ``requires_grad = False`` on every parameter whose name starts with ``prefix``.

    Names come from ``model.named_parameters()``, e.g. ``"layers.0.weight"``.
    Return how many parameter tensors were frozen. Frozen parameters receive
    no ``.grad`` in backward and are skipped by ``count_parameters``. (Floor 8
    freezes a whole language model this way and trains a sliver on the side.)
    """
    raise NotImplementedError("freeze() is unwritten")
