"""ROOM 8.6 - THE PREFERENCE SCALE

    A brass balance on a plinth. On one pan, the continuation the chronicles
    actually took. On the other, a passage lifted from somewhere else in the
    book: plausible, well-formed, wrong. The Chronicler must learn to tip the
    scale, and the book must not be rewritten under it.

After SFT a model answers. Preference optimisation teaches it which of two
answers is *better*. The classic route (RLHF) trains a reward model on human
comparisons and then runs PPO against it. Direct Preference Optimization
(Rafailov et al., 2023) removes both steps. The KL-regularised RL objective
has a closed-form optimum, and solving it for the reward gives

    r(x, y) = beta * log( pi(y|x) / pi_ref(y|x) ) + const(x)

Put that into the Bradley-Terry model of "chosen beats rejected" and the
constant cancels:

    L = -log sigmoid( beta * [ (log pi(y_w) - log pi_ref(y_w))
                             - (log pi(y_l) - log pi_ref(y_l)) ] )

A classification loss on log-probability ratios. No reward model, no
sampling, no PPO. ``beta * (log pi - log pi_ref)`` is the *implicit reward*.

The pieces:
  pi_ref   a frozen deep copy of the starting model. Rewards are measured
           relative to it; it is what keeps the policy from wandering off.
  beta     how far the policy may drift before the loss saturates. Large
           beta: short leash. 0.1 is the usual default.
  step 0   the policy IS the reference, every implicit reward is 0, and the
           loss is exactly ln 2 = 0.693. Watch it fall.

Two hygiene points the trials check. Dropout OFF for the policy: the loss is
a difference of log-probabilities, and dropout noise in the policy alone
reads as a fake reward (the common implementations disable it by default).
And the reference is never handed to the optimizer: diff it afterwards.
"""

from __future__ import annotations

import copy  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401

from dungeon.artifacts.tiny_gpt import CharTokenizer


def sequence_logprob(model: nn.Module, ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Summed log-probability of the response tokens, one number per row: shape ``(B,)``.

    ``ids`` is ``(B, T)`` int64. Position ``t - 1`` of the model's logits
    predicts ``ids[:, t]``. Take ``log_softmax`` over the vocabulary, gather the
    log-prob of the token that actually came next, and sum over the response
    positions ``t = prompt_len, ..., T - 1``. Prompt tokens are conditioned on,
    never scored. Keep the graph: the policy needs gradients through this.
    """
    raise NotImplementedError("sequence_logprob() is unwritten")


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(loss, chosen_reward, rejected_reward)`` from four ``(B,)`` tensors of sequence log-probs.

    ``chosen_reward = beta * (policy_chosen - ref_chosen)`` and likewise for
    rejected (the implicit rewards, returned detached for logging).
    ``loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()``, a scalar
    attached to the policy's graph. When the two margins are equal the loss is
    ln 2; it falls as the chosen margin outgrows the rejected one.
    """
    raise NotImplementedError("dpo_loss() is unwritten")


def make_preference_pairs(
    text: str,
    tokenizer: CharTokenizer,
    n: int,
    prompt_len: int,
    response_len: int,
    generator: torch.Generator | None = None,
) -> dict:
    """``{"chosen": (n, L), "rejected": (n, L), "prompt_len": prompt_len}`` with ``L = prompt_len + response_len``.

    Encode ``text`` once. For each pair draw a window start ``s`` with
    ``torch.randint(..., generator=generator)``: ``chosen`` is the window
    ``data[s : s + L]`` (prompt plus its true continuation). ``rejected``
    shares the same first ``prompt_len`` tokens but its response is
    ``data[r : r + response_len]`` for a second random start ``r`` drawn from
    anywhere in the corpus: real text, wrong place. The chronicles repeat
    themselves, so compare content: redraw ``r`` while the lifted passage
    equals the true continuation. Both tensors int64.
    """
    raise NotImplementedError("make_preference_pairs() is unwritten")


def frozen_reference(model: nn.Module) -> nn.Module:
    """A deep copy of ``model`` with every parameter ``requires_grad=False``, in eval mode.

    The input is untouched. This is pi_ref: it never sees an optimizer.
    """
    raise NotImplementedError("frozen_reference() is unwritten")


def dpo_step(
    policy: nn.Module,
    reference: nn.Module,
    batch: dict,
    beta: float,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """One DPO update on ``batch`` (a dict like ``make_preference_pairs`` returns, B rows).

    Reference log-probs under ``torch.no_grad()``; policy log-probs with the
    graph; ``dpo_loss``; ``optimizer.zero_grad(set_to_none=True)``,
    ``backward()``, ``step()``. Return Python floats:
    ``{"loss", "chosen_reward", "rejected_reward", "margin", "reward_accuracy"}``
    where rewards and margin are batch means and ``reward_accuracy`` is the
    fraction of rows with ``chosen_reward > rejected_reward``.
    """
    raise NotImplementedError("dpo_step() is unwritten")


def train_dpo(
    policy: nn.Module,
    reference: nn.Module,
    pairs: dict,
    steps: int,
    beta: float,
    lr: float,
    batch_size: int = 8,
    generator: torch.Generator | None = None,
) -> list[dict[str, float]]:
    """``steps`` DPO updates of the trainable parameters of ``policy``; return each step's dict.

    AdamW over ``[p for p in policy.parameters() if p.requires_grad]`` only.
    Each step draws ``batch_size`` row indices with ``torch.randint(...,
    generator=generator)`` and slices ``pairs["chosen"]`` / ``pairs["rejected"]``
    (keep ``"prompt_len"``). Dropout OFF: put the policy in eval mode for the
    steps (gradients still flow) and restore its previous mode after. The
    reference is only ever read.
    """
    raise NotImplementedError("train_dpo() is unwritten")


def evaluate_preferences(policy: nn.Module, reference: nn.Module, pairs: dict, beta: float, batch_size: int = 32) -> dict[str, float]:
    """Held-out bookkeeping over every pair, no grad, eval mode (restored), as Python floats.

    Keys: ``"margin"`` (mean chosen_reward - rejected_reward),
    ``"reward_accuracy"`` (fraction of pairs with chosen_reward > rejected_reward),
    ``"policy_chosen"``, ``"policy_rejected"``, ``"ref_chosen"``, ``"ref_rejected"``
    (mean summed log-probs). At step 0 margin and accuracy are both exactly 0.
    """
    raise NotImplementedError("evaluate_preferences() is unwritten")
