"""ROOM 8.6 - THE PREFERENCE SCALE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

DPO (Rafailov et al., 2023) turns "make the model prefer y_w to y_l" into a
classification loss on log-probability ratios against a frozen reference:

    L = -log sigmoid( beta * [ (log pi(y_w) - log pi_ref(y_w)) - (log pi(y_l) - log pi_ref(y_l)) ] )

The two bracketed differences, times beta, are the implicit rewards. At step
0 the policy is the reference, both rewards are 0 and the loss is ln 2. The
policy trains with dropout off: a difference of log-probabilities has no room
for noise that the reference does not share.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from dungeon.artifacts.tiny_gpt import CharTokenizer


def sequence_logprob(model: nn.Module, ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Sum of log p(ids[:, t] | ids[:, :t]) over the response positions t >= prompt_len; shape (B,)."""
    logits, _ = model(ids)
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)  # position t - 1 predicts ids[:, t]
    per_token = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, T - 1): per_token[:, t - 1] = log p(ids[:, t])
    return per_token[:, prompt_len - 1 :].sum(dim=-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The DPO loss and the two implicit rewards (detached, for logging)."""
    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss, chosen_reward.detach(), rejected_reward.detach()


def make_preference_pairs(
    text: str,
    tokenizer: CharTokenizer,
    n: int,
    prompt_len: int,
    response_len: int,
    generator: torch.Generator | None = None,
) -> dict:
    """Prompt + true continuation (chosen) versus prompt + a continuation lifted from elsewhere (rejected)."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    length = prompt_len + response_len
    if len(data) < length + response_len:
        raise ValueError(f"text has {len(data)} tokens; need at least {length + response_len}")
    starts = torch.randint(0, len(data) - length + 1, (n,), generator=generator)
    chosen = torch.stack([data[s : s + length] for s in starts.tolist()])
    rejected = chosen.clone()
    for row, s in enumerate(starts.tolist()):
        true_continuation = chosen[row, prompt_len:]
        while True:
            r = int(torch.randint(0, len(data) - response_len + 1, (1,), generator=generator))
            lifted = data[r : r + response_len]
            if not torch.equal(lifted, true_continuation):  # the corpus repeats itself; compare content, not starts
                break
        rejected[row, prompt_len:] = lifted
    return {"chosen": chosen, "rejected": rejected, "prompt_len": prompt_len}


def frozen_reference(model: nn.Module) -> nn.Module:
    """pi_ref: a deep copy that no optimizer will ever see."""
    reference = copy.deepcopy(model)
    for p in reference.parameters():
        p.requires_grad_(False)
    reference.eval()
    return reference


def _rewards(policy: nn.Module, reference: nn.Module, batch: dict, beta: float):
    prompt_len = batch["prompt_len"]
    with torch.no_grad():
        ref_chosen = sequence_logprob(reference, batch["chosen"], prompt_len)
        ref_rejected = sequence_logprob(reference, batch["rejected"], prompt_len)
    policy_chosen = sequence_logprob(policy, batch["chosen"], prompt_len)
    policy_rejected = sequence_logprob(policy, batch["rejected"], prompt_len)
    loss, chosen_reward, rejected_reward = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    return loss, chosen_reward, rejected_reward, policy_chosen, policy_rejected, ref_chosen, ref_rejected


def dpo_step(
    policy: nn.Module,
    reference: nn.Module,
    batch: dict,
    beta: float,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """One update of the policy on one batch of preference pairs."""
    loss, chosen_reward, rejected_reward, *_rest = _rewards(policy, reference, batch, beta)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "loss": loss.item(),
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
        "margin": (chosen_reward - rejected_reward).mean().item(),
        "reward_accuracy": (chosen_reward > rejected_reward).float().mean().item(),
    }


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
    """DPO over the trainable parameters of the policy, dropout off, reference read-only."""
    params = [p for p in policy.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Nothing to train: every parameter of the policy is frozen.")
    optimizer = torch.optim.AdamW(params, lr=lr)
    n_pairs = pairs["chosen"].size(0)
    was_training = policy.training
    policy.eval()  # dropout off: the loss is a difference of log-probs and must not carry noise the reference lacks
    history: list[dict[str, float]] = []
    for _ in range(steps):
        ix = torch.randint(0, n_pairs, (batch_size,), generator=generator)
        batch = {"chosen": pairs["chosen"][ix], "rejected": pairs["rejected"][ix], "prompt_len": pairs["prompt_len"]}
        history.append(dpo_step(policy, reference, batch, beta, optimizer))
    policy.train(was_training)
    return history


@torch.no_grad()
def evaluate_preferences(policy: nn.Module, reference: nn.Module, pairs: dict, beta: float, batch_size: int = 32) -> dict[str, float]:
    """Implicit-reward margin and accuracy plus the four mean log-probs, over every pair."""
    was_training = policy.training
    policy.eval()
    sums = {k: 0.0 for k in ("margin", "reward_accuracy", "policy_chosen", "policy_rejected", "ref_chosen", "ref_rejected")}
    n_pairs = pairs["chosen"].size(0)
    for start in range(0, n_pairs, batch_size):
        batch = {k: pairs[k][start : start + batch_size] for k in ("chosen", "rejected")}
        batch["prompt_len"] = pairs["prompt_len"]
        _loss, chosen_reward, rejected_reward, pc, pr, rc, rr = _rewards(policy, reference, batch, beta)
        sums["margin"] += (chosen_reward - rejected_reward).sum().item()
        sums["reward_accuracy"] += (chosen_reward > rejected_reward).float().sum().item()
        sums["policy_chosen"] += pc.sum().item()
        sums["policy_rejected"] += pr.sum().item()
        sums["ref_chosen"] += rc.sum().item()
        sums["ref_rejected"] += rr.sum().item()
    policy.train(was_training)
    return {k: v / n_pairs for k, v in sums.items()}
