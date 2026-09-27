"""ROOM 1.5 - THE MOMENTUM CHAMBER  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Every step is a pure function: (params, grads, state, hparams) -> (new_params,
new_state). Nothing passed in is modified. A fresh optimizer has state == {}.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Step = Callable[[np.ndarray, np.ndarray, dict, dict], tuple[np.ndarray, dict]]


def sgd_step(params: np.ndarray, grads: np.ndarray, state: dict, hparams: dict) -> tuple[np.ndarray, dict]:
    """params - lr * grads. SGD keeps no state, so the (empty) state is returned as is."""
    return params - hparams["lr"] * grads, dict(state)


def momentum_step(params: np.ndarray, grads: np.ndarray, state: dict, hparams: dict) -> tuple[np.ndarray, dict]:
    """Heavy ball (the PyTorch convention):

        v      <- beta * v + grads          (v starts at zeros)
        params <- params - lr * v
    """
    beta = hparams.get("beta", 0.9)
    v = beta * state.get("v", np.zeros_like(params)) + grads
    return params - hparams["lr"] * v, {"v": v}


def nesterov_step(params: np.ndarray, grads: np.ndarray, state: dict, hparams: dict) -> tuple[np.ndarray, dict]:
    """Nesterov momentum in the form PyTorch uses (SGD(nesterov=True)):

        v      <- beta * v + grads
        params <- params - lr * (grads + beta * v)

    The iterate here is the *look-ahead* point of the classical formulation, so
    the gradient you are handed is already the look-ahead gradient.
    """
    beta = hparams.get("beta", 0.9)
    v = beta * state.get("v", np.zeros_like(params)) + grads
    return params - hparams["lr"] * (grads + beta * v), {"v": v}


def adam_step(params: np.ndarray, grads: np.ndarray, state: dict, hparams: dict) -> tuple[np.ndarray, dict]:
    """Adam (Kingma & Ba, Algorithm 1) with bias correction.

        t      <- t + 1
        m      <- beta1 * m + (1 - beta1) * grads
        v      <- beta2 * v + (1 - beta2) * grads^2
        m_hat  =  m / (1 - beta1^t)          bias correction: m, v start at 0 and
        v_hat  =  v / (1 - beta2^t)          would otherwise be far too small early on
        params <- params - lr * m_hat / (sqrt(v_hat) + eps)
    """
    lr = hparams["lr"]
    beta1 = hparams.get("beta1", 0.9)
    beta2 = hparams.get("beta2", 0.999)
    eps = hparams.get("eps", 1e-8)

    t = state.get("t", 0) + 1
    m = beta1 * state.get("m", np.zeros_like(params)) + (1.0 - beta1) * grads
    v = beta2 * state.get("v", np.zeros_like(params)) + (1.0 - beta2) * grads**2
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    new_params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    return new_params, {"m": m, "v": v, "t": t}


def run_optimizer(
    step_fn: Step,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    hparams: dict,
    steps: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Drive any step function from x0 for `steps` updates.

    Returns (final params, history) with history[k] a copy after k updates
    (history[0] is x0; len(history) == steps + 1). x0 is not modified.
    """
    params = np.array(x0, dtype=float)
    state: dict = {}
    history = [params.copy()]
    for _ in range(steps):
        params, state = step_fn(params, grad_fn(params), state, hparams)
        history.append(params.copy())
    return params, history
