"""ROOM 1.5 - THE MOMENTUM CHAMBER

    A long, narrow valley. A boulder sits at the top. Push it and it does not
    stop where you stop pushing: it remembers. Rolled down the flat axis it
    gathers speed; bounced between the steep walls it cancels itself out.

Plain gradient descent forgets everything between steps. The optimizers below
keep a little *state* and use it to move faster along consistent directions
and to damp oscillation along inconsistent ones. Write each as a PURE
function:

    new_params, new_state = step(params, grads, state, hparams)

* `state` is a dict; a fresh optimizer has state == {}. Missing entries mean
  "zeros" (or t = 0 for Adam). Return a NEW dict; never mutate the one passed in.
* `hparams` is a dict: "lr" always; "beta" (default 0.9) for momentum and
  Nesterov; "beta1" 0.9, "beta2" 0.999, "eps" 1e-8 for Adam. Use .get() with
  those defaults.
* Never modify `params` or `grads` in place. `params - lr * v` makes a new array.

THE FORMULAS (the trial checks these exactly, so use these conventions):

  SGD             params <- params - lr * g

  Momentum        v <- beta * v + g                 (heavy ball, PyTorch convention)
                  params <- params - lr * v

  Nesterov        v <- beta * v + g
                  params <- params - lr * (g + beta * v)
                  (the form PyTorch's SGD(nesterov=True) uses; the iterate is
                  the look-ahead point of the classical formulation)

  Adam            t <- t + 1
                  m <- beta1 * m + (1 - beta1) * g
                  v <- beta2 * v + (1 - beta2) * g^2
                  m_hat = m / (1 - beta1^t),   v_hat = v / (1 - beta2^t)
                  params <- params - lr * m_hat / (sqrt(v_hat) + eps)

Why bias correction: m and v start at zero, so after one step m = 0.1 g and
v = 0.001 g^2, and the ratio m / sqrt(v) is 3.16 g / |g| instead of the
intended g / |g|. Dividing by (1 - beta^t) undoes the shrinkage; with it, the
very first Adam step moves every coordinate by almost exactly lr, whatever
the gradient's scale. That scale-independence is Adam's whole appeal.

Why momentum wins on a narrow valley: with the best fixed lr, plain GD on a
quadratic needs about (kappa / 2) * log(1/tol) steps, where kappa is the
condition number. Heavy ball with the right beta needs about
(sqrt(kappa) / 2) * log(1/tol). For kappa = 100 that is 10x fewer, and the
trial makes you watch it happen.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Step = Callable[[np.ndarray, np.ndarray, dict, dict], tuple[np.ndarray, dict]]


def sgd_step(params: np.ndarray, grads: np.ndarray, state: dict, hparams: dict) -> tuple[np.ndarray, dict]:
    """Plain gradient descent step. Returns (params - lr * grads, state) with state unchanged (a copy is fine)."""
    raise NotImplementedError("sgd_step() is unwritten")


def momentum_step(params: np.ndarray, grads: np.ndarray, state: dict, hparams: dict) -> tuple[np.ndarray, dict]:
    """Heavy-ball momentum. State key: "v" (velocity, same shape as params, starts at zeros)."""
    raise NotImplementedError("momentum_step() is unwritten")


def nesterov_step(params: np.ndarray, grads: np.ndarray, state: dict, hparams: dict) -> tuple[np.ndarray, dict]:
    """Nesterov momentum, PyTorch form. State key: "v"."""
    raise NotImplementedError("nesterov_step() is unwritten")


def adam_step(params: np.ndarray, grads: np.ndarray, state: dict, hparams: dict) -> tuple[np.ndarray, dict]:
    """Adam with bias correction. State keys: "m", "v" (arrays like params) and "t" (int step count)."""
    raise NotImplementedError("adam_step() is unwritten")


def run_optimizer(
    step_fn: Step,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    hparams: dict,
    steps: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Drive any step function: start with state = {}, apply `steps` updates from x0.

    Returns (final params, history) with history[k] a COPY of the params after k
    updates, history[0] == x0, len(history) == steps + 1. x0 is not modified.
    (Same contract as gradient_descent in Room 1.3: that loop is this one with sgd_step.)
    """
    raise NotImplementedError("run_optimizer() is unwritten")
