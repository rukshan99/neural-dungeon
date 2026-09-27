"""gradient_check.py - a portable numerical gradient checker.  (Loot from Floor 1.)

Drop this file next to any numpy code and use it as a test fixture:

    from gradient_check import check_gradient

    report = check_gradient(loss_fn, grad_fn, x)   # x: any-shape float array
    print(report)                                   # verdict + worst element
    assert report.passed

Central differences in float64, relative error, and a verdict you can read.
numpy only. Slow on purpose: two function evaluations per element. Use it on
small inputs (a few hundred elements at most) and on *random* points.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Scalar = Callable[[np.ndarray], float]
Vector = Callable[[np.ndarray], np.ndarray]

# Verdict thresholds for float64 central differences with eps ~ 1e-5.
CORRECT_BELOW = 1e-7
SUSPICIOUS_BELOW = 1e-4


def numerical_gradient(f: Scalar, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of f at x, for x of any shape. x is not modified."""
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        up = x.copy()
        down = x.copy()
        up[idx] += eps
        down[idx] -= eps
        grad[idx] = (f(up) - f(down)) / (2.0 * eps)
    return grad


def relative_error(a: np.ndarray, b: np.ndarray, tiny: float = 1e-12) -> float:
    """||a - b|| / (||a|| + ||b|| + tiny). 0 is perfect, ~1 is 'not even the same direction'."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + tiny))


def verdict(rel: float) -> str:
    if rel < CORRECT_BELOW:
        return "correct"
    if rel < SUSPICIOUS_BELOW:
        return "suspicious (a kink? a loose eps? float32 somewhere?)"
    if rel > 0.9:
        return "wrong (sign error or wrong shape)"
    return "wrong"


@dataclass
class GradCheckReport:
    rel_error: float
    max_abs_diff: float
    worst_index: tuple[int, ...]
    numerical: np.ndarray
    analytic: np.ndarray
    tol: float

    @property
    def passed(self) -> bool:
        return self.rel_error < self.tol

    def __str__(self) -> str:
        num = self.numerical[self.worst_index]
        ana = self.analytic[self.worst_index]
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] relative error {self.rel_error:.3e} ({verdict(self.rel_error)}); "
            f"worst element {self.worst_index}: numerical {num:+.6e} vs analytic {ana:+.6e}"
        )


def check_gradient(f: Scalar, grad_f: Vector, x: np.ndarray, eps: float = 1e-5, tol: float = 1e-6) -> GradCheckReport:
    """Compare grad_f(x) with central differences of f at x.

    Raises ValueError if grad_f(x) does not have x's shape: that is a bug worth
    naming rather than a number worth averaging.
    """
    x = np.asarray(x, dtype=np.float64)
    numerical = numerical_gradient(f, x, eps)
    analytic = np.asarray(grad_f(x), dtype=np.float64)
    if analytic.shape != numerical.shape:
        raise ValueError(f"grad_f returned shape {analytic.shape} for an input of shape {numerical.shape}")
    diff = np.abs(numerical - analytic)
    worst = tuple(int(i) for i in np.unravel_index(int(np.argmax(diff)), diff.shape)) if diff.size else ()
    return GradCheckReport(
        rel_error=relative_error(numerical, analytic),
        max_abs_diff=float(diff.max()) if diff.size else 0.0,
        worst_index=worst,
        numerical=numerical,
        analytic=analytic,
        tol=tol,
    )


def check_gradient_at_random_points(
    f: Scalar,
    grad_f: Vector,
    shape: tuple[int, ...],
    n_points: int = 3,
    seed: int = 0,
    scale: float = 1.0,
    eps: float = 1e-5,
    tol: float = 1e-6,
) -> list[GradCheckReport]:
    """Run check_gradient at n_points random inputs of the given shape (standard normal times scale).

    Bugs love to cancel at zeros and symmetric inputs; random points give them nowhere to hide.
    """
    rng = np.random.default_rng(seed)
    return [check_gradient(f, grad_f, scale * rng.standard_normal(shape), eps=eps, tol=tol) for _ in range(n_points)]


def assert_gradient_close(f: Scalar, grad_f: Vector, x: np.ndarray, eps: float = 1e-5, tol: float = 1e-6, name: str = "gradient") -> None:
    """One-liner for test suites: raises AssertionError with the report in the message."""
    report = check_gradient(f, grad_f, x, eps=eps, tol=tol)
    if not report.passed:
        raise AssertionError(f"{name} failed the gradient check: {report}")


# ---------------------------------------------------------------------------
# Demo: python gradient_check.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=4)

    def softmax_ce(logits: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)
        logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return float(-np.mean(logp[np.arange(len(labels)), labels]))

    def softmax_ce_grad(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        p[np.arange(len(labels)), labels] -= 1.0
        return p / len(labels)

    def softmax_ce_grad_forgot_the_mean(logits: np.ndarray) -> np.ndarray:
        return softmax_ce_grad(logits) * len(labels)

    x = rng.standard_normal((4, 5))
    print("correct gradient :", check_gradient(softmax_ce, softmax_ce_grad, x))
    print("missing the 1/N  :", check_gradient(softmax_ce, softmax_ce_grad_forgot_the_mean, x))
    print("sign flipped     :", check_gradient(softmax_ce, lambda z: -softmax_ce_grad(z), x))
    for i, report in enumerate(check_gradient_at_random_points(softmax_ce, softmax_ce_grad, (4, 5))):
        print(f"random point {i}   :", report)
