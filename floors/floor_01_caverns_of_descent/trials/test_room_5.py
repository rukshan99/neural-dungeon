"""TRIAL 1.5 - THE MOMENTUM CHAMBER

Each optimizer step against its reference formula on random gradient
sequences, purity (nothing passed in may change), Adam's bias correction,
and the race down a narrow valley that plain SGD loses.
"""

import math

import numpy as np
import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_5_momentum_chamber")

rng = np.random.default_rng(15)


# --------------------------------------------------------------- references
def _ref_momentum(p, g, v, lr, beta):
    v = beta * v + g
    return p - lr * v, v


def _ref_nesterov(p, g, v, lr, beta):
    v = beta * v + g
    return p - lr * (g + beta * v), v


def _ref_adam(p, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    t += 1
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g**2
    m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)
    return p - lr * m_hat / (np.sqrt(v_hat) + eps), m, v, t


def _first_step_below(history, tol):
    for k, x in enumerate(history):
        if np.linalg.norm(x) < tol:
            return k
    return None


# ---------------------------------------------------------------------- sgd
def test_sgd_step_is_a_plain_step():
    p, g = rng.standard_normal(4), rng.standard_normal(4)
    new_p, new_state = room.sgd_step(p, g, {}, {"lr": 0.3})
    assert isinstance(new_state, dict), "Return (new_params, new_state) with new_state a dict (empty is fine for SGD)."
    np.testing.assert_allclose(new_p, p - 0.3 * g, err_msg="sgd_step is params - lr * grads.")


# ------------------------------------------------------------------ momentum
def test_momentum_remembers_the_last_six_pushes():
    p = rng.standard_normal(3)
    state, v_ref, p_ref = {}, np.zeros(3), p.copy()
    for k in range(6):
        g = rng.standard_normal(3)
        p, state = room.momentum_step(p, g, state, {"lr": 0.1, "beta": 0.9})
        p_ref, v_ref = _ref_momentum(p_ref, g, v_ref, 0.1, 0.9)
        assert "v" in state, "momentum_step's state must carry the velocity under the key 'v'."
        np.testing.assert_allclose(state["v"], v_ref, rtol=1e-12, err_msg=f"step {k + 1}: v <- beta * v + g")
        np.testing.assert_allclose(p, p_ref, rtol=1e-12, err_msg=f"step {k + 1}: params <- params - lr * v")


def test_momentum_with_beta_zero_is_sgd():
    p, g = rng.standard_normal(5), rng.standard_normal(5)
    with_momentum, _ = room.momentum_step(p, g, {"v": rng.standard_normal(5)}, {"lr": 0.05, "beta": 0.0})
    plain, _ = room.sgd_step(p, g, {}, {"lr": 0.05})
    np.testing.assert_allclose(with_momentum, plain, err_msg="With beta = 0 the velocity is just g, so momentum reduces to SGD.")


def test_momentum_defaults_to_beta_of_nine_tenths():
    p, g = rng.standard_normal(3), rng.standard_normal(3)
    v0 = rng.standard_normal(3)
    new_p, _ = room.momentum_step(p, g, {"v": v0}, {"lr": 0.1})
    np.testing.assert_allclose(new_p, p - 0.1 * (0.9 * v0 + g), err_msg="When 'beta' is missing from hparams, use 0.9.")


# ------------------------------------------------------------------ nesterov
def test_nesterov_matches_the_lookahead_form():
    p = rng.standard_normal(3)
    state, v_ref, p_ref = {}, np.zeros(3), p.copy()
    for k in range(6):
        g = rng.standard_normal(3)
        p, state = room.nesterov_step(p, g, state, {"lr": 0.1, "beta": 0.9})
        p_ref, v_ref = _ref_nesterov(p_ref, g, v_ref, 0.1, 0.9)
        np.testing.assert_allclose(state["v"], v_ref, rtol=1e-12, err_msg=f"step {k + 1}: v <- beta * v + g (same as momentum)")
        np.testing.assert_allclose(p, p_ref, rtol=1e-12, err_msg=f"step {k + 1}: params <- params - lr * (g + beta * v)")


def test_nesterov_is_not_just_momentum_in_a_hat():
    p, g = rng.standard_normal(3), rng.standard_normal(3)
    hp = {"lr": 0.1, "beta": 0.9}
    p_mom, _ = room.momentum_step(p, g, {}, hp)
    p_nes, _ = room.nesterov_step(p, g, {}, hp)
    assert not np.allclose(p_mom, p_nes), "Nesterov's step includes the extra beta * v look-ahead term; it must differ from heavy ball."


# ---------------------------------------------------------------------- adam
def test_adam_matches_the_reference_step_by_step():
    p = rng.standard_normal(4)
    state = {}
    p_ref, m_ref, v_ref, t_ref = p.copy(), np.zeros(4), np.zeros(4), 0
    for k in range(8):
        g = rng.standard_normal(4) * 10 ** rng.uniform(-2, 2)
        p, state = room.adam_step(p, g, state, {"lr": 0.01})
        p_ref, m_ref, v_ref, t_ref = _ref_adam(p_ref, g, m_ref, v_ref, t_ref, lr=0.01)
        assert state.get("t") == t_ref, f"step {k + 1}: state['t'] must count steps (expected {t_ref}, got {state.get('t')})."
        np.testing.assert_allclose(state["m"], m_ref, rtol=1e-12, err_msg=f"step {k + 1}: m <- beta1 m + (1 - beta1) g")
        np.testing.assert_allclose(state["v"], v_ref, rtol=1e-12, err_msg=f"step {k + 1}: v <- beta2 v + (1 - beta2) g^2")
        np.testing.assert_allclose(p, p_ref, rtol=1e-10, err_msg=f"step {k + 1}: params <- params - lr m_hat / (sqrt(v_hat) + eps)")


def test_adams_first_step_is_lr_in_every_coordinate_whatever_the_scale():
    p = np.zeros(3)
    g = np.array([1e-3, 5.0, -100.0])
    new_p, _ = room.adam_step(p, g, {}, {"lr": 0.1})
    deltas = new_p - p
    np.testing.assert_allclose(np.abs(deltas), 0.1, rtol=1e-4, err_msg=(
        f"With bias correction the first Adam step is lr * g / |g|, i.e. +-0.1 in every coordinate. Got {deltas}. "
        "Without the correction m / sqrt(v) = 0.1 / sqrt(0.001) = 3.16, and the step would be 0.316."
    ))
    assert np.all(np.sign(deltas) == -np.sign(g)), "Each coordinate moves against its gradient."


def test_adam_ignores_the_scale_of_the_gradient():
    grads = rng.standard_normal((20, 3))
    p_small, p_big = np.zeros(3), np.zeros(3)
    s_small, s_big = {}, {}
    for g in grads:
        p_small, s_small = room.adam_step(p_small, g, s_small, {"lr": 0.05})
        p_big, s_big = room.adam_step(p_big, 1000.0 * g, s_big, {"lr": 0.05})
    np.testing.assert_allclose(p_small, p_big, atol=1e-6, err_msg=(
        "Multiplying every gradient by 1000 should leave Adam's trajectory (almost) unchanged: m_hat and sqrt(v_hat) scale together."
    ))


# -------------------------------------------------------------------- purity
@pytest.mark.parametrize("name", ["sgd_step", "momentum_step", "nesterov_step", "adam_step"])
def test_no_step_mutates_its_inputs(name):
    step = getattr(room, name)
    p, g = rng.standard_normal(3), rng.standard_normal(3)
    _, state = step(p, g, {}, {"lr": 0.1})
    p_before, g_before = p.copy(), g.copy()
    state_before = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}
    keys_before = set(state)
    step(p, g, state, {"lr": 0.1})
    np.testing.assert_array_equal(p, p_before, err_msg=f"{name} modified params in place. Return a new array.")
    np.testing.assert_array_equal(g, g_before, err_msg=f"{name} modified grads in place.")
    assert set(state) == keys_before, f"{name} added or removed keys in the state dict it was given. Return a new dict."
    for k, v in state_before.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(state[k], v, err_msg=f"{name} modified state['{k}'] in place.")
        else:
            assert state[k] == v, f"{name} modified state['{k}'] in place."


# ------------------------------------------------------------- the driver
def test_run_optimizer_with_sgd_is_gradient_descent():
    x0 = np.array([2.0, -1.0])
    before = x0.copy()
    final, history = room.run_optimizer(room.sgd_step, lambda x: x, x0, {"lr": 0.1}, steps=10)
    np.testing.assert_allclose(final, x0 * 0.9**10, rtol=1e-12)
    assert len(history) == 11 and np.array_equal(history[0], x0), "history[0] is x0 and there is one entry per step."
    np.testing.assert_array_equal(x0, before, err_msg="run_optimizer modified x0.")


# ----------------------------------------------------------------- the race
def test_momentum_outruns_sgd_down_the_narrow_valley():
    a = np.array([1.0, 100.0])  # curvatures: condition number 100
    grad = lambda x: a * x  # noqa: E731
    x0 = np.array([1.0, 1.0])
    L, mu = a.max(), a.min()
    lr_sgd = 2.0 / (L + mu)  # the best constant step for plain GD on this bowl
    lr_hb = 4.0 / (math.sqrt(L) + math.sqrt(mu)) ** 2  # Polyak's optimal heavy-ball parameters
    beta_hb = ((math.sqrt(L) - math.sqrt(mu)) / (math.sqrt(L) + math.sqrt(mu))) ** 2

    _, sgd_hist = room.run_optimizer(room.sgd_step, grad, x0, {"lr": lr_sgd}, steps=1500)
    _, hb_hist = room.run_optimizer(room.momentum_step, grad, x0, {"lr": lr_hb, "beta": beta_hb}, steps=1500)
    sgd_steps = _first_step_below(sgd_hist, 1e-6)
    hb_steps = _first_step_below(hb_hist, 1e-6)
    assert sgd_steps is not None, "Plain SGD at lr = 2/(L + mu) should reach 1e-6 within 1500 steps (about 700). It did not: check sgd_step."
    assert hb_steps is not None, "Heavy ball with Polyak's parameters should reach 1e-6 in about 100 steps. It never did: check momentum_step."
    assert hb_steps * 3 < sgd_steps, (
        f"SGD took {sgd_steps} steps, momentum {hb_steps}. Theory: SGD converges like ((k-1)/(k+1))^t and heavy ball like "
        f"((sqrt(k)-1)/(sqrt(k)+1))^t with k = 100, so momentum should need several times fewer steps."
    )


def test_adam_also_crosses_the_valley():
    a = np.array([1.0, 100.0])
    final, _ = room.run_optimizer(room.adam_step, lambda x: a * x, np.array([1.0, 1.0]), {"lr": 0.1}, steps=600)
    assert np.linalg.norm(final) < 1e-6, (
        f"Adam at lr = 0.1 should sit within 1e-6 of the bottom after 600 steps; it is at distance {np.linalg.norm(final):.2e}."
    )
