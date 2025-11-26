"""
tests/test_utils.py

Unit tests for utility functions in `rl_capstone.utils`.

These utilities support:
- RNG seeding
- epsilon decay schedules
- action-selection helpers (ε-greedy, softmax, tie-breaking)
- Q-table initialization
- greedy policy extraction
- evaluation of a fixed policy in GridWorld
- simple smoothing for returns/steps
- episode logging
- GridWorld-specific helpers for visualization

The goal of these tests is to ensure numerical correctness, reproducibility,
and stable API behavior across development.
"""

import numpy as np
import pytest

from rl_capstone.utils import (
    set_seed, eps_linear, eps_exp, argmax_random_tie_break,
    epsilon_greedy_action, softmax_action, init_q_table,
    greedy_policy_from_0, evaluate_policy, EpisodeLog,
    rolling, greedy_action, run_greedy_episode,
    idx_traj_to_rc_path, value_grid
)
from rl_capstone.gridworld import GridWorld, WorldSettings


# =====================================================================
# RNG SEEDING
# =====================================================================

def test_set_seed_reproducible():
    """
    set_seed(seed) should return deterministic RNGs.
    Two RNGs constructed with the same seed must generate the same sequence.
    """
    g1 = set_seed(123)
    g2 = set_seed(123)
    assert g1.integers(1000) == g2.integers(1000)


def test_set_seed_different():
    """
    RNGs with different seeds should produce different outputs.
    """
    g1 = set_seed(1)
    g2 = set_seed(2)
    assert g1.integers(1000) != g2.integers(1000)


# =====================================================================
# EPSILON SCHEDULES
# =====================================================================

def test_eps_linear_basic():
    """
    eps_linear should interpolate linearly between start and end.
    At halfway (t=5 of 10), result should be mid-way.
    """
    e = eps_linear(5, 1.0, 0.0, 10)
    assert pytest.approx(e, rel=1e-5) == 0.5


def test_eps_exp_basic():
    """
    eps_exp must return start value at t = 0 (no decay yet).
    """
    e = eps_exp(0, 1.0, 0.1, 10)
    assert e == pytest.approx(1.0)


# =====================================================================
# ACTION SELECTION
# =====================================================================

def test_argmax_random_tie_break_two_ties():
    """
    argmax_random_tie_break should choose uniformly among tied maxima.
    In this case Q[1] and Q[2] are tied for max.
    """
    rng = set_seed(0)
    x = np.array([1.0, 3.0, 3.0])

    for _ in range(20):
        a = argmax_random_tie_break(x, rng)
        assert a in (1, 2)


def test_epsilon_greedy_action():
    """
    With ε = 0, epsilon-greedy must choose the greedy action.
    """
    rng = set_seed(0)
    Q = np.array([[0.0, 1.0]])
    s = 0
    a = epsilon_greedy_action(Q, s, 0.0, rng)
    assert a == 1   # greedy action


def test_softmax_action_probabilities():
    """
    softmax_action should sample each action with some chance
    if Q-values are equal (symmetry).
    """
    rng = set_seed(0)
    Q = np.array([[1.0, 1.0]])
    s = 0

    samples = [softmax_action(Q, s, 0.1, rng) for _ in range(50)]
    assert 0 in samples and 1 in samples  # both appear


# =====================================================================
# Q-TABLE HELPERS
# =====================================================================

def test_init_q_table_shape():
    """
    init_q_table should create a table of shape (S, A) with the given init value.
    """
    Q = init_q_table(5, 4, 0.5)
    assert Q.shape == (5, 4)
    assert np.allclose(Q, 0.5)


def test_greedy_policy_from_0_simple():
    """
    greedy_policy_from_0 must choose argmax for each state row.
    """
    Q = np.array([[1.0, 2.0], [3.0, 1.0]])
    pi = greedy_policy_from_0(Q)
    assert (pi == np.array([1, 0])).all()


# =====================================================================
# POLICY EVALUATION
# =====================================================================

def test_evaluate_policy_runs():
    """
    evaluate_policy should run multiple rollouts and return mean return and mean length.
    """
    env = GridWorld(WorldSettings())
    Q = init_q_table(env.num_states, env.num_actions)
    pi = greedy_policy_from_0(Q)

    R, L = evaluate_policy(env, pi, episodes=3)

    assert isinstance(R, float)
    assert isinstance(L, float)


# =====================================================================
# LOGGING & SMOOTHING
# =====================================================================

def test_episode_log_append():
    """
    EpisodeLog.append should correctly store returns and lengths.
    """
    log = EpisodeLog([], [])
    log.append(1.0, 10)
    assert log.returns == [1.0]
    assert log.lengths == [10]


def test_rolling():
    """
    rolling(x, w) should compute a moving average of window w.
    """
    x = [1, 2, 3, 4]
    r = rolling(x, 2)
    assert len(r) == len(x)
    assert pytest.approx(r[-1], rel=1e-5) == 3.5  # (3+4)/2


# =====================================================================
# GRIDWORLD-SPECIFIC HELPERS
# =====================================================================

def test_greedy_action():
    """
    greedy_action should choose the action with highest Q-value.
    """
    Q = np.array([[0.0, 5.0]])
    assert greedy_action(Q, 0) == 1


def test_run_greedy_episode():
    """
    run_greedy_episode should:
    - produce a return (float)
    - produce a trajectory (list of states)
    """
    env = GridWorld(WorldSettings())
    Q = init_q_table(env.num_states, env.num_actions)

    G, traj = run_greedy_episode(env, Q)

    assert isinstance(G, float)
    assert len(traj) >= 1


def test_idx_traj_to_rc_path():
    """
    idx_traj_to_rc_path should convert state indices → (row, col) tuples.
    """
    env = GridWorld(WorldSettings())
    path = idx_traj_to_rc_path(env, [0, 1, 2])
    assert path[0] == env._to_pos(0)


def test_value_grid_dimensions():
    """
    value_grid(env, Q) should return a V(s) grid of shape (rows, cols).
    """
    env = GridWorld(WorldSettings())
    Q = init_q_table(env.num_states, env.num_actions)
    V = value_grid(env, Q)
    assert V.shape == (env.rows, env.cols)
