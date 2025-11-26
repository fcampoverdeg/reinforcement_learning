"""
tests/test_rl_algorithms.py

Unit tests for the tabular RL algorithms defined in `rl_algorithms.py`.

The goals of this test suite are to verify that:

- The shared epsilon schedule `_epsilon` behaves as expected (monotonic decay).
- Each learning algorithm (Q-Learning, SARSA, Dyna-Q) runs end-to-end on a
  tiny deterministic GridWorld and produces a finite, non-trivial Q-table.
- Dyna-Q with planning steps enabled performs at least as well as pure
  model-free learning on a small benchmark task.
- The `*_train_with_logs` variants return correctly structured log
  dictionaries whose lengths are consistent with the training configuration.

These tests are intentionally lightweight so they can be run frequently
during development, while still catching regressions in algorithm behavior.
"""

import numpy as np
import pytest

from rl_capstone.gridworld import GridWorld, WorldSettings
from rl_capstone.rl_algorithms import (
    TrainConfig,
    _epsilon,
    q_learning,
    sarsa,
    dyna_q,
    q_learning_train_with_logs,
    sarsa_train_with_logs,
    dyna_q_train_with_logs,
    LogConfig,
)
from rl_capstone.utils import greedy_policy_from_0, evaluate_policy


# ---------------------------------------------------------------------
# Fixtures: small deterministic environment + default configs
# ---------------------------------------------------------------------

@pytest.fixture
def tiny_env():
    """
    A 3x3 deterministic GridWorld with no pits or walls.

    Layout (row, col):
        S . G
        . . .
        . . .

    - Start at (2, 0)
    - Goal at  (0, 2)
    - No wind: transitions are deterministic
    - Small step penalty, positive goal reward

    This environment is small enough to make tests fast and stable while
    still requiring non-trivial learning.
    """
    settings = WorldSettings(
        width=3,
        height=3,
        start=(2, 0),
        goal=(0, 2),
        pits=(),
        walls=(),
        wind_chance=0.0,
        step_penalty=-0.01,
        goal_reward=1.0,
        pit_penalty=-1.0,
        seed=0,
    )
    env = GridWorld(settings)
    env.seed(0)
    return env


@pytest.fixture
def tiny_cfg():
    """
    Default training configuration for most algorithm tests.

    Uses a modest number of episodes so that:
    - learning is non-trivial, and
    - the tests still run quickly.
    """
    return TrainConfig(
        episodes=50,
        max_steps=50,
        alpha=0.5,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay_steps=1000,
        seed=0,
        q_init=0.0,
        planning_steps=5,
    )
    

@pytest.fixture
def tiny_logcfg():
    """
    Logging configuration for `*_train_with_logs` tests.

    We keep `eval_episodes` and `snapshot_every` small so tests remain fast
    while still validating the snapshotting logic.
    """
    return LogConfig(
        snapshot_every=10,
        eval_episodes=3,
        seed=0,
    )


# ---------------------------------------------------------------------
# epsilon schedule
# ---------------------------------------------------------------------

def test_epsilon_schedule_monotonic(tiny_cfg):
    """
    _epsilon should:
      - start at eps_start
      - end at approximately eps_end
      - decay monotonically between them.
    """
    eps0 = _epsilon(0, tiny_cfg)
    eps_mid = _epsilon(500, tiny_cfg)
    eps_end = _epsilon(2000, tiny_cfg)

    assert eps0 == tiny_cfg.eps_start
    assert eps_mid < eps0
    assert eps_end == pytest.approx(tiny_cfg.eps_end)
    assert eps0 >= eps_mid >= eps_end


# ---------------------------------------------------------------------
# Q-Learning basic correctness
# ---------------------------------------------------------------------

def test_q_learning_runs_and_returns_valid_Q(tiny_env, tiny_cfg):
    """
    Q-learning should run end-to-end and produce a finite, non-zero Q-table.
    """
    Q = q_learning(tiny_env, tiny_cfg)
    S, A = tiny_env.num_states, tiny_env.num_actions

    assert Q.shape == (S, A)
    assert np.all(np.isfinite(Q))
    # Q should not remain all zeros (must have learned something)
    assert not np.allclose(Q, 0.0)


# ---------------------------------------------------------------------
# SARSA correctness
# ---------------------------------------------------------------------

def test_sarsa_runs_and_returns_valid_Q(tiny_env, tiny_cfg):
    """
    SARSA should run end-to-end and produce a finite, non-zero Q-table.
    """
    Q = sarsa(tiny_env, tiny_cfg)
    S, A = tiny_env.num_states, tiny_env.num_actions

    assert Q.shape == (S, A)
    assert np.all(np.isfinite(Q))
    assert not np.allclose(Q, 0.0)


# ---------------------------------------------------------------------
# Dyna-Q correctness
# ---------------------------------------------------------------------

def test_dyna_q_runs_and_returns_valid_Q(tiny_env, tiny_cfg):
    """
    Dyna-Q should run end-to-end and produce a finite, non-zero Q-table.
    """
    Q = dyna_q(tiny_env, tiny_cfg)
    S, A = tiny_env.num_states, tiny_env.num_actions

    assert Q.shape == (S, A)
    assert np.all(np.isfinite(Q))
    assert not np.allclose(Q, 0.0)


# ---------------------------------------------------------------------
# Dyna-Q planning accelerates learning
# ---------------------------------------------------------------------

def test_dyna_q_planning_improves_learning(tiny_env):
    """
    Compare Dyna-Q with and without planning.

    The version with non-zero `planning_steps` should achieve at least
    as good final return (usually better) as the version with K = 0
    when evaluated on the tiny environment.
    """
    cfg_no_plan = TrainConfig(
        episodes=60,
        max_steps=60,
        alpha=0.5,
        gamma=0.99,
        eps_start=0.3,
        eps_end=0.05,
        eps_decay_steps=200,
        seed=0,
        planning_steps=0,
    )
    cfg_plan = TrainConfig(
        episodes=60,
        max_steps=60,
        alpha=0.5,
        gamma=0.99,
        eps_start=0.3,
        eps_end=0.05,
        eps_decay_steps=200,
        seed=0,
        planning_steps=10,
    )

    Q0 = dyna_q(tiny_env, cfg_no_plan)
    Q1 = dyna_q(tiny_env, cfg_plan)

    pi0 = greedy_policy_from_0(Q0)
    pi1 = greedy_policy_from_0(Q1)

    R0, _ = evaluate_policy(tiny_env, pi0, episodes=20, max_steps=200)
    R1, _ = evaluate_policy(tiny_env, pi1, episodes=20, max_steps=200)

    # Planning should not hurt performance, and usually helps.
    assert R1 >= R0 - 0.05
    assert R1 > R0 or np.isclose(R1, R0)


# ---------------------------------------------------------------------
# Training-with-logs integration tests
# ---------------------------------------------------------------------

def test_train_with_logs_q_learning(tiny_env, tiny_cfg, tiny_logcfg):
    """
    q_learning_train_with_logs should return:
      - a valid Q-table, and
      - a log dict with 'returns', 'steps', and 'snapshots' of
        appropriate length.
    """
    Q, logs = q_learning_train_with_logs(tiny_env, tiny_cfg, logcfg=tiny_logcfg)
    assert "returns" in logs and "steps" in logs and "snapshots" in logs
    assert len(logs["returns"]) == tiny_cfg.episodes
    assert Q.shape[0] == tiny_env.num_states


def test_train_with_logs_sarsa(tiny_env, tiny_cfg, tiny_logcfg):
    """
    sarsa_train_with_logs should produce a consistent log structure.
    """
    Q, logs = sarsa_train_with_logs(tiny_env, tiny_cfg, logcfg=tiny_logcfg)
    assert "returns" in logs and "steps" in logs and "snapshots" in logs
    assert len(logs["returns"]) == tiny_cfg.episodes


def test_train_with_logs_dyna(tiny_env, tiny_cfg, tiny_logcfg):
    """
    dyna_q_train_with_logs should produce a consistent log structure and
    at least one snapshot for downstream visualization.
    """
    Q, logs = dyna_q_train_with_logs(tiny_env, tiny_cfg, logcfg=tiny_logcfg)
    assert "returns" in logs and "steps" in logs and "snapshots" in logs
    assert len(logs["returns"]) == tiny_cfg.episodes
    assert len(logs["snapshots"]) > 0
