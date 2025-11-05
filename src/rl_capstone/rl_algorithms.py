"""
rl_algorithms.py - Tabular RL algorithms for GridWorld.

Implements:
- q_learning (off-policy TD control)
- sarsa(0)   (on-policy TD control)
- dyna_q     (model-learning + planning updates)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from .utils import (
    set_seed, init_q_table, epsilon_greedy_action, argmax_random_tie_break,
)

@dataclass
class TrainConfig:
    """
    Common hyperparameters for tabular control.
    """
    episodes: int = 500
    max_steps: int = 1000
    alpha: float = 0.1
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000
    seed: Optional[int] = 0
    q_init: float = 0.0
    # Dyna-Q
    planning_steps: int = 20


def _epsilon(t: int, cfg: TrainConfig) -> float:
    """
    Linear epsilon schedule from cfg.eps_start to cfg.eps_end.

    Parameters
    ----------
    t : int
        Global step counter.
    cfg : TrainConfig

    Returns
    -------
    float
    """
    if cfg.eps_decay_steps <= 0:
        return cfg.eps_end
    frac = min(1.0, max(0.0, t / cfg.eps_decay_steps))
    return (1.0 - frac) * cfg.eps_start + frac * cfg.eps_end


def q_learning(env, cfg: TrainConfig) -> np.ndarray:
    """
    Tabular Q-Learning with epsilon-greedy exploration.

    Returns
    -------
    Q : np.ndarray
        Learned Q-table of shape (S, A) ~ (State, Action).
    """
    rng = set_seed(cfg.seed)
    S, A = env.num_states, env.num_actions
    Q = init_q_table(S, A, cfg.q_init)

    step_count = 0
    for _ in range(cfg.episodes):
        s = env.reset()
        for _ in range(cfg.max_steps):
            eps = _epsilon(step_count, cfg)
            a = epsilon_greedy_action(Q, s, eps, rng)
            s2, r, done, _ = env.step(a)

            # TD target: r + gamma * max_a' Q(s', a')
            td_target = r + cfg.gamma * (0.0 if done else np.max(Q[s2]))
            Q[s, a] += cfg.alpha * (td_target - Q[s, a])

            s = s2
            step_count += 1
            if done:
                break
    return Q


def sarsa(env, cfg: TrainConfig) -> np.ndarray:
    """Tabular SARSA(0) with epsilon-greedy exploration.

    Returns
    -------
    Q : np.ndarray
        Learned Q-table of shape (S, A).
    """
    rng = set_seed(cfg.seed)
    S, A = env.num_states, env.num_actions
    Q = init_q_table(S, A, cfg.q_init)

    step_count = 0
    for _ in range(cfg.episodes):
        s = env.reset()
        a = epsilon_greedy_action(Q, s, _epsilon(step_count, cfg), rng)
        for _ in range(cfg.max_steps):
            s2, r, done, _ = env.step(a)
            a2 = epsilon_greedy_action(Q, s2, _epsilon(step_count, cfg), rng)
            td_target = r + cfg.gamma * (0.0 if done else Q[s2, a2])
            Q[s, a] += cfg.alpha * (td_target - Q[s, a])

            s, a = s2, a2
            step_count += 1
            if done:
                break
    return Q


def dyna_q(env, cfg: TrainConfig) -> np.ndarray:
    """
    Dyna-Q: Model-free TD updates + planning from a learned one-step model.

    Notes
    -----
    - Model stores the last observed (r, s', done) for each (s, a).
    - Planning draws K simulated updates per real step by sampling (s, a) pairs
      from the set of visited pairs uniformly at random.
    """
    rng = set_seed(cfg.seed)
    S, A = env.num_states, env.num_actions
    Q = init_q_table(S, A, cfg.q_init)

    # One-step model: dict[(s, a)] = (r, s2, done)
    model: Dict[Tuple[int, int], Tuple[float, int, bool]] = {}
    visited: list[Tuple[int, int]] = []

    def _update_Q(s: int, a: int, r: float, s2: int, done: bool) -> None:
        td_target = r + cfg.gamma * (0.0 if done else np.max(Q[s2]))
        Q[s, a] += cfg.alpha * (td_target - Q[s, a])

    step_count = 0
    for _ in range(cfg.episodes):
        s = env.reset()
        for _ in range(cfg.max_steps):
            eps = _epsilon(step_count, cfg)
            a = epsilon_greedy_action(Q, s, eps, rng)

            s2, r, done, _ = env.step(a)

            # Real update
            _update_Q(s, a, r, s2, done)

            # Update model
            if (s, a) not in model:
                visited.append((s, a))
            model[(s, a)] = (r, s2, done)

            # Planning
            for _k in range(cfg.planning_steps):
                si, ai = visited[int(rng.integers(len(visited)))]
                ri, s2i, di = model[(si, ai)]
                _update_Q(si, ai, ri, s2i, di)

            s = s2
            step_count += 1
            if done:
                break
    return Q
