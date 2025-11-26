"""
rl_algorithms.py - Tabular RL algorithms for GridWorld.

Implements the core learning algorithms used throughout the project:

- q_learning           : Off-policy TD control with ε-greedy exploration
- sarsa (SARSA(0))     : On-policy TD control with ε-greedy exploration
- dyna_q               : Model-based TD using a simple one-step model + planning

It also provides "train_with_logs" variants that:
    - record per-episode returns and episode lengths
    - periodically store snapshots of the Q-table for later visualization
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import rl_capstone.utils as U

from .utils import (
    set_seed,
    init_q_table,
    epsilon_greedy_action,
    argmax_random_tie_break,
)


# =====================================================================
# Configuration dataclasses
# =====================================================================

@dataclass
class TrainConfig:
    """
    Hyperparameters for tabular control algorithms.

    Parameters
    ----------
    episodes : int
        Number of training episodes.
    max_steps : int
        Maximum number of environment steps per episode
        (safety cap; the episode is truncated if this is reached).
    alpha : float
        Learning rate for all TD updates.
    gamma : float
        Discount factor ∈ [0, 1).
    eps_start : float
        Initial exploration rate ε₀ for ε-greedy policies.
    eps_end : float
        Final exploration rate ε_T after decay.
    eps_decay_steps : int
        Number of global steps over which ε is linearly decayed
        from eps_start to eps_end. If ≤ 0, ε is fixed at eps_end.
    seed : int or None
        Random seed used for NumPy’s Generator in the algorithm.
    q_init : float
        Initial value for all entries in the Q-table.
    planning_steps : int
        Number of simulated (planning) updates per real step for Dyna-Q.
        Ignored by the pure model-free algorithms.
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
    # Dyna-Q specific
    planning_steps: int = 20


@dataclass
class LogConfig:
    """
    Configuration for training-time logging and evaluation snapshots.

    These settings are used by the `*_train_with_logs` variants of the
    algorithms to control how often we:
      - take "snapshots" of the current Q-table, and
      - run greedy evaluation episodes for diagnostic purposes.

    Parameters
    ----------
    snapshot_every : int
        Take a snapshot every `snapshot_every` training episodes.
        Snapshots are also taken on the first and last episodes.
    eval_episodes : int
        Number of greedy evaluation rollouts to perform per snapshot.
        The mean return across these rollouts is stored in the logs.
    seed : int or None
        Optional separate RNG seed for evaluation; if None, uses
        the global/numpy default.
    """
    snapshot_every: int = 40
    eval_episodes: int = 5
    seed: Optional[int] = 0


# =====================================================================
# Epsilon schedule helper
# =====================================================================

def _epsilon(t: int, cfg: TrainConfig) -> float:
    """
    Linear epsilon schedule from cfg.eps_start to cfg.eps_end.

    This function is used by all algorithms to compute ε at a given
    global step index `t`.

    Parameters
    ----------
    t : int
        Global step counter (incremented every environment step,
        not every episode).
    cfg : TrainConfig
        Training configuration holding schedule parameters.

    Returns
    -------
    float
        Exploration probability ε_t ∈ [cfg.eps_end, cfg.eps_start].
    """
    if cfg.eps_decay_steps <= 0:
        # No decay: directly use the final value.
        return cfg.eps_end
    # Fraction of decay completed in [0, 1]
    frac = min(1.0, max(0.0, t / cfg.eps_decay_steps))
    # Linear interpolation between start and end.
    return (1.0 - frac) * cfg.eps_start + frac * cfg.eps_end


# =====================================================================
# Basic algorithms (no logging)
# =====================================================================

def q_learning(env, cfg: TrainConfig) -> np.ndarray:
    """
    Tabular Q-Learning with ε-greedy exploration.

    This is the *off-policy* algorithm: it learns the greedy policy
    w.r.t. Q while following an ε-greedy behavior policy.

    Parameters
    ----------
    env : GridWorld-like environment
        Must expose attributes `num_states`, `num_actions`, and methods
        `reset()` -> int, `step(a)` -> (int, float, bool, dict).
    cfg : TrainConfig
        Hyperparameters controlling learning rate, discount, and exploration.

    Returns
    -------
    Q : np.ndarray of shape (S, A)
        Learned state-action value function.
    """
    rng = set_seed(cfg.seed)
    S, A = env.num_states, env.num_actions
    Q = init_q_table(S, A, cfg.q_init)

    step_count = 0  # global step counter (for ε scheduling)

    for _ in range(cfg.episodes):
        s = env.reset()
        for _ in range(cfg.max_steps):
            # ε-greedy behavior policy
            eps = _epsilon(step_count, cfg)
            a = epsilon_greedy_action(Q, s, eps, rng)

            # Environment step
            s2, r, done, _ = env.step(a)

            # Off-policy TD target: r + γ max_a' Q(s', a')
            td_target = r + cfg.gamma * (0.0 if done else np.max(Q[s2]))
            Q[s, a] += cfg.alpha * (td_target - Q[s, a])

            s = s2
            step_count += 1
            if done:
                break

    return Q


def sarsa(env, cfg: TrainConfig) -> np.ndarray:
    """
    Tabular SARSA(0) with ε-greedy exploration.

    SARSA is an *on-policy* method: it evaluates and improves the same
    ε-greedy behavior policy it uses to interact with the environment.

    Parameters
    ----------
    env : GridWorld-like environment
        Must expose attributes `num_states`, `num_actions`, and methods
        `reset()` -> int, `step(a)` -> (int, float, bool, dict).
    cfg : TrainConfig
        Hyperparameters controlling learning rate, discount, and exploration.

    Returns
    -------
    Q : np.ndarray of shape (S, A)
        Learned state-action value function.
    """
    rng = set_seed(cfg.seed)
    S, A = env.num_states, env.num_actions
    Q = init_q_table(S, A, cfg.q_init)

    step_count = 0

    for _ in range(cfg.episodes):
        s = env.reset()
        # Choose initial action using current ε
        a = epsilon_greedy_action(Q, s, _epsilon(step_count, cfg), rng)

        for _ in range(cfg.max_steps):
            # Take action a, observe (s', r, done)
            s2, r, done, _ = env.step(a)

            # Choose next action a' from *same* ε-greedy policy
            a2 = epsilon_greedy_action(Q, s2, _epsilon(step_count, cfg), rng)

            # On-policy TD target: r + γ Q(s', a')
            td_target = r + cfg.gamma * (0.0 if done else Q[s2, a2])
            Q[s, a] += cfg.alpha * (td_target - Q[s, a])

            # Move to next state/action
            s, a = s2, a2
            step_count += 1
            if done:
                break

    return Q


def dyna_q(env, cfg: TrainConfig) -> np.ndarray:
    """
    Vanilla Dyna-Q: model-free TD updates + planning from a learned one-step model.

    The algorithm alternates between:
      1) Real experience: interact with the environment and update Q.
      2) Planning: sample previously seen (s, a) pairs, query a one-step
         model M(s, a) → (r, s', done), and perform additional Q-updates.

    Notes
    -----
    - The model is "last-visit": for each (s, a) we store only the most
      recent transition (r, s', done).
    - `cfg.planning_steps` controls how many simulated updates are done
      per real step.
    """
    rng = set_seed(cfg.seed)
    S, A = env.num_states, env.num_actions
    Q = init_q_table(S, A, cfg.q_init)

    # One-step model: dict[(s, a)] = (r, s', done)
    model: Dict[Tuple[int, int], Tuple[float, int, bool]] = {}
    # List of visited state-action pairs, used for sampling during planning
    visited: List[Tuple[int, int]] = []

    def _update_Q(s: int, a: int, r: float, s2: int, done: bool) -> None:
        """Internal TD update shared between real and simulated transitions."""
        td_target = r + cfg.gamma * (0.0 if done else np.max(Q[s2]))
        Q[s, a] += cfg.alpha * (td_target - Q[s, a])

    step_count = 0

    for _ in range(cfg.episodes):
        s = env.reset()

        for _ in range(cfg.max_steps):
            # Behavior policy (ε-greedy)
            eps = _epsilon(step_count, cfg)
            a = epsilon_greedy_action(Q, s, eps, rng)

            # ---------- Real step ----------
            s2, r, done, _ = env.step(a)

            # Update Q from real experience
            _update_Q(s, a, r, s2, done)

            # ---------- Update model ----------
            if (s, a) not in model:
                # First time this (s, a) is seen: add to sampling list
                visited.append((s, a))
            model[(s, a)] = (r, s2, done)

            # ---------- Planning updates ----------
            for _k in range(cfg.planning_steps):
                # Sample a previously seen (s, a) uniformly
                si, ai = visited[int(rng.integers(len(visited)))]
                ri, s2i, di = model[(si, ai)]
                # Simulated TD update using model transition
                _update_Q(si, ai, ri, s2i, di)

            s = s2
            step_count += 1
            if done:
                break

    return Q


# =====================================================================
# Small helpers used by the logging versions
# =====================================================================

def _greedy_action(Q: np.ndarray, s: int, rng=None) -> int:
    """
    Choose a greedy action w.r.t. Q[s] with uniform tie-breaking.

    Parameters
    ----------
    Q : np.ndarray of shape (S, A)
    s : int
        State index.
    rng : np.random.Generator or None
        Optional RNG for reproducible tie-breaking.

    Returns
    -------
    int
        Action index chosen greedily.
    """
    row = Q[s]
    maxv = np.max(row)
    choices = np.flatnonzero(np.isclose(row, maxv))
    if rng is None:
        return int(np.random.choice(choices))
    return int(rng.choice(choices))


def _run_greedy_episode(env, Q: np.ndarray, max_steps: int = 1000):
    """
    Roll out one episode using the greedy policy induced by Q.

    This is used only for evaluation/snapshotting and does *not* update Q.

    Parameters
    ----------
    env : GridWorld-like environment
    Q : np.ndarray
        State-action value table.
    max_steps : int
        Maximum steps before truncation.

    Returns
    -------
    G : float
        Cumulative reward obtained in the episode.
    traj : List[int]
        Sequence of visited states (by index).
    """
    s = env.reset()
    G = 0.0
    traj = [s]

    for _ in range(max_steps):
        a = _greedy_action(Q, s)
        s2, r, done, _ = env.step(a)
        G += r
        traj.append(s2)
        s = s2
        if done:
            break

    return G, traj


# =====================================================================
# Q-Learning with logging
# =====================================================================

def q_learning_train_with_logs(env, cfg: TrainConfig, logcfg: LogConfig):
    """
    Q-Learning training loop that records per-episode statistics and snapshots.

    This function mirrors `q_learning` but additionally returns a log dict
    suitable for plotting learning curves and visualizing the evolution of Q.

    Parameters
    ----------
    env : GridWorld-like environment
    cfg : TrainConfig
        Hyperparameters for Q-Learning.
    logcfg : LogConfig
        Controls how often to take snapshots and how to evaluate them.

    Returns
    -------
    Q : np.ndarray of shape (S, A)
        Learned Q-table.
    logs : dict
        Dictionary with keys:
          - "returns": np.ndarray of episodic returns
          - "steps":   np.ndarray of episodic lengths
          - "snapshots": list of dicts with:
                {
                  "episode": int,
                  "avg_return": float (mean greedy return),
                  "Q": np.ndarray copy of Q at snapshot time
                }
    """
    # Local import avoids circulars in some setups
    import rl_capstone.utils as U

    rng = U.set_seed(cfg.seed)
    S, A = env.num_states, env.num_actions
    Q = U.init_q_table(S, A, cfg.q_init)

    def epsilon(t: int) -> float:
        """Local ε-schedule helper (same as _epsilon but closure-friendly)."""
        if cfg.eps_decay_steps <= 0:
            return cfg.eps_end
        frac = min(1.0, max(0.0, t / cfg.eps_decay_steps))
        return (1.0 - frac) * cfg.eps_start + frac * cfg.eps_end

    step_count = 0
    returns: List[float] = []
    steps:   List[int]   = []
    snapshots: List[Dict] = []

    for ep in range(cfg.episodes):
        s = env.reset()
        G, ep_steps = 0.0, 0

        for _ in range(cfg.max_steps):
            eps = epsilon(step_count)
            a = U.epsilon_greedy_action(Q, s, eps, rng)
            s2, r, done, _ = env.step(a)

            # Q-Learning TD target (off-policy)
            td_target = r + cfg.gamma * (0.0 if done else np.max(Q[s2]))
            Q[s, a] += cfg.alpha * (td_target - Q[s, a])

            s = s2
            G += r
            ep_steps += 1
            step_count += 1
            if done:
                break

        # Store per-episode statistics
        returns.append(G)
        steps.append(ep_steps)

        # ---- Snapshotting / evaluation ----
        take = (
            (ep == 0) or
            ((ep + 1) % logcfg.snapshot_every == 0) or
            (ep == cfg.episodes - 1)
        )
        if take:
            eval_returns = []
            for _ in range(logcfg.eval_episodes):
                Ge, _traj = _run_greedy_episode(env, Q, max_steps=cfg.max_steps)
                eval_returns.append(Ge)
            snapshots.append({
                "episode": ep + 1,
                "avg_return": float(np.mean(eval_returns)),
                "Q": Q.copy(),
            })

    logs = {
        "returns": np.array(returns),
        "steps":   np.array(steps),
        "snapshots": snapshots,
    }
    return Q, logs


# =====================================================================
# SARSA with logging
# =====================================================================

def sarsa_train_with_logs(env, cfg: TrainConfig, logcfg: LogConfig):
    """
    SARSA(0) with ε-greedy behavior and training-time logging.

    This function is analogous to `sarsa`, but additionally logs episode
    returns/lengths and greedy evaluation snapshots for use in the
    visualization notebooks.

    Returns
    -------
    Q : np.ndarray
        Learned Q-table.
    logs : dict
        See `q_learning_train_with_logs` for the exact structure.
    """
    from rl_capstone.utils import set_seed, init_q_table, epsilon_greedy_action

    rng = set_seed(cfg.seed)
    S, A = env.num_states, env.num_actions
    Q = init_q_table(S, A, cfg.q_init)

    returns: List[float] = []
    steps:   List[int]   = []
    snapshots: List[Dict] = []

    step_count = 0

    for ep in range(cfg.episodes):
        s = env.reset()
        G = 0.0
        # Initial ε-greedy action
        a = epsilon_greedy_action(Q, s, _epsilon(step_count, cfg), rng)

        for t in range(cfg.max_steps):
            s2, r, done, _ = env.step(a)

            if not done:
                # Next action from same ε-greedy policy (on-policy)
                a2 = epsilon_greedy_action(Q, s2, _epsilon(step_count + 1, cfg), rng)
                td_target = r + cfg.gamma * Q[s2, a2]
            else:
                # Terminal state: no bootstrap term
                a2 = None
                td_target = r

            Q[s, a] += cfg.alpha * (td_target - Q[s, a])

            G += r
            # If done, next action is irrelevant; we store 0 just to keep type int
            s, a = s2, (a2 if a2 is not None else 0)
            step_count += 1

            if done:
                returns.append(G)
                steps.append(t + 1)
                break
        else:
            # Hit max_steps without termination
            returns.append(G)
            steps.append(cfg.max_steps)

        # ---- Snapshotting ----
        take = (
            (ep == 0) or
            ((ep + 1) % logcfg.snapshot_every == 0) or
            (ep == cfg.episodes - 1)
        )
        if take:
            eval_returns = []
            for _ in range(logcfg.eval_episodes):
                Ge, _traj = _run_greedy_episode(env, Q, max_steps=cfg.max_steps)
                eval_returns.append(Ge)
            snapshots.append({
                "episode": ep + 1,
                "avg_return": float(np.mean(eval_returns)),
                "Q": Q.copy(),
            })

    logs = {
        "returns":  np.array(returns),
        "steps":    np.array(steps),
        "snapshots": snapshots,
    }
    return Q, logs


# =====================================================================
# Dyna-Q with logging
# =====================================================================

def dyna_q_train_with_logs(env, cfg: TrainConfig, logcfg: LogConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Dyna-Q with logging: real TD updates + planning from a one-step model.

    This is the logged version of `dyna_q`. In addition to learning Q,
    it collects per-episode statistics and periodic greedy evaluation
    snapshots for later analysis.

    Parameters
    ----------
    env : GridWorld-like environment
    cfg : TrainConfig
        Hyperparameters including `planning_steps`.
    logcfg : LogConfig
        Controls snapshot frequency and evaluation.

    Returns
    -------
    Q : np.ndarray
        Learned Q-table.
    logs : dict
        Same structure as in `q_learning_train_with_logs`.
    """
    rng = U.set_seed(cfg.seed)
    S, A = env.num_states, env.num_actions
    Q = U.init_q_table(S, A, cfg.q_init)

    # One-step model and visited list for sampling during planning
    model: Dict[Tuple[int, int], Tuple[float, int, bool]] = {}
    visited: List[Tuple[int, int]] = []

    def _update_Q(s: int, a: int, r: float, s2: int, done: bool):
        """Internal TD update used for both real and simulated transitions."""
        td_target = r + cfg.gamma * (0.0 if done else np.max(Q[s2]))
        Q[s, a] += cfg.alpha * (td_target - Q[s, a])

    returns, steps, snapshots = [], [], []
    step_count = 0

    for ep in range(cfg.episodes):
        s = env.reset()
        G, ep_steps = 0.0, 0

        for _ in range(cfg.max_steps):
            # Behavior policy (ε-greedy)
            a = U.epsilon_greedy_action(Q, s, _epsilon(step_count, cfg), rng)

            # ----- Real step -----
            s2, r, done, _ = env.step(a)
            _update_Q(s, a, r, s2, done)

            # ----- Update model memory -----
            if (s, a) not in model:
                visited.append((s, a))
            model[(s, a)] = (r, s2, done)

            # ----- Planning phase -----
            K = getattr(cfg, "planning_steps", 20)
            for _k in range(K):
                si, ai = visited[int(rng.integers(len(visited)))]
                ri, s2i, di = model[(si, ai)]
                _update_Q(si, ai, ri, s2i, di)

            s = s2
            G += r
            ep_steps += 1
            step_count += 1
            if done:
                break

        # Per-episode stats
        returns.append(G)
        steps.append(ep_steps)

        # ----- Snapshots for later rendering -----
        take = (
            (ep == 0) or
            ((ep + 1) % logcfg.snapshot_every == 0) or
            (ep == cfg.episodes - 1)
        )
        if take:
            eval_returns = []
            for _ in range(logcfg.eval_episodes):
                Ge, _traj = _run_greedy_episode(env, Q, max_steps=cfg.max_steps)
                eval_returns.append(Ge)
            snapshots.append({
                "episode": ep + 1,
                "avg_return": float(np.mean(eval_returns)),
                "Q": Q.copy(),
            })

    logs = {
        "returns":  np.array(returns),
        "steps":    np.array(steps),
        "snapshots": snapshots,
    }
    return Q, logs
