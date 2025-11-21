"""
utils.py - Small, reusable helpers for tabular RL and experiment management.

Includes:
- Seeding and RNG utilities
- Epsilon / temperature schedules
- Epsilon-greedy & softmax action selection (with tie-breaking)
- Q-tabe helpers and greedy policy extraction
- Evaluation harness (average return over N episodes)
- Simple moving-average & plotting for learning curves
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Reproducibility / RNG
# -----------------------------

def set_seed(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a NumPy Generator seeded with `seed`.

    Parameters
    ----------
    seed : int or None
        If None, uses unpredictable entropy; else deterministic.

    Returns
    -------
    np.random.Generator
    """
    return np.random.default_rng(seed)


# -----------------------------
# Schedules
# -----------------------------

def eps_linear(t: int, eps_start: float, eps_end: float, t_max: int) -> float:
    """
    Linear epsilon schedule from eps_start to eps_end over t in [0, t_max].

    Notes
    -----
    Clamps to [eps_end, eps_start]
    """
    if t_max <= 0:
        return eps_end
    frac = max(0.0, min(1.0, t / t_max))
    return (1.0 - frac) * eps_start + frac * eps_end

def eps_exp(t: int, eps_start: float, eps_end: float, half_life: float) -> float:
    """
    Exponential epsilon schedule with a given half-life in steps.

    eps(t) = eps_end + (eps_start - eps_end) * 0.5 ** (t / half_life)
    """
    if half_life <= 0:
        return eps_end
    return eps_end + (eps_start - eps_end) * (0.5) ** (t / half_life)


# -----------------------------
# Action selection
# -----------------------------
def argmax_random_tie_break(x: np.ndarray, rng: np.random.Generator) -> int:
    """
    Argmax with uniform tie-breaking.

    Parameters
    ----------
    x : np.ndarray shape (A,)
    rng : np.random.Generator

    Returns
    -------
    int
        Index of the chosen maximum
    """
    maxv = np.max(x)
    ties = np.flatnonzero(x == maxv)
    return int(rng.choice(ties))
    

def epsilon_greedy_action(Q: np.ndarray, s: int, epsilon: float,
                          rng: np.random.Generator) -> int:
    """
    Choose action using epsilon-greedy

    Parameters
    ----------
    Q : np.ndarray
        Table of shape (S, A).
    s : int
        State index.
    epsilon : float
        Exploration probability in [0, 1].
    rng : np.random.Generator

    Returns
    -------
    int
        Chosen action.
    """
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return argmax_random_tie_break(Q[s], rng)
    

def softmax_action(Q: np.ndarray, s: int, tau: float,
                   rng: np.random.Generator) -> int:
    """
    Choose action using Boltzmann (softmax) with temperature `tau`.

    Notes
    -----
    Lower tau -> more greedy; higher tau -> more exploratory.
    """
    z = (Q[s] - Q[s].max()) / max(1e-8, tau)
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(len(Q[s]), p=p))


# -----------------------------
# Q-table & policy helpers
# -----------------------------

def init_q_table(num_states: int, num_actions: int, init_value: float = 0.0) -> np.ndarray:
    """
    Create a tabular Q of shape (S, A) filled with `init_value`.
    """
    return np.full((num_states, num_actions), init_value, dtype=float)


def greedy_policy_from_0(Q: np.ndarray) -> np.ndarray:
    """
    Greedy policy π(s) = argmax_a Q(s, a), tie-broken uniformly.
    Returns array or shape (S,) with integer actions.
    """
    rng = np.random.default_rng(0)
    S, A = Q.shape
    pi = np.zeros(S, dtype=int)
    for s in range(S):
        pi[s] = argmax_random_tie_break(Q[s], rng)
    return pi


# -----------------------------
# Evaluation harness
# -----------------------------

def evaluate_policy(env,
                    policy: np.ndarray,
                    episodes: int = 20,
                    max_steps: int = 1000,
                    seed: Optional[int] = 123) -> Tuple[float, float]:
    """
    Evaluate a deterministic policy on `env`.

    Parameters
    ----------
    env : GridWorld-like
        Must support reset() -> s, step(a) -> (s', r, done, info)
    policy : np.darray
        Array of actions, shape (S,).
    episodes : int
        Number of test rollouts.
    max_steps : int
        Safety cap per episode
    seed : int or None
        If provided, reseeds env rng for reproducibility.


    Returns
    -------
    mean_return : float
    mean_length : float
    """
    rng = set_seed(seed)
    returns = []
    lengths = []
    for _ in range(episodes):
        env.seed(int(rng.integers(0, 10_000_000)))
        s = env.reset()
        G = 0.0
        for t in range(max_steps):
            a = int(policy[s])
            s, r, done, _ = env.step(a)
            G += r
            if done:
                lengths.append(t + 1)
                break
        else:
            lengths.append(max_steps)
        returns.append(G)
    return float(np.mean(returns)), float(np.mean(lengths))


# -----------------------------
# Logging / plotting
# -----------------------------

@dataclass
class EpisodeLog:
    """
    Per-episode metrics gathered during training
    """
    returns: List[float]
    lengths: List[int]
    
    def append(self, G: float, L: int) -> None:
        self.returns.append(G)
        self.lengths.append(L)


def plot_learning_curve(returns: List[float], window: int = 21,
                        title: str = "Learning Curve") -> None:
    """
    Plot raw and smoothed episode returns.
    """
    plt.figure(figsize=(7.5, 4))
    r = np.asarray(returns, dtype=float)
    rs = moving_average(r, window)
    plt.plot(r, alpha=0.35, label="Return (raw)")
    plt.plot(rs, linewidth=2.0, label=f"Return (MA{window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# GridWorld-specific helpers
# -----------------------------

def greedy_action(Q: np.ndarray, s: int, rng=None) -> int:
    """
    Greedy action with uniform tie-breaking over argmax_a Q[s, a].
    """
    row = Q[s]
    maxv = np.max(row)
    choices = np.flatnonzero(np.isclose(row, maxv))
    if rng is None:
        return int(np.random.choice(choices))
    return int(rng.choice(choices))

def run_greedy_episode(env, Q: np.ndarray, max_steps: int = 1000):
    """
    Roll out one greedy episode using Q on the given env.

    Returns
    -------
    G : float
        Cumulative return.
    traj : List[int]
        List of visited states (indices).
    """
    s = env.reset()
    G = 0.0
    traj = [s]
    for _ in range(max_steps):
        a = greedy_action(Q, s)
        s2, r, done, _ = env.step(a)
        G += r
        traj.append(s2)
        s = s2
        if done:
            break
    return G, traj


def rolling(x, k: int = 25) -> np.ndarray:
    """
    Rolling average similar to your 'rolling' helper:
    - uses 'valid' convolution
    - pads the front with the first smoothed value.

    This keeps the length equal to len(x).
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.array([])
    k = max(1, min(k, len(x)))
    y = np.convolve(x, np.ones(k)/k, mode="valid")
    pad = np.full(k-1, y[0])
    return np.concatenate([pad, y])


def idx_traj_to_rc_path(env, traj_idx):
    """
    Convert a trajectory of state indices into (row, col) positions
    using the GridWorld's internal _to_pos(s) helper.
    """
    return [env._to_pos(s) for s in traj_idx]


def value_grid(env, Q: np.ndarray) -> np.ndarray:
    """
    Map V(s) = max_a Q(s,a) onto an (rows x cols) grid.
    """
    V = np.max(Q, axis=1)
    G = np.zeros((env.rows, env.cols))
    for s in range(env.num_states):
        r, c = env._to_pos(s)
        G[r, c] = V[s]
    return G


def plot_value_and_policy(env, Q: np.ndarray,
                          title: str = "Value & Policy (Top-Left Origin)") -> None:
    """
    Visualize value function as a heatmap + greedy policy arrows.
    Assumes:
      - env._to_pos(s) -> (row, col)
      - env._is_wall((r,c)) and env.is_terminal((r,c)) exist (for masking).
    """
    H, W = env.rows, env.cols
    Vg = value_grid(env, Q)

    plt.figure(figsize=(6.6, 6.6))
    plt.imshow(Vg, origin='upper')
    plt.colorbar(label="V(s) = maxₐ Q(s,a)")
    plt.title(title)
    plt.xticks(range(W))
    plt.yticks(range(H))

    # arrows (0:Up, 1:Right, 2:Down, 3:Left)
    action_to_vec = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}
    X, Y, U, V = [], [], [], []
    for s in range(env.num_states):
        r, c = env._to_pos(s)

        # Skip walls/terminals if your env supports these checks
        if hasattr(env, "_is_wall") and env._is_wall((r, c)):
            continue
        if hasattr(env, "is_terminal") and env.is_terminal((r, c)):
            continue

        a = int(np.argmax(Q[s]))
        dr, dc = action_to_vec[a]
        X.append(c)
        Y.append(r)
        U.append(dc)
        V.append(dr)

    plt.quiver(X, Y, U, V, scale=1, angles='xy', scale_units='xy', width=0.004)
    plt.grid(False)
    plt.tight_layout()
    plt.show()












