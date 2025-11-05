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


def moving_average(x: Iterable[float], k: int) -> np.ndarray:
    """
    Simple centered moving average with window k (odd -> centered exactly).
    """
    x = np.asarray(list(x), dtype=float)
    if k <= 1 or k > len(x):
        return x
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")


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







