"""
Package init - expose a clean, minimal API for end users.

Usage
-----
from rl_capstone import GridWorld, WorldSettings
from rl_capstone import q_learning, sarsa, dyna_q
from rl_capstone import utils    # Optional: schedules, evaluation, etc.
"""

from .gridworld import GridWorld, WorldSettings
from .rl_algorithms import q_learning, sarsa, dyna_q

# Expose utils as a module so users can do: from rl_capstone import utils
from . import utils

__all__ = [
    "GridWorld",
    "WorldSettings",
    "q_learning",
    "sarsa",
    "dyna_q",
    "utils",
]

__version__ = "0.1.0"