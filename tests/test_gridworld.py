"""
tests/test_gridworld.py

Unit tests for the GridWorld environment implementation.

These tests verify:
- Correct handling of dimensions, indexing, and coordinate mappings
- Proper initialization and state reset behavior
- Deterministic transitions when wind is disabled
- Correct handling of walls, pits, out-of-bounds transitions
- Termination conditions (goal or pit)
- Wind-driven stochasticity and reproducibility under seeding
- Utility methods (neighbors(), sample_action(), state_index(), etc.)

This file ensures that the GridWorld implementation remains stable,
deterministic when expected, and correctly enforces environment dynamics.
"""

import os
import sys
import numpy as np
import pytest

# ---------------------------------------------------------------------
# Import setup (ensures src/ is visible when running pytest from project root)
# ---------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rl_capstone import GridWorld, WorldSettings


# =====================================================================
# Helper: small clean world
# =====================================================================

def make_empty_world(
    width=3,
    height=3,
    start=(1, 1),
    goal=(0, 0),
    wind_chance=0.0,
) -> GridWorld:
    """
    Construct a minimal GridWorld with:
      - no walls, no pits
      - configurable wind
      - consistent rewards and seed for reproducible tests

    Useful for controlled deterministic testing.
    """
    settings = WorldSettings(
        width=width,
        height=height,
        start=start,
        goal=goal,
        pits=(),
        walls=(),
        wind_chance=wind_chance,
        step_penalty=-0.01,
        goal_reward=1.0,
        pit_penalty=-1.0,
        seed=0,
    )
    return GridWorld(settings)


# =====================================================================
# Basic environment properties & indexing
# =====================================================================

def test_basic_dimensions_and_state_count():
    """
    GridWorld dimensions should match the WorldSettings.
    State count = width * height, actions = 4 fixed directions.
    """
    settings = WorldSettings()
    env = GridWorld(settings)

    assert env.rows == settings.height
    assert env.cols == settings.width
    assert env.num_states == settings.width * settings.height
    assert env.num_actions == 4


def test_to_index_and_back_roundtrip():
    """
    Converting (row, col) -> index -> (row, col) must be invertible.
    """
    env = make_empty_world(width=4, height=5)

    test_positions = [(0, 0), (0, 3), (4, 0), (2, 2), (4, 3)]
    for pos in test_positions:
        idx = env._to_index(pos)
        back = env._to_pos(idx)
        assert back == pos


def test_to_pos_out_of_bounds_raises():
    """
    _to_pos(idx) must raise an error for indices outside valid range.
    """
    env = make_empty_world(width=3, height=3)
    with pytest.raises(ValueError):
        env._to_pos(-1)
    with pytest.raises(ValueError):
        env._to_pos(env.num_states)  # out of bounds


def test_reset_sets_start_state_and_returns_index():
    """
    reset() must place the agent at the start cell and return its index.
    """
    settings = WorldSettings(start=(10, 0))
    env = GridWorld(settings)

    idx = env.reset()
    assert env.state == settings.start
    assert idx == env._to_index(settings.start)


# =====================================================================
# Step dynamics (no wind)
# =====================================================================

def test_step_simple_move_no_wind():
    """
    With wind=0, transitions should be fully deterministic.
    A right move should update state correctly.
    """
    env = make_empty_world(width=3, height=3, start=(1, 1), goal=(0, 0))
    env.reset()

    next_idx, reward, done, info = env.step(1)  # Right
    assert env.state == (1, 2)
    assert next_idx == env._to_index((1, 2))
    assert reward == env.settings.step_penalty
    assert done is False
    assert isinstance(info, dict)


def test_step_blocked_by_wall():
    """
    If the agent attempts to move into a wall, it should remain in place
    but still incur a step penalty.
    """
    settings = WorldSettings(
        width=3,
        height=3,
        start=(1, 1),
        goal=(0, 0),
        pits=(),
        walls=((1, 2),),
        wind_chance=0.0,
        step_penalty=-0.01,
        goal_reward=1.0,
        pit_penalty=-1.0,
        seed=0,
    )
    env = GridWorld(settings)
    env.reset()

    next_idx, reward, done, _ = env.step(1)  # Right into wall
    assert env.state == (1, 1)  # did not move
    assert next_idx == env._to_index((1, 1))
    assert reward == env.settings.step_penalty
    assert done is False


def test_step_blocked_out_of_bounds():
    """
    Transitions that go out of grid bounds should be blocked:
    the agent stays put but receives a step penalty.
    """
    env = make_empty_world(width=3, height=3, start=(0, 0), goal=(2, 2))
    env.reset()

    for action in (0, 3):  # Up, Left when at (0,0)
        next_idx, reward, done, _ = env.step(action)
        assert env.state == (0, 0)
        assert next_idx == env._to_index((0, 0))
        assert reward == env.settings.step_penalty
        assert done is False


# =====================================================================
# Goal & pit behavior
# =====================================================================

def test_reaching_goal_gives_goal_reward_and_terminates():
    """
    Stepping onto the goal cell must deliver goal_reward and set done=True.
    """
    env = make_empty_world(width=3, height=1, start=(0, 0), goal=(0, 1))
    env.reset()

    next_idx, reward, done, _ = env.step(1)
    assert env.state == (0, 1)
    assert next_idx == env._to_index((0, 1))
    assert reward == env.settings.goal_reward
    assert done is True


def test_entering_pit_gives_penalty_and_terminates():
    """
    Entering a pit must deliver pit_penalty and set done=True.
    """
    settings = WorldSettings(
        width=3,
        height=1,
        start=(0, 0),
        goal=(0, 2),
        pits=((0, 1),),
        walls=(),
        wind_chance=0.0,
        step_penalty=-0.01,
        goal_reward=1.0,
        pit_penalty=-1.0,
        seed=0,
    )
    env = GridWorld(settings)
    env.reset()

    next_idx, reward, done, _ = env.step(1)
    assert env.state == (0, 1)
    assert reward == env.settings.pit_penalty
    assert done is True


def test_is_terminal_for_goal_and_pit():
    """
    Both goal and pit cells must be recognized as terminal,
    whether referenced by coordinates or flat index.
    """
    settings = WorldSettings(
        width=3,
        height=2,
        start=(1, 0),
        goal=(0, 2),
        pits=((1, 2),),
        walls=(),
        wind_chance=0.0,
        step_penalty=-0.01,
        goal_reward=1.0,
        pit_penalty=-1.0,
        seed=0,
    )
    env = GridWorld(settings)

    assert env.is_terminal(settings.goal)
    assert env.is_terminal(settings.pits[0])
    assert env.is_terminal((1, 1)) is False

    idx_nonterm = env._to_index((1, 1))
    assert env.is_terminal(idx_nonterm) is False


# =====================================================================
# RNG & wind behavior
# =====================================================================

def test_sample_action_range():
    """
    sample_action() should generate valid action indices in [0, 3].
    """
    env = make_empty_world()
    env.seed(123)

    actions = [env.sample_action() for _ in range(100)]
    assert all(0 <= a < 4 for a in actions)


def test_seed_reproducibility_with_wind():
    """
    With identical settings + seed, two envs must generate identical stochastic transitions.
    """
    settings = WorldSettings(
        width=4,
        height=4,
        start=(1, 1),
        goal=(0, 0),
        pits=(),
        walls=(),
        wind_chance=0.3,
        step_penalty=-0.01,
        goal_reward=1.0,
        pit_penalty=-1.0,
        seed=123,
    )

    env1 = GridWorld(settings)
    env2 = GridWorld(settings)

    env1.seed(999)
    env2.seed(999)

    env1.reset()
    env2.reset()

    actions = [0, 1, 2, 3, 0, 1, 2, 3]
    states1, states2 = [], []

    for a in actions:
        s1, _, _, _ = env1.step(a)
        s2, _, _, _ = env2.step(a)
        states1.append(s1)
        states2.append(s2)

    assert states1 == states2


def test_wind_changes_action_when_always_on():
    """
    When wind_chance=1.0, intended action must be overridden by wind.
    For UP (0), wind may push LEFT or RIGHT depending on internal RNG.
    """
    env = make_empty_world(width=3, height=3, start=(1, 1), wind_chance=1.0)
    env.seed(0)
    env.reset()

    next_idx, _, _, _ = env.step(0)
    r, c = env.state

    assert r == 1  # same row
    assert c in (0, 2)  # pushed left or right
    assert next_idx == env._to_index((r, c))


# =====================================================================
# Convenience utilities
# =====================================================================

def test_neighbors_respects_walls_and_bounds():
    """
    neighbors(cell) must return reachable cells excluding:
    - out-of-bounds moves
    - walls
    """
    settings = WorldSettings(
        width=3,
        height=3,
        start=(1, 1),
        goal=(0, 0),
        pits=(),
        walls=((1, 2),),  # wall to the right of center
        wind_chance=0.0,
        step_penalty=-0.01,
        goal_reward=1.0,
        pit_penalty=-1.0,
        seed=0,
    )
    env = GridWorld(settings)

    nbs = env.neighbors((1, 1))
    assert set(nbs) == {(0, 1), (2, 1), (1, 0)}  # Up, Down, Left


def test_state_index_matches_internal_state():
    """
    state_index() must match the index of env.state.
    """
    env = make_empty_world(width=4, height=4, start=(2, 3), goal=(0, 0))
    env.reset()
    assert env.state == (2, 3)
    assert env.state_index() == env._to_index((2, 3))
