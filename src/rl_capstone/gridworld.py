"""
GridWorld: a compact, stochastic 2D environment for RL experiments.

- Deterministic layout (walls, pits, start, goal)
- Stochastic "wind" that can rotate the intended action left/right
- Gym-like API: reset(), step(), render()/plot()
- Coordinates are (row, col) with (0, 0) at the top-left cell.

This file exposes:
    - WorldSettings: dataclass with environment configuration
    - GridWorld: the environment class
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Possible actions the agent can take in the grid world (↑→↓←)
# Encoded as: 0=Up, 1=Right, 2=Down, 3=Left
MOVES: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),  # Up
    1: (0, 1),   # Right
    2: (1, 0),   # Down
    3: (0, -1)   # Left
}

@dataclass(frozen=True)
class WorldSettings:
    """
    WorldSettings
    -------------
    Immutable configuration for the GridWorld environment.

    Parameters
    ----------
    width : int
        Number of columns in the grid.
    height : int
        Number of rows in the grid.
    start : tuple[int, int]
        Start cell (row, col).
    goal : tuple[int, int]
        Goal cell (row, col).
    pits : tuple[tuple[int, int]], optional
        Cells that end the episode with `pit_penalty`.
    walls : tuple[tuple[int, int]], optional
        Impassable cells.
    wind_chance : float
        Probability that an action is rotated left/right uniformly at random.
    step_penalty : float
        Reward applied at every time step (usually negative).
    goal_reward : float
        Reward for reaching the goal.
    pit_penalty : float
        Reward (negative) for entering a pit.
    seed : int
        Seed for the RNG used by the environment.
    """
    width: int = 11
    height: int = 11
    start: tuple[int, int] = (10, 0)
    goal: tuple[int, int] = (0, 10)

    # Position of pits around the grid world
    pits: Tuple[Tuple[int, int], ...] = (
        (2, 2), (4, 1), (6, 9), (8, 8)
    )

    # Walls (impassible cells)         
    walls: Tuple[Tuple[int, int], ...] = (
    (9, 1), (9, 2), (8, 2), (10, 4), (9, 4), (8, 4), (8, 5),
    # new vertical spines (with gaps to allow multiple routes)
    (1, 3), (2, 3), (3, 3), (6, 3),
    (0, 6), (1, 6), (3, 6), (4, 6), (8, 6), (9, 6),
    # horizontal ribbons
    (5, 1), (5, 2), (5, 4), (5, 5), (5, 8),
    # top-right loop/corridor
    (1, 8), (1, 9), (2, 9), (3, 9),
    # lower-left meander (keeps start area interesting but open)
    (8, 1),
    # mid-right pocket near existing (7,7)
    (6, 7), (7, 8), (6, 8),
    )

    wind_chance: float = 0.1      # Chance to rotate action to a random perpendicular
    step_penalty: float = -0.01   # small penalty every step
    goal_reward: float = 1.0
    pit_penalty: float = -1.0
    seed: int = 0


class GridWorld:
    """
    A stochastic 2D reinforcement learning environment where an agent moves in a grid
    with walls, pits, and a goal. Designed for comparing model-free and model-based RL
    algorithms such as Q-Learning, SARSA, and Dyna-Q.

    The agent starts at a defined position and can move in one of four directions:
    up, right, down, or left. Actions may be affected by stochastic "wind" that causes
    the agent to move in a random perpendicular direction with some probability.
    The environment provides small step penalties, a large reward for reaching the goal,
    and a large negative reward for entering pits.

    Notes
    -----
    - Coordinate system uses (row, col) with (0, 0) at the top-left.
    - Actions are encoded as: 0=Up, 1=Right, 2=Down, 3=Left.
    - The public API mirrors Gym's minimal interface: reset(), step(), render()/plot().
    """

    # ---------------------------------------------------------------------
    # Construction & basic properties
    # ---------------------------------------------------------------------
    def __init__(self, settings: WorldSettings) -> None:
        """
        Initialize a GridWorld instance.

        Parameters
        ----------
        settings : WorldSettings
            Immutable configuration for the environment. See `WorldSettings`.

        Attributes
        ----------
        rows, cols : int
            Cached height/width for quick access.
        num_states : int
            Number of grid cells.
        num_actions : int
            Always 4 (Up, Right, Down, Left).
        state : tuple[int, int]
            Current agent position (row, col).
        rng : np.random.Generator
            Random number generator for wind and stochastic transitions.
        """
        
        # store settings and setup random generator
        self.settings: WorldSettings = settings
        self.rng: np.random.Generator = np.random.default_rng(settings.seed)

        # setup dimensions and counters
        self.rows: int = settings.height
        self.cols: int = settings.width

        # number of states = rows x columns
        self.num_states: int = self.rows * self.cols
        self.num_actions: int = 4

        # start position of the agent
        self.state: Tuple[int, int] = settings.start

        # Validate critical cells
        for cell in (settings.start, settings.goal):
            self._ensure_in_bounds(cell)
        if settings.start in settings.walls:
            raise ValueError("Start cannot be a wall.")
        if settings.goal in settings.walls:
            raise ValueError("Goal cannot be a wall.")
                
        
    
    # --------------------------------------------------------
    # Indexing Helpers
    # --------------------------------------------------------

    def _to_index(self, pos: Tuple[int, int]) -> int:
        """
        Convert (row, col) to a flat integer index in [0, num_states).

        Parameters
        ----------
        pos : tuple[int, int]
            A grid coordinate (row, col).

        Returns
        -------
        int
            Flat index = row * cols + col.
        """
        r, c = pos
        return r * self.cols + c

    
    def _to_pos(self, index: int) -> Tuple[int, int]:
        """
        Convert a flat integer in [0, num_states] back to (row, col)

        Parameters
        ----------
        index : int
            Flat state index

        Returns
        -------
        tuple[int, int]
            The (row, col) position.

        Raises
        ------
        ValueError
            If the index is not in [0, num_states].
        """
        if not (0 <= index < self.num_states):
            raise ValueError(f"Index out of range: {index}")
        return (index // self.cols, index % self.cols)

    
    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        """
        Check whether a position lies within the grid.

        Parameters
        ----------
        pos : tuple[int, int]

        Returns
        -------
        bool
            True iff 0 <= row < rows and 0 <= col < cols.
        """
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    
    def _ensure_in_bounds(self, pos: Tuple[int, int]) -> None:
        """
        Assert that a position is within the grid; raise if not

        Parameters
        ----------
        pos : tuple[int, int]

        Return
        ------
        None

        Raises
        ------
        ValueError
            if `pos` is out o bounds.
        """
        if not self._in_bounds(pos):
            raise ValueError(f"Out-of-bounds position: {pos}")

    
    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """
        Check whether a position is a wall

        Parameters
        ----------
        pos : tuple[int, int]

        Returns
        -------
        bool
            True iff `pos` is in `settings.walls`.
        """
        return pos in self.settings.walls

    
    # --------------------------------------------------------
    # Public API
    # -------------------------------------------------------

    def reset(self) -> int:
        """
        Reset the environment to the start state.

        Returns
        -------
        int
            The flat index of the start state.
        """
        self.state = self.settings.start
        return self._to_index(self.state)


    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.

        With probability `wind_chance`, the intended action is rotated left or right
        (each with 0.5 probability). If the resulting move would hit a wall or leave
        the grid, the agent stays in place. Reward is `step_penalty` by default,
        `goal_reward` when reaching the goal, and `pit_penalty` when entering a pit.
        The episode terminates upon goal or pit.

        Parameters
        ----------
        action : int
            Action encoded as 0=Up, 1=Right, 2=Down, 3=Left.

        Returns
        -------
        next_state : int
            Flat index of the next state.
        reward : float
            Reward for this transition.
        done : bool
            True iff the episode has terminated (goal or pit).
        info : dict
            Extra diagnostic info (empty by default).

        Raises
        ------
        AssertionError
            If `action` is not in {0,1,2,3}.
        """
        assert 0 <= action < 4, f"Invalid action: {action}"

        # Apply wind (rotate left/right) with given probability
        if self.rng.random() < self.settings.wind_chance:
            if self.rng.random() < 0.5:
                action = (action + 1) % 4
            else:
                action = (action - 1) % 4

        dr, dc = MOVES[action]
        nr, nc = self.state[0] + dr, self.state[1] + dc
        candidate = (nr, nc)

        # Block movement if out of bounds or into a wall
        if (not self._in_bounds(candidate)) or self._is_wall(candidate):
            next_pos = self.state
        else:
            next_pos = candidate

        self.state = next_pos

        reward: float = self.settings.step_penalty
        done: bool = False

        if next_pos == self.settings.goal:
            reward = self.settings.goal_reward
            done = True
        elif next_pos in self.settings.pits:
            reward = self.settings.pit_penalty
            done = True

        return self._to_index(self.state), reward, done, {}
        

    def is_terminal(self, state: Tuple[int, int] | int) -> bool:
        """
        Check if a state is terminal (goal or pit).

        Parameters
        ----------
        state : tuple[int, int] or int
            Either a (row, col) pair or a flat index.

        Returns
        -------
        bool
            True iff `state` is the goal or a pit.
        """
        pos = state if isinstance(state, tuple) else self._to_pos(state)
        return(pos == self.settings.goal) or (pos in self.settings.pits)
        

    def sample_action(self) -> int:
        """
        Sample a random action uniformly from {0, 1, 2, 3}.

        Returns
        -------
        int
            A random action.
        """
        return int(self.rng.integers(0, 4))
        

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Reseed the environment's RNG.

        Parameters
        ----------
        seed : int or None
            New seed. If None, a random seed is drawn from OS entropy.
        """
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

            
    # ---------------------------------------------------------------------
    # Rendering (matplotlib)
    # ---------------------------------------------------------------------

    def render(self, path: Optional[Iterable[Tuple[int, int]]] = None,
               show_agent: bool = True,
               title: str = "GridWorld Environment") -> None:
        """
        Render the environment with matplotlib.

        Squares are centered, gridlines align to cell edges, and (0,0) is the
        top-left visually to match the environment's coordinate system.

        Parameters
        ----------
        path : Iterable[tuple[int, int]] or None
            Optional sequence of (row, col) cells to draw as a path.
        show_agent : bool
            If True, draws the current agent position.
        title : str
            Figure title.

        Notes
        -----
        - Uses imshow with y-axis inverted to match (0,0) at top-left.
        - Cell centers are at (c+0.5, r+0.5) in axis coordinates.
        """
        grid = np.zeros((self.rows, self.cols))
        for (r, c) in self.settings.walls:
            grid[r, c] = 1
        for (r, c) in self.settings.pits:
            grid[r, c] = 2
        gr, gc = self.settings.goal;
        grid[gr, gc] = 3
    
        colors = [
            '#eef8ea',  # 0 empty
            '#b0b0b0',  # 1 walls
            '#ef9a9a',  # 2 pits
            '#66bb6a',  # 3 goal
        ]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        
        # The origin='upper' keeps (0,0) at top-left (as in your grid logic)
        ax.imshow(grid, cmap=cmap, norm=norm, origin='lower',
                  extent=[0, self.cols, 0, self.rows], interpolation="none")

        # Put (0, 0) visually at top-left
        ax.invert_yaxis()
    
        # Major grid lines: cell edges
        ax.set_xticks(np.arange(0, self.cols + 1, 1), minor=False)
        ax.set_yticks(np.arange(0, self.rows + 1, 1), minor=False)
        ax.grid(True, which='major', color='k', linewidth=0.4, alpha=0.15)
        ax.set_xlim(0, self.cols)
        ax.set_ylim(self.rows, 0)    # because y is inverted
        ax.set_aspect('equal')

        # Hide MAJOR tick labels
        ax.tick_params(axis='both', which='major',
                       labelbottom=False, labelleft=False, length=0)

        # Centered labels as MINOR ticks at cell centers
        x_centers = np.arange(0.5, self.cols, 0.5)     # 0.5, 1.5, 2.5, ...
        y_centers = np.arange(0.5, self.rows, 0.5)

        # Build pattern: blank - 0 - blank - 1 - blank - 2 - ...
        def interleaved_labels(n: int) -> list[str]:
            """
            Build pattern: blank, '0', blank, '1', blank, '2', ...
            so that numbers appear on every alternate center.
            """
            labels=[]
            for k in range(n):
                if k % 2 == 0:
                    labels.append(str(k // 2))  # numeric label
                else:
                     labels.append("")   # blank space
            return labels
            
        # Set Grid Minor Ticks
        ax.set_xticks(x_centers, minor=True)
        ax.set_yticks(y_centers, minor=True)

        # Set Labels on Major Grid
        x_minor_labels = interleaved_labels(len(x_centers))
        y_minor_labels = interleaved_labels(len(y_centers)) 

        ax.set_xticklabels(x_minor_labels, minor=True)
        ax.set_yticklabels(y_minor_labels, minor=True)
    
        # Start marker
        sr, sc = self.settings.start
        ax.scatter(sc + 0.5, sr + 0.5, s=180, marker='D',
                   facecolors='#4fc3f7', edgecolors='black', label='Start', zorder=5)
    
        # Agent marker
        if show_agent:
            ar, ac = self.state
            ax.scatter(ac + 0.5, ar + 0.5, s=220, marker='o',
                       facecolors='#1565c0', edgecolors='black', label='Agent', zorder=6)
            
        # Goal marker
        gr, gc = self.settings.goal
        ax.scatter(gc + 0.5, gr + 0.5, s=180, marker='*',
                       facecolors="#66bb6a", edgecolors='black', label="Goal", zorder=6)
    
        # Optional path (through cell centers)
        if path is not None:
            path = list(path)
            if len(path) > 1:
                rows, cols = zip(*path)
                xs = np.asarray(cols, dtype=float) + 0.5
                ys = np.asarray(rows, dtype=float) + 0.5
                ax.plot(xs, ys, linewidth=3.2, label='Path', zorder=4)
    
        ax.set_title(title, fontsize=20, pad=10)
        ax.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0., labelspacing=1,
                  loc='upper left', frameon=True, fontsize=15)
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------
    # Convenience utilities (optional, but nice for algorithms/debugging)
    # ---------------------------------------------------------------------
    def neighbors(self, pos: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
        """
        Legal neighbor cells for an (unwinded) move from `pos` (ignores wind).

        Parameters
        ----------
        pos : tuple[int, int]

        Returns
        -------
        tuple[tuple[int, int], ...]
            Neighbor cells reachable by primitive moves (blocked positions
            are excluded; staying in place due to a block is *not* returned).
        """
        nbs = []
        for a in range(4):
            dr, dc = MOVES[a]
            q = (pos[0] + dr, pos[1] + dc)
            if self._in_bounds(q) and not self._is_wall(q):
                nbs.append(q)
        return tuple(nbs)

    def state_index(self) -> int:
        """
        Get the flat index of the current state.

        Returns
        -------
        int
        """
        return self._to_index(self.state)
    
    
    
    
    
    
    
    
    
    
    
    
    
            