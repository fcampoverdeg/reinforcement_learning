'''

This class represents the simplified, two-dimensional grid environment where an agent navigates from a starting point to a goal.

Goal: Deterministic layout + stochastic "wind" + walls/pits/goal + Gym-like API.

'''

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing import Tuple, Dict, Any

# Possible actions the agent can take in the grid world (↑→↓←)
MOVES = {
    0: (-1, 0),  # Up
    1: (0, 1),   # Right
    2: (1, 0),   # Down
    3: (0, -1)   # Left
}

@dataclass
class WorldSettings:
    '''
    WorldSettings, includes variables from the environment, like: width, height, start (x,     y), goal (x, y), pits (x, y), walls, (x, y), wind probability, step's reward, goal's         reward, pit's reward, and seed.
    '''
    width: int = 11
    height: int = 11
    start: tuple = (10, 0)
    goal: tuple = (0, 10)
    pits: tuple = ((2,2),)
    walls: tuple = ((1, 1), (1, 2), (3, 3))
    wind_chance: float = 0.1       # change to move in a random direction
    step_penalty: float = -0.01   # small penalty every step
    goal_reward: float = 1.0
    pit_penalty: float = -1.0
    seed: int = 0


class GridWorld:
    '''


    '''
    # Initializer, set up the environment for GridWorld
    def __init__(self, settings: WorldSettings):
        # store settings and setup random generator
        self.settings = settings
        self.random = np.random.default_rng(settings.seed)

        # setup dimensions and counters
        self.rows = settings.height
        self.cols = settings.width

        # start position of the agent
        self.position = settings.start

        # number of states = rows x columns
        self.num_states = self.rows * self.cols
        self.num_actions = 4
        
    
    # --------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------

    """ Convert (row, col) to a single integer index. """
    def _to_index(self, pos):
        r, c = pos
        return r * self.cols + c

    """ Check if position is inside the grid. """
    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    
    # --------------------------------------------------------
    # Main API
    # -------------------------------------------------------

    """Reset to the start position."""
    def reset(self):
        self.position = self.settings.start
        return self._to_index(self.position)

    """Take one action in the environment."""
    def step(self, action):
        assert 0 <= action < 4

        # Maybe apply wind: small chance to turn left/right
        if self.random.random() < self.settings.wind_chance:
            if self.random.random() < 0.5:
                action = (action + 1) % 4    # turn right
            else:
                action = (action -1) % 4    # turn left

        # Apply movement
        move = MOVES[action]
        new_pos = (self.position[0] + move[0], self.position[1] + move[1])

        # Check walls and boundaries
        if (not self._in_bounds(new_pos)) or (new_pos in self.settings.walls):
            new_pos = self.position    # The agent does not move

        # Update position
        self.position = new_pos

        # Compute reward
        reward = self.settings.step_penalty
        done = False

        if new_pos == self.settings.goal:
            reward = self.settings.goal_reward
            done = True
        elif new_pos in self.settings.pits:
            reward = self.settings.pit_penalty
            done = True

        return self._to_index(new_pos), reward, done, {}


    def plot(self, path=None, show_agent=True, title="GridWorld Environment"):
        """
        Professional GridWorld visualization.
        - Squares centered and correctly oriented (top row = 0)
        - Path goes through cell centers
        - Smaller, neat grid cells
        """
        grid = np.zeros((self.rows, self.cols))
        for (r, c) in self.settings.walls: grid[r, c] = 1
        for (r, c) in self.settings.pits:  grid[r, c] = 2
        gr, gc = self.settings.goal;      grid[gr, gc] = 3
    
        colors = ['#eef8ea',  # empty
                  '#b0b0b0',  # walls
                  '#ef9a9a',  # pits
                  '#66bb6a']  # goal
        cmap = ListedColormap(colors)
        norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    
        fig, ax = plt.subplots(figsize=(9, 9))
        
        # The origin='upper' keeps (0,0) at top-left (as in your grid logic)
        ax.imshow(grid, cmap=cmap, norm=norm, origin='upper',
                  extent=[0, self.cols, 0, self.rows], interpolation="none")
    
        # Grid lines at cell edges (MAJOR ticks)
        ax.set_xticks(np.arange(0, self.cols + 1, 1), minor=False)
        ax.set_yticks(np.arange(0, self.rows + 1, 1), minor=False)
        ax.grid(True, which='major', color='k', linewidth=0.4, alpha=0.15)
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')

        # Hide MAJOR tick labels (keep only grid lines)
        ax.tick_params(axis='both', which='major', labelbottom=False, labelleft=False, length=0)

        # Centered labels as MINOR ticks at cell centers
        x_centers = np.arange(0.5, self.cols, 0.5)     # 0.5, 1.5, 2.5, ...
        y_centers = np.arange(0.5, self.rows, 0.5)

        ax.set_xticks(x_centers, minor=True)
        ax.set_yticks(y_centers, minor=True)

        # Build pattern: blank - 0 - blank - 1 - blank - 2 - ...
        def interleaved_labels(n):
            labels=[]
            for k in range(n):
                if k % 2 == 0:
                    labels.append(str(k // 2))  # numeric label
                else:
                     labels.append("")   # blank space

            return labels

        x_minor_labels = interleaved_labels(len(x_centers))
        y_minor_labels = interleaved_labels(len(y_centers)) 

        ax.set_xticklabels(x_minor_labels, minor=True)
        ax.set_yticklabels(y_minor_labels, minor=True)
    
        # Start marker
        sr, sc = self.settings.start
        ax.scatter(sc + 0.5, self.rows - sr - 0.5, s=180, marker='D',
                   facecolors='#4fc3f7', edgecolors='black', label='Start', zorder=5)
    
        # Agent marker
        if show_agent:
            ar, ac = self.position
            ax.scatter(ac + 0.5, self.rows - ar - 0.5, s=220, marker='o',
                       facecolors='#1565c0', edgecolors='black', label='Agent', zorder=6)
            
        # Goal marker
        gr, gc = self.settings.goal
        ax.scatter(gc + 0.5, self.rows - gr - 0.5, s=180, marker='*',
                       facecolors="#66bb6a", edgecolors='black', label="Goal", zorder=6)
    
        # Path (centered)
        if path is not None and len(path) > 1:
            rows, cols = zip(*path)
            xs = np.array(cols, dtype=float) + 0.5
            ys = self.rows - (np.array(rows, dtype=float) + 0.5)
            ax.plot(xs, ys, linewidth=3.5, color='#60b37a',
                    label='Learned Path', zorder=4)
    
        ax.set_title(title, fontsize=20, pad=10)
        ax.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0., labelspacing=1, loc='upper left', frameon=True, fontsize=15)
        plt.tight_layout()
        plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
            