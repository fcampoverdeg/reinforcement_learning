"""
Here I am doing a small example of RL from GeeksforGeeks.org, just to understand how to implement a basic RL system.
While getting used to Numpy and MatplotLib, machine learning terminology, etc.
"""

import numpy as np
import matplotlib as plt
from matplotlib.colors import ListedColormap

'''
Step 1: Import libraries and Define Maze, Start and Goal

The maze is represented as a 2D NumPy array.
0 values are safe paths, 1s are obstacles the agent must avoid
Start and goal define the positions where the agent begins and where it aims to reach
'''
maze = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
])

start = (0, 0)
goal = (9, 9)

'''
Step 2: Define RL Parameters and Initialize Q-Table

num_episodes   ->   Number of times the agent will attempt to navigate the maze.
alpha          ->   Learning rate that controls how much new information overrides old information.
gamma          ->   Discount factor giving more weight to immediate rewards.
epsilon        ->   Probability of exploration vs exploitation; starts higher to explore more.
actions(moves) ->   left, right, up, down.
Q              ->   Is the Q-Table initialized to zero; it stores expected rewards for each state-action pair.

Rewards are set to penalize hitting obstacles, reward reaching the goal and slightly penalize each
step to find shortest paths.
'''

num_episodes = 5000 # Max episodes it can reach before terminating
alpha = 0.1
gamma = 0.9
epsilon = 0.5

reward_fire = -10   # Falls into a 1
reward_goal = 50    # Reaches the goal (9, 9)
reward_step = -1    # Each step reward -1

#            left    right     up     down
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

Q = np.zeros(maze.shape + (len(actions),))  # Q-table, shape (rows, cols, actions)

'''
Step 3: Helper Function for Maze Validity and Action Selection

is_valid       ->   Ensures the agent can only move inside the maze and avoids obstacles.
choose_action  ->   Implements exploration (random action) vs exploitation (best learned action) strategy
'''

def is_valid(pos):
    
    r, c = pos                          # Initialize rows and cols

    if r < 0 or r >= maze.shape[0]:     # row index has to be within [0, number_of_rows - 1]
        return False
    if c < 0 or c >= maze.shape[1]:     # col index has to be within [0, number_of_cols - 1]
        return False
    if maze[r, c] == 1:                 # ensures the cell is not an obstacle ( 1 = wall )
        return False
    return True

# With probability ε (epsilon), explore; otherwise, exploit what you already know"
def choose_action(state):
    # np.random.random() -> returns a float in [0, 1]
    # epsilon is a probability ( 0.5 -> 50% chance)
    if np.random.random() < epsilon:
        return np.random.randint(len(actions))  # Agent takes a random actions, ignoring it's learned Q-value
    else:
        return np.argmax(Q[state])  # Otherwise; Agent will select the best Q-value at the current state
    
'''
Step 4: Train the Agent with Q-Learning Algorithm

Runs multiple episodes for the agent to learn.
During each episode, the agent selects actions and updates its Q-Table using the Q-Learning formula:
    
    Q(s,a)=Q(s,a)+α[r+γmaxa′​Q(s′,a′)−Q(s,a)]

total_rewards -> tracks cumulative rewards per episode
epsilon -> decays gradually to reduce randomness over time.
'''