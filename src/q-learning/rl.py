"""
Here I am doing a small example of RL from GeeksforGeeks.org, just to understand how to implement a basic RL system.
While getting used to Numpy and MatplotLib, machine learning terminology, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
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

rewards_all_episodes = []

for episode in range(num_episodes):
    state = start
    total_rewards = 0
    done = False
    
    while not done:
        action_index = choose_action(state)
        action = actions[action_index]

        next_state = (state[0] + action[0], state[1] + action[1])

        if not is_valid(next_state):
            reward = reward_fire
            done = True
        elif next_state == goal:
            reward = reward_goal
            done = True
        else:
            reward = reward_step

        old_value = Q[state][action_index]
        next_max = np.max(Q[next_state]) if is_valid(next_state) else 0

        Q[state][action_index] = old_value + alpha * \
            (reward + gamma * next_max - old_value)
        
        state = next_state
        total_rewards = reward

    # global epsilon
    epsilon = max(0.01, epsilon * 0.995)
    rewards_all_episodes.append(total_rewards)

'''
Step 5: Extract the Optimal path after Training

-> This function follows the highest Q-values at each state to extract the best path.
-> It stops when the goal is reached or no valid next moves are available
-> The visited set prevents cycles.

'''

def get_optimal_path (Q, start, goal, actions, maze, max_steps=200):
    path = [start]
    state = start
    visited = set()

    for _ in range(max_steps):
        if state == goal:
            break
        visited.add(state)

        best_action = None
        best_value = -float('inf')

        for idx, move in enumerate(actions):
            next_state = (state[0] + move[0], state[1] + move[1])

            if (0 <= next_state[0] < maze.shape[0] and
                0 <= next_state[1] < maze.shape[1] and
                maze[next_state] == 0 and
                    next_state not in visited):

                if Q[state][idx] > best_value:
                    best_value = Q[state][idx]
                    best_action = idx

        if best_action is None:
            break

        move = actions[best_action]
        state = (state[0] + move[0], state[1] + move[1])
        path.append(state)

    return path

optimal_path = get_optimal_path(Q, start, goal, actions, maze)
                
'''
Step 6: Visualize the Maze, Robot Path, Start and Goal

-> The maze and path are visualized using a calmig green color palette.
-> The start and goal positions are visually highlighted.
-> The learned path is drawn clearly to demonstrate the agent's solution.

'''

def plot_maze_with_path(path):
    cmap = ListedColormap(['#eef8ea', '#a8c79c'])

    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap=cmap)

    plt.scatter(start[1], start[0], marker='o', color='#81c784', edgecolors='black',
                s=200, label='Start (Robot)', zorder=5)
    plt.scatter(goal[1], goal[1], marker='*', color='#388e3c', edgecolors='black',
                s=300, label='Goal (Diamond)', zorder=5)
    
    rows, cols = zip(*path)
    plt.plot(cols, rows, color='#60b37a', linewidth=4,
             label='Learned Path', zorder=4)
    
    plt.title('Reinforcement Learning: Robot Maze Navigation')
    plt.gca().invert_yaxis()
    plt.xticks(range(maze.shape[1]))
    plt.yticks(range(maze.shape[0]))
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_maze_with_path(optimal_path)

    