![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# Reinforcement Learning in GridWorld
A comparative study of Q-Learning, SARSA, and Dyna-Q with planning and robustness analysis.

---

## Project Overview
This project implements and analyzes core **tabular reinforcement learning algorithms** inside a custom **stochastic GridWorld environment**.

* A full **GridWorld environment** with walls, pits, wind (stochasticity), rewards, indexing logic, and deterministic seeding.

* Algorithms Implemented:

    * **Q-Learning** (off-policy TD control)
    * **SARSA(0)** (on-policy TD control)
    * **Dyna-Q** (model-based RL + planning updates)

* Experiments on:

    * **Planning sweep (varying Dyna-Q planning steps $K$)**
    * **Robustness across environment layouts & random seeds**
    * **Final Policy evaluation**

* A complete unit test suite for **all algorithms, GridWorld,** and **utils**.

* Well-structured Jupyter notebooks for analysis.

* Reusable experiment utilities (seed experiments, plotting, train-with-logs functions).


**Goal:** Understand **sample efficiency, stability**, and **policy robustness** across model-free and model-based RL methods in controlled environments.

---

## Project Structure
```
reinforcement_learning/
│
├── src/rl_capstone/
│   ├── gridworld.py        # Environment implementation
│   ├── rl_algorithms.py    # Q-Learning, SARSA, Dyna-Q + logging variants
│   ├── utils.py            # Action selection, schedules, evaluation, plotting
│   └── __init__.py
│
├── notebooks/
│   ├── 00_RL.ipynb
│   ├── 01_q_learning.ipynb
│   ├── 02_sarsa.ipynb
│   ├── 03_dyna_q.ipynb
│   ├── 04_comparison_models.ipynb
│   ├── 05_k_sweep.ipynb
│   ├── 06_robustness.ipynb
│   └── 07_results.ipynb
│
├── tests/
│   ├── test_gridworld.py
│   ├── test_rl_algorithms.py
│   └── test_utils.py
│
├── data/
│   ├── q_tables/           # Saved NumPy Q-tables
│   └── robustness/         # npz files for seed stability experiments
│
├── reports/                
│   ├── figs/               # Stores the result's graphs
│   ├── Report.tex/         # LaTex file
│   └── Report.pdf/         # PDF or markdown summary
│
├── requirements.txt
└── README.md

```

---

## Environment Setup

This project uses a local Python virtual environment (`.venv`) and Jupyter notebook for analysis.
The virtual environment is **not** committed to Github, so you must create it after cloning the repository.

### 1. Clone the Repository
```bash
git clone https://github.com/fcampoverdeg/reinforcement_learning.git
cd reinforcement_learning
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # on macOS/Linux
# .venv\Scripts\activate    # On Windows PowerShell
```
You should now see (`.venv`) at the beggining of the shell prompt

### 3. Install Dependencies
```bash
pip install -r requirements.txt

# In case 'requirements.txt' is not available
pip install numpy scipy matplotlib jupyterlab ipykernel pandas tqdm \
            black ruff pytest pytest-cov mypy gymnasium pygame

# Install packages in editable mode
pip install -e
```

### 4. Run Unit Tests (Recommended)
Ensures GridWorld + all algorithms work correctly.

```bash
pytest -q
```

### 5. Register Jupyter Kernel

This step makes your virtual environment visible inside JupyterLab:

```bash
python -m ipykernel install --user --name gridrl --display-name "Python (gridrl)"
```

### 6. Launch JupyterLab
```bash
jupyter lab
```

---

## Research Questions Addressed

1. How do Dyna-Q planning steps $K$ affect sample efficiency?

2. How sensitive are Q-learning, SARSA, and Dyna-Q to ε-greedy schedules?

3. How robust are policies across seeds, layout changes, and wind noise?

4. Does model-based planning consistently improve stability and convergence?

All findings are documented in the **Results notebook**

---

## Author
Felipe Campoverde  
Virginia Tech - RL Capstone Research Project