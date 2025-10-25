# Reinforcement Learning (RL)
Trains an agent to act in an environment by balancing exploration and exploitation to maximize cumulative reward over time.


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
```

### 4. Register Jupyter Kernel

This step makes your virtual environment visible inside JupyterLab:

```bash
python -m ipykernel install --user --name gridrl --display-name "Python (gridrl)"
```

### 5. Launch JupyterLab
```bash
jupyter lab
```