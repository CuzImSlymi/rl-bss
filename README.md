# RL-BSS: Reinforcement Learning for Bee Swarm Simulator

This project is an MVP of a Reinforcement Learning AI that learns to play Bee Swarm Simulator. The AI controls the character's movements to collect honey, which is the primary reward signal.

## Project Structure

- `rl_bss/`: Main project folder.
  - `agent/`: Contains the RL agent logic (e.g., DQN).
  - `env/`: The Bee Swarm Simulator environment wrapper.
  - `utils/`: Utility functions for keyboard control and position tracking.
  - `config.py`: Configuration file for hyperparameters and settings.
  - `main.py`: The main training loop.
- `models/`: Directory to save trained models.
- `logs/`: Directory for training logs.
- `requirements.txt`: Python dependencies.

## Getting Started

### Prerequisites

- Python 3.7+ (3.10 stable)
- An installation of Roblox

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CuzImSlymi/rl-bss
   cd rl-bss
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### How to Run

1. **Open Bee Swarm Simulator:** Make sure the game is running and the active window, fulllscreen.

2. **Run the training script:**
   ```bash
   python rl_bss/main.py
   ```

The AI will then start interacting with the game, and you can monitor its progress through the console output and the logs in the `logs/` directory.

## Customization

- **RL Algorithm:** You can change the RL algorithm by modifying the `RL_ALGORITHM` variable in `config.py` and implementing the corresponding agent in the `agent/` directory.
- **Hyperparameters:** All hyperparameters for the agent and the environment can be tuned in `config.py`.
- **Reward System:** The reward function can be customized in `env/bss_env.py` to encourage different behaviors.

## Important Notes

- **`GetHoneyFunction()`:** The `GetHoneyFunction()` in `env/bss_env.py` is a placeholder.
- **Reset Mechanism:** The `reset()` function in `env/bss_env.py` uses a placeholder for restarting the game.

## Creator

- **slymi**

