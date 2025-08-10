# Technical Specifications
CREATOR = "slymi"
VERSION = "0.1.0"

# RL Algorithm Configuration
RL_ALGORITHM = "DQN"  # Easily changeable (e.g., "PPO", "A2C")

# Bee Swarm Simulator Environment Configuration
GAME_WINDOW_TITLE = "Roblox"
ROBLOX_PROCESS_NAME = "RobloxPlayerBeta.exe"
DEEPLINK_URL = "roblox://placeId=1537690962"

# Action Space
# w, a, s, d, space, idle
ACTION_SPACE_SIZE = 6
ACTION_MAP = {
    0: 'w',
    1: 'a',
    2: 's',
    3: 'd',
    4: 'space',
    5: 'idle'
}
KEY_PRESS_DURATION = 0.1  # seconds

# Observation Space
# [honey_count, pos_x, pos_y, time_elapsed, stuck_counter]
OBSERVATION_SPACE_SHAPE = (5,)

# Reward System
REWARD_SCALING = 1.0
PENALTY_STUCK = -1.0
PENALTY_DAMAGE = -10.0
PENALTY_WASTE_ABILITY = -0.5
PENALTY_TIME = -0.1

# Episode Management
EPISODE_DURATION = 300  # seconds (5 minutes)
STUCK_DETECTION_THRESHOLD = 10  # seconds with no honey progress
GAME_LOAD_TIME = 20 # seconds to wait for Roblox to load

# DQN Agent Hyperparameters
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
LEARNING_RATE = 0.001

# Training Configuration
NUM_EPISODES = 1000
MODEL_SAVE_PATH = "models/rl_bss_model.pth"
LOG_FILE = "logs/training.log"