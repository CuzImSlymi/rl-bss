import time
import numpy as np
import subprocess
import psutil
import webbrowser

from rl_bss import config
from rl_bss.utils import keyboard
from rl_bss.utils.position import Position

# Placeholder for the user-provided function
def GetHoneyFunction():
    """
    This is a placeholder function. The user will implement this
    to get the current honey amount from the game.
    """
    # For testing, we can simulate honey increase
    return np.random.randint(1000, 2000)

class BeeSwarmEnv:
    def __init__(self):
        self.action_space = np.arange(config.ACTION_SPACE_SIZE)
        self.observation_space = np.zeros(config.OBSERVATION_SPACE_SHAPE)
        self.position = Position()
        self.start_time = 0
        self.stuck_counter = 0
        self.start_honey = 0
        self.previous_honey = 0

    def _close_roblox(self):
        """Finds and terminates the Roblox process."""
        print(f"Attempting to close {config.ROBLOX_PROCESS_NAME}...")
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] == config.ROBLOX_PROCESS_NAME:
                try:
                    p = psutil.Process(proc.info['pid'])
                    p.terminate()
                    p.wait(timeout=5)
                    print(f"Successfully terminated {config.ROBLOX_PROCESS_NAME}.")
                except psutil.NoSuchProcess:
                    print(f"{config.ROBLOX_PROCESS_NAME} was already closed.")
                except psutil.TimeoutExpired:
                    print(f"Termination timed out. Forcing kill on {config.ROBLOX_PROCESS_NAME}.")
                    p.kill()
                except Exception as e:
                    print(f"An error occurred while closing Roblox: {e}")
                return # Assume one instance
        print(f"{config.ROBLOX_PROCESS_NAME} not found running.")


    def _launch_roblox(self):
        """Launches Bee Swarm Simulator via its deeplink."""
        print(f"Launching Roblox via deeplink: {config.DEEPLINK_URL}")
        try:
            webbrowser.open(config.DEEPLINK_URL)
        except Exception as e:
            print(f"Failed to launch Roblox deeplink: {e}")


    def reset(self):
        """
        Resets the environment for a new episode.
        This involves closing and restarting Roblox.
        """
        print("Resetting environment...")
        self._close_roblox()
        time.sleep(5) # Grace period before relaunch
        self._launch_roblox()
        
        print(f"Waiting {config.GAME_LOAD_TIME} seconds for the game to load...")
        time.sleep(config.GAME_LOAD_TIME)

        self.position.reset()
        self.start_time = time.time()
        self.stuck_counter = 0
        
        # It's crucial that GetHoneyFunction works correctly after reset
        self.start_honey = GetHoneyFunction()
        self.previous_honey = self.start_honey
        print("Environment reset complete.")
        
        return self._get_observation()

    def step(self, action):
        """
        Performs an action in the environment.
        """
        key = config.ACTION_MAP.get(action)
        if key and key != 'idle':
            keyboard.press_and_release(key)
            if key in ['w', 'a', 's', 'd']:
                self.position.move(key)

        current_honey = GetHoneyFunction()
        
        # Calculate reward
        reward = self._calculate_reward(current_honey)
        
        # Update state
        self.previous_honey = current_honey
        
        # Check for episode end
        done = self._is_done()
        
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """
        Gets the current state of the environment.
        """
        pos_x, pos_y = self.position.get_coords()
        time_elapsed = time.time() - self.start_time
        
        return np.array([
            self.previous_honey,
            pos_x,
            pos_y,
            time_elapsed,
            self.stuck_counter
        ])

    def _calculate_reward(self, current_honey):
        """
        Calculates the reward for the current step.
        """
        reward = 0
        
        # Honey progress
        honey_diff = current_honey - self.previous_honey
        reward += honey_diff * config.REWARD_SCALING

        # Penalties
        if honey_diff <= 0:
            self.stuck_counter += 1
            if self.stuck_counter > config.STUCK_DETECTION_THRESHOLD:
                reward += config.PENALTY_STUCK
        else:
            self.stuck_counter = 0

        if current_honey < self.previous_honey:
            reward += config.PENALTY_DAMAGE

        # Time penalty
        reward += config.PENALTY_TIME
        
        return reward

    def _is_done(self):
        """
        Checks if the episode is finished.
        """
        time_elapsed = time.time() - self.start_time
        return time_elapsed > config.EPISODE_DURATION

    def close(self):
        """
        Closes the environment.
        """
        print("Closing environment.")
        self._close_roblox()