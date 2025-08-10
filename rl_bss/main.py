import os
import logging
from rl_bss import config
from rl_bss.env.bss_env import BeeSwarmEnv
from rl_bss.agent.dqn_agent import DQNAgent

def setup_logging():
    """Sets up the logging configuration."""
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    logging.basicConfig(filename=config.LOG_FILE, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main training loop."""
    setup_logging()
    
    env = BeeSwarmEnv()
    agent = DQNAgent(config.OBSERVATION_SPACE_SHAPE, config.ACTION_SPACE_SIZE)

    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    # Load existing model if available
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Loading model from {config.MODEL_SAVE_PATH}")
        logging.info(f"Loading model from {config.MODEL_SAVE_PATH}")
        agent.load_model(config.MODEL_SAVE_PATH)

    for i_episode in range(config.NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        if i_episode % config.TARGET_UPDATE == 0:
            agent.update_target_net()

        if i_episode % 10 == 0:
            agent.save_model(config.MODEL_SAVE_PATH)
            print(f"Episode {i_episode}, Total Reward: {total_reward}, Model Saved")
            logging.info(f"Episode {i_episode}, Total Reward: {total_reward}, Model Saved")

    env.close()
    print("Training finished.")
    logging.info("Training finished.")

if __name__ == "__main__":
    main()
