import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from rl_bss import config

class QNetwork(nn.Module):
    def __init__(self, input_shape, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        
        self.policy_net = QNetwork(state_shape, action_size)
        self.target_net = QNetwork(state_shape, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.memory = ReplayBuffer(config.BUFFER_SIZE)
        self.steps_done = 0

    def select_action(self, state):
        eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * \
                        np.exp(-1. * self.steps_done / config.EPS_DECAY)
        self.steps_done += 1
        
        if random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                return self.policy_net(state).max(1)[1].view(1, 1).item()
        else:
            return random.randrange(self.action_size)

    def learn(self):
        if len(self.memory) < config.BATCH_SIZE:
            return

        transitions = self.memory.sample(config.BATCH_SIZE)
        batch = list(zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch[0]))
        action_batch = torch.LongTensor(np.array(batch[1])).unsqueeze(1)
        reward_batch = torch.FloatTensor(np.array(batch[2]))
        next_state_batch = torch.FloatTensor(np.array(batch[3]))
        done_batch = torch.BoolTensor(np.array(batch[4]))

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (config.GAMMA * next_q_values * ~done_batch)

        loss = nn.functional.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
