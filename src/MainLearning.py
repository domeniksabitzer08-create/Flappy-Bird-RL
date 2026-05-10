# Environment
import copy
from collections import deque, namedtuple

import numpy as np
from tqdm.auto import tqdm

from flappy_bird_env import *
# Torch
import torch
from torch import nn

### -------- SETUP -------- ###
LR = 0.01
GAMMA = 0.9
GAMMA_DECAY = 0.99


class DQN(nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_units: int = 64):
        super(DQN, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features),
        )

    def forward(self, x):
        x = self.layer_stack(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity: int, batch_size: int):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "is_done"])

    def add(self, state, action, reward, next_state, is_done):
        experience = self.experience(state, action, reward, next_state, is_done)
        self.memory.append(experience)

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        return batch

    def can_provide_sample(self):
        return len(self.memory) >= self.batch_size

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, epsilon, gamma):
        # networks
        self.online_net = DQN(input_features=10, output_features=2)
        self.target_net = copy.deepcopy(self.online_net)
        # Hyperparameter
        self.epsilon = epsilon
        self.gamma = gamma
        # environment
        self.env = FlappyBirdEnv(difficulty=4, render=False)

    def choose_action(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.sample()
        else:
            with torch.no_grad():
                y_logit = self.online_net(state)
                return torch.argmax(y_logit).item()

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train(self, episodes: int):
        buffer = ReplayBuffer(10000, 32)
        for episode in tqdm(range(episodes)):
            state, reward, is_done, score = self.env.reset()
            while not is_done:
                # choose action
                action = self.choose_action(state)
                # execute action
                next_state, reward, is_done, score = self.env.step(action)
                # add new experience to buffer
                buffer.add(state, action, reward, next_state, is_done)

                # if enough samples were collected, the optimization can be performed
                if buffer.can_provide_sample():
                    batch = buffer.sample()
                    q_values = self.online_net(batch)
                    with torch.no_grad():
                        q_values_next = self.target_net(next_state)
                    # Use Bellerman equation










def debug_model_shape():
    dummy = torch.rand((5,2))
    print(f"dummy: {dummy.shape}")
    dummy = dummy.flatten().unsqueeze(0)
    print(f"flatten dummy: {dummy.shape}")
    model = DQN(input_features=10, output_features=2)
    y_logit = model(dummy)
    print(f"logit: {y_logit}")
    y_pred = torch.argmax(y_logit, dim=1)
    print(f"action: {y_pred}")




if __name__ == '__main__':
    debug_model_shape()



