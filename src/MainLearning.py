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
        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LR)
        self.loss_fn = torch.nn.MSELoss()

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
            state = state.flatten() # flatten the state and add extra batch dim
            while not is_done:
                # choose action
                action = self.choose_action(state)
                # execute action
                next_state, reward, is_done, score = self.env.step(action)
                next_state = next_state.flatten()
                # add new experience to buffer
                buffer.add(state, action, reward, next_state, is_done)
                state = next_state
                # if enough samples were collected, the optimization can be performed
                if buffer.can_provide_sample():
                    # get a batch of experiences
                    batch = buffer.sample()
                    # put each experience part in a single Tensor
                    state_tensor = torch.stack([e.state for e in batch])
                    action_tensor = torch.tensor([e.action for e in batch],dtype=torch.long)
                    reward_tensor = torch.tensor([e.reward for e in batch], dtype=torch.float32)
                    next_state_tensor = torch.stack([e.next_state for e in batch])
                    is_done_tensor = torch.tensor([e.is_done for e in batch], dtype=torch.float32)
                    # add an extra dimension to fit gathering
                    action_tensor = action_tensor.unsqueeze(1)
                    # calculate Q-Values
                    q_values = self.online_net(state_tensor).squeeze(dim=1) # get rid of not necessary dim
                    q_values = q_values.gather(1, action_tensor).squeeze(dim=1)
                    # Bellerman equation
                    with torch.no_grad():
                        q_target = reward_tensor + GAMMA * self.target_net(next_state_tensor).max(dim=1,keepdim=True)[0].squeeze(dim=1) * (1 - is_done_tensor)
                    # training procedure
                    loss = self.loss_fn(q_values, q_target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


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
    agent = Agent(0.99, GAMMA)
    agent.train(10)



