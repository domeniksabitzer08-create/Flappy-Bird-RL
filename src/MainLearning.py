# Environment
import copy
from collections import deque, namedtuple

import numpy as np
from tqdm.auto import tqdm
import time
from flappy_bird_env import *
# Torch
import torch
from torch import nn

### -------- SETUP -------- ###
LR = 0.001
GAMMA = 0.99
EPSILON = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995
EPISODES = 3000
DIFFICULTY = 20

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
    def __init__(self, epsilon, gamma, epsilon_decay, epsilon_min):
        # networks
        self.online_net = DQN(input_features=10, output_features=2)
        self.target_net = copy.deepcopy(self.online_net)
        self.network_sync_rate = 1000
        # Hyperparameter
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        # environment
        self.env = FlappyBirdEnv(difficulty=DIFFICULTY, render=False)
        self.test_env = FlappyBirdEnv(difficulty=DIFFICULTY, render=True)
        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LR)
        self.loss_fn = torch.nn.MSELoss()
        # data management
        self.train_score = 0
        self.train_reward = 0
        self.train_steps = 0

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
        step_count = 0
        for episode in tqdm(range(episodes)):
            ### Data ###
            episode_reward = 0
            state, reward, is_done, episode_score = self.env.reset()
            state = state.flatten() # flatten the state and add extra batch dim
            while not is_done:
                step_count += 1
                # choose action
                action = self.choose_action(state)
                # execute action
                next_state, reward, is_done, episode_score = self.env.step(action)
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
                        q_target = reward_tensor + self.gamma * self.target_net(next_state_tensor).max(dim=1,keepdim=True)[0].squeeze(dim=1) * (1 - is_done_tensor)
                    # training procedure
                    loss = self.loss_fn(q_values, q_target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    ### UPDATE DATA ###
                    episode_reward += reward

                    # sync networks
                    if step_count % self.network_sync_rate == 0:
                        self.sync_target()

            # Epsilon decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            ### ---- DATA ---- ###
            # save the score and reward
            self.train_score += episode_score
            self.train_reward += episode_reward
            try:
                avg_score = self.train_score / episode
                avg_reward = self.train_reward / episode
                avg_steps = step_count / episode
            except ZeroDivisionError:
                avg_score = episode_score
                avg_reward = episode_reward
                avg_steps = step_count / 1
            # Print out every 100 episode
            if episode % 100 == 0:
                print(f"Episode: {episode} | avg_score: {avg_score:.2f} | total_score: {self.train_score}  | avg reward: {avg_reward} | avg steps: {avg_steps}")
    def test(self, episodes: int):
        for episodes in tqdm(range(episodes)):
            state, reward, is_done, score = self.test_env.reset()
            state = state.flatten()
            while not is_done:
                action = torch.argmax(self.online_net(state)).item()
                next_state, reward, is_done, score = self.test_env.step(action)
                next_state = next_state.flatten()
                state = next_state



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

def messure_env_time():
    env = FlappyBirdEnv(difficulty=4, render=False)
    state, _, _, _ = env.reset()
    start = time.time()
    for _ in range(100):
        action = env.sample()
        env.step(action)
    print(time.time() - start)


if __name__ == '__main__':
    agent = Agent(EPSILON, GAMMA, EPSILON_DECAY, EPSILON_MIN)
    agent.train(EPISODES)
    agent.test(100)



