# Environment
import copy
from collections import deque, namedtuple
from os import mkdir

import numpy as np
from tqdm.auto import tqdm
import time
import os
from pathlib import Path
from flappy_bird_env import *
# Torch
import torch
from torch import nn

### ---------------- SETUP ---------------- ###
# -> TRAINING/TESTING SETTINGS
USE_EXISTING_MODEL = True
MODEL_NAME = "DQN_64_V_10"
TRAINING = True
# -> HYPERPARAMETER
LR = 0.00005
GAMMA = 0.99
EPSILON = 0.1
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9999
EPISODES = 5000
TEST_EPISODES = 30
DIFFICULTY = 20
### -------- PATH SETTINGS -------- ###
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


class DQN(nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_units: int = 64):
        super(DQN, self).__init__()
        self.hidden_units = hidden_units
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
    def __init__(self, epsilon, gamma, epsilon_decay, epsilon_min, use_existing_model=False, model_name=None ):
        # networks - use existing or create new
        if use_existing_model:
            online_net = self.load_model(model_name)
        else:
            online_net = DQN(input_features=10, output_features=2)

        self.online_net = online_net
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
        sprint_score = 0
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
            sprint_score += episode_score
            self.train_reward += episode_reward
            try:
                avg_score = self.train_score / episode
                avg_reward = self.train_reward / episode
                avg_steps = step_count / episode
                avg_sprint_score = sprint_score / 100
            except ZeroDivisionError:
                avg_score = episode_score
                avg_reward = episode_reward
                avg_steps = step_count / 1
                avg_sprint_score = 0
            # Print out every 100 episode
            if episode % 100 == 0 and episode != 0:
                print(f"\nEpisode: {episode} | avg sprint score: {avg_sprint_score:.2f} | avg reward: {avg_reward:.3f} | avg steps: {avg_steps:.3f}  | Epsilon: {self.epsilon:.4f}")
                sprint_score = 0
        print("Training Complete!")
        self.save_model()

    def test(self, episodes: int):
        total_score = 0
        env = self.env
        for episode in tqdm(range(episodes)):
            state, reward, is_done, episode_score = env.reset()
            state = state.flatten()
            while not is_done:
                action = torch.argmax(self.online_net(state)).item()
                next_state, reward, is_done, episode_score = env.step(action)
                next_state = next_state.flatten()
                state = next_state
            if episode == (episodes -10):
                avg_score = total_score / episodes
                print(f"test result (avg score): {avg_score:.3f}")
                env = self.test_env
            total_score += episode_score

    ### SAVING AND LOADING MODEL ###
    def save_model(self):
        name = f"{self.online_net.__class__.__name__}_{self.online_net.hidden_units}_V_{len(os.listdir(MODEL_DIR))}"
        torch.save(self.online_net, fr"{MODEL_DIR}\{name}.pth")
        print(f"saved model under name: {name}")
        return fr"{MODEL_DIR}\{name}.pth"

    @staticmethod
    def load_model(model_name: str):
        base_path = MODEL_DIR
        model_name = fr"{base_path}\{model_name}.pth"
        try:
            model = torch.load(model_name, weights_only=False)
            print(f"model {model_name}.pth was loaded")
            return model
        except FileNotFoundError:
            print(f"{model_name} was not found!")
            raise FileNotFoundError


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
    agent = Agent(EPSILON, GAMMA, EPSILON_DECAY, EPSILON_MIN, USE_EXISTING_MODEL, MODEL_NAME)
    if TRAINING:
        agent.train(EPISODES)
    agent.test(TEST_EPISODES)



