# Environment
import copy
from collections import deque, namedtuple
from operator import truediv
from os import mkdir

import numpy as np
from torch.export.pt2_archive.constants import MODELS_DIR
from torchgen.api.cpp import return_names
from tqdm.auto import tqdm
import time
import os
import dill
from pathlib import Path
from flappy_bird_env import *
# Torch
import torch
from torch import nn
# Tensorboard
from torch.utils.tensorboard import SummaryWriter
def naming():
    if TRAINING:
        if USE_EXISTING_MODEL:
            v = MODEL_VERSION
            retrain = "RETRAINED"
### ---------------- SETUP ---------------- ###
# -> TRAINING/TESTING SETTINGS
USE_EXISTING_MODEL = True
MODEL_VERSION = 32
MODEL_NAME = f"DQN_64_V_{MODEL_VERSION}"
TRAINING = True
# -> HYPERPARAMETER
LR = 0.001
GAMMA = 0.99
EPSILON = 0.01
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99907
EPISODES = 1000
TEST_EPISODES = 110
DIFFICULTY = 25
# -> NAMING
if not USE_EXISTING_MODEL:
    MODEL_VERSION = len(os.listdir(MODELS_DIR))
### -------- PATH SETTINGS -------- ###
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
BUFFER_DIR = BASE_DIR / "buffers"
RUNS_DIR = BASE_DIR / "runs"
EXP_PATH = BASE_DIR / fr"experiments"



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

# Assigning Experience outside the class so pickle can access it
EXPERIENCE = namedtuple("Experience", ["state", "action", "reward", "next_state", "is_done"])
class ReplayBuffer:
    def __init__(self, capacity: int, batch_size: int):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, is_done):
        experience = EXPERIENCE(state, action, reward, next_state, is_done)
        self.memory.append(experience)

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        return batch

    def can_provide_sample(self):
        return len(self.memory) >= self.batch_size

    def __len__(self):
        return len(self.memory)

def save_buffer(buffer: ReplayBuffer):
    path = BUFFER_DIR / f"_V_{len(os.listdir(MODEL_DIR))}_buffer.pkl"
    with open(path, "wb") as f:
        dill.dump(buffer.memory, f)
    print("saved buffer")


class Agent:
    def __init__(self, epsilon, gamma, epsilon_decay, epsilon_min, use_existing_model=False, model_name=None, model_version=None):
        # networks and buffer - use existing or create new
        self.buffer = ReplayBuffer(10000, 32)
        if use_existing_model:
            online_net = self.load_model(model_name)
            self.buffer.memory = self.load_buffer(model_version=model_version)
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
        self.highest_score = 0

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
        buffer = self.buffer
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


            # Print out every 1000 episode
            if episode % 100 == 0 and episode != 0:
                print(f"\nEpisode: {episode} | avg sprint score: {avg_sprint_score:.2f} | avg reward: {avg_reward:.3f} | avg steps: {avg_steps:.3f}  | Epsilon: {self.epsilon:.4f}")
                sprint_score = 0
                # save the model if it is the best performing one
                if self.highest_score < avg_sprint_score:
                    self.highest_score = avg_sprint_score
                    self.save_model()
                    save_buffer(buffer)

        print("Training Complete!")
        self.buffer = buffer
        self.save_model()
        save_buffer(buffer)

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
                avg_score = total_score / episode
                print(f"test result (avg score): {avg_score:.3f}")
                env = self.test_env
            total_score += episode_score

    ### SAVING AND LOADING MODEL AND BUFFER ###
    def save_state(self):
        pass


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
            print(f"model {model_name} was loaded")
            return model
        except FileNotFoundError:
            print(f"{model_name} was not found!")
            raise FileNotFoundError

    @staticmethod
    def load_buffer(model_version: int):
        path = BUFFER_DIR / f"_V_{model_version}_buffer.pkl"
        with open(path, "rb") as f:
             x = dill.load(f)
        print("loaded buffer")
        return x



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
    agent = Agent(EPSILON, GAMMA, EPSILON_DECAY, EPSILON_MIN, USE_EXISTING_MODEL, MODEL_NAME, MODEL_VERSION)
    if TRAINING:
        agent.train(EPISODES)
    agent.test(TEST_EPISODES)



