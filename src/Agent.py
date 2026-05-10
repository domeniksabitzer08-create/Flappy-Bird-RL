# Environment
from flappy_bird_env import *
# Torch
import torch
from torch import nn

### -------- SETUP -------- ###
LR = 0.01
GAMMA = 0.9
GAMMA_DECAY = 0.99


class Model(nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_units: int = 32):
        super(Model, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features),
        )

    def forward(self, x):
        x = self.layer_stack(x)
        return x

def debug_model_shape():
    dummy = torch.rand((5,2))
    print(f"dummy: {dummy.shape}")
    dummy = dummy.flatten().unsqueeze(0)
    print(f"flatten dummy: {dummy.shape}")
    model = Model(input_features=10, output_features=3)
    y_logit = model(dummy)
    print(f"logit: {y_logit}")
    y_pred = torch.argmax(y_logit, dim=1)
    print(f"action: {y_pred}")

if __name__ == '__main__':
    debug_model_shape()



