import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

class policyNN(nn.Module):
    """
    
    Policy Neural Network that returns the probabilities of actions to be taken
    given the current state input of the network, as well as the value of the
    current board state.

    Input is in the form of 8x8x119 and output will be in the form of 8x8x73

    """
    def __init__(self, config):
        super(policyNN, self).__init__()

        inc = config.get("in_channels", 119)

        self.conv1 = nn.Conv2d(inc, 256, kernel_size=3, padding=1)

        self.conv_p1 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv_p2 = nn.Conv2d(256, 73, kernel_size=1)

        self.conv_v1 = nn.Conv2d(256, 1, kernel_size=1)
        self.fc_v1 = nn.Linear(64, 256)
        self.fc_v2 = nn.Linear(256, 1)

    def policy_head(self, x: tensor) -> tensor:

        x = self.conv_p1(x)
        x = nn.ReLU()(x)
        x = self.conv_p2(x)
        x = torch.flatten(x)

        return x

    def value_head(self, x: tensor) -> tensor:

        x = self.conv_v1(x)
        x = nn.ReLU()(x)
        x = torch.flatten(x)
        x = self.fc_v1(x)
        x = self.fc_v2(x)
        x = torch.tanh()(x)

        return x   

    def forward(self, s: tensor) -> tuple:
        
        x = self.conv1(s)
        x = nn.ReLU()(s)


        policy = self.policy_head(x)
        value = self.value_head(x)

        return (policy, value)

