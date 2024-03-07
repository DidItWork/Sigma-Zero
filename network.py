import torch
import torch.nn as nn
import torch.nn.functional as F

class policyNN(nn.Module):
    """
    
    Policy Neural Network that returns the probabilities of actions to be taken
    given the current state input of the network, as well as the value of the
    current board state.

    """
    def __init__(self, config):
        super(policyNN, self).__init__()

    def forward(self, s):
        pass