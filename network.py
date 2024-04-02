import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from chess_tensor import actionsToTensor
from random import shuffle
from torch.optim import Adam
from torch import tensor, Tensor
from typing import Optional, Callable

"""
Copied from pytorch implementation of resnet
https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
"""

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
"""
End of ResNet
"""

class policyNN(nn.Module):
    """
    
    Policy Neural Network that returns the probabilities of actions to be taken
    given the current state input of the network, as well as the value of the
    current board state.

    Input is in the form of 8x8x119 and output will be in the form of a policy
    vector and a scalar value.

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

        self.resnet_blocks = []

        self.optimiser = Adam(self.parameters(), lr=0.001, weight_decay=0.0001)

        for _ in range(19):

            self.resnet_blocks.append(BasicBlock(256, 256, 1))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

    def policy_head(self, x: tensor) -> tensor:

        x = self.conv_p1(x)
        x = nn.ReLU()(x)
        x = self.conv_p2(x)
        x = torch.flatten(x, start_dim=1)

        return x

    def value_head(self, x: tensor) -> tensor:

        x = self.conv_v1(x)
        x = nn.ReLU()(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_v1(x)
        x = self.fc_v2(x)
        x = torch.tanh(x)

        return x   

    def forward(self, x: tensor, policy_mask: tensor = None) -> tuple:

        x = self.conv1(x)
        x = nn.ReLU()(x)

        x = self.resnet_blocks(x)

        policy = self.policy_head(x).cpu()

        value = self.value_head(x).cpu()

        # print(x.shape, policy.shape)

        if policy_mask == None:

            policy_mask = torch.ones(policy.shape)

        #masked softmax

        policy_exp = torch.exp(policy)*policy_mask

        policy_exp_sum = torch.sum(policy_exp, dim=1)-torch.sum(policy_mask)

        policy = policy_exp/policy_exp_sum

        return (policy, value)
    
    def backward(self, training_data) -> None:
        shuffle(training_data)
        for index in range(len(training_data)):
            game_history = training_data[index]
            loss = torch.zeros(1).to("cuda").requires_grad_(True)
            for move in zip(game_history["states"], game_history["actions"], game_history["rewards"]):
                # print(move)
                # policy_mask = validActionsToTensor(move[1]).unsqueeze(0)
                p, v = self.forward(move[0].unsqueeze(0).cuda())
                p_target, v_target = actionsToTensor(move[1])[0], move[2]
                p_target = torch.reshape(p_target, (p_target.size()[0], 1))
                v_target = torch.tensor(v_target, dtype=torch.float32, device="cuda", requires_grad=True)
                v = torch.tensor(v, dtype=torch.float32, device="cuda", requires_grad=True)
                # print(p, v)
                # print(p_target, v_target)
                # print(p.size(), p_target.size())
                # print(torch.log(p.to("cuda")).size())
                move_loss = torch.sub(
                        torch.pow(torch.sub(v_target.to("cuda"), v.to("cuda")), 2), 
                        torch.matmul(torch.log(p.to("cuda")), p_target.to("cuda"))
                )
                print(move_loss)
                # print(move_loss.size())
                if torch.any(move_loss.isnan()):
                    loss = torch.add(loss, move_loss)

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()





        # for game_history in training_data:
        #     # loss = torch.tensor()
        #     for move in zip(game_history["states"], game_history["actions"], game_history["rewards"]):
        #         p, v = self.forward(move[0].unsqueeze(0).cuda())
        #         c = 2
        #         print(p, v)
        #         # print(self.parameters)
        #         move_loss = torch.sub(
        #             torch.pow((move[2] - v), 2), 
        #             torch.add(
        #                 (move[1].T * torch.log(p)), 
        #                 (torch.mul(torch.pow(torch.abs(self.parameters), 2)), c)
        #             )
        #         )
        #         print(move_loss)
        #         print(move_loss.size())

if __name__=="__main__":

    from chess_tensor import *

    config = dict()

    network = policyNN(config).to("cuda")

    game = ChessTensor()

    board = game.get_representation().unsqueeze(0).cuda()

    policy, value = network(board)

    print(policy)
    print(value)
