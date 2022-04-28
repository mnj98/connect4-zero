import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class Connect4NNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args


        super(Connect4NNet, self).__init__()

        #self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.flatten = nn.Flatten()

        self.risk_assessment_layer = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1) #sees if a spot could be considered risky, k size 5x5, ouput is board size (should mark spots at risk)
        self.planning_layer = nn.Linear(self.board_x * self.board_y * 8, self.board_x * self.board_y)
        self.action_layer = nn.Linear(self.board_x * self.board_y, self.action_size) #uses risky spots to determine output
        self.assessment_layer = nn.Linear(self.board_x * self.board_y, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_y, self.board_x)                # batch_size x 1 x board_x x board_y

        s = F.relu(self.flatten(self.risk_assessment_layer(s)))
        s = self.planning_layer(s)

        pi = self.action_layer(s)
        v = self.assessment_layer(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
