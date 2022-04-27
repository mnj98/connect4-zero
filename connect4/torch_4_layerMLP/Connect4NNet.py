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
        self.fc1 = nn.Linear(self.board_x * self.board_y, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)

        self.fc5 = nn.Linear(256, self.action_size)
        self.fc6 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)

    def forward(self, s):
        #                                                                                           s: batch_size x board_x x board_y
        s = s.view(-1, self.board_x * self.board_y)                                                 # batch_size x board_x * board_y
        s = F.relu(self.bn1(self.fc1(s)))                                                           # batch_size x 256
        s = F.relu(self.bn2(self.fc2(s)))                                                           # batch_size x 512
        s = F.dropout(F.relu(self.bn3(self.fc3(s))), p=self.args.dropout, training=self.training)   # batch_size x 512
        s = F.dropout(F.relu(self.bn4(self.fc4(s))), p=self.args.dropout, training=self.training)   # batch_size x 256

        pi = self.fc5(s)                                                                            # batch_size x action_size
        v = self.fc6(s)                                                                             # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
