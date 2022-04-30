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


        #Model increase the number of nodes to a large number then back down to a small number (512) twice
        super(Connect4NNet, self).__init__()
        self.fc1 = nn.Linear(self.board_x * self.board_y, self.board_x * self.board_y*8) #Increase number of noeds
        self.fc2 = nn.Linear(self.board_x * self.board_y*8, 512) #decrease number of nodes
        self.fc3 = nn.Linear(512, self.board_x * self.board_y*8) #increase number of nodes
        self.fc4 = nn.Linear(self.board_x * self.board_y*8, 256) #decrease number of nodes

        #output
        self.fc5 = nn.Linear(256, self.action_size)
        self.fc6 = nn.Linear(256, 1)



    def forward(self, s):
        #                                                                                           s: batch_size x board_x x board_y
        s = s.view(-1, self.board_x * self.board_y)                                                 # batch_size x board_x * board_y
        s = F.leaky_relu(self.fc1(s))                                                          # batch_size x 256
        s = F.leaky_relu(self.fc2(s))                                                          # batch_size x 512
        s = F.dropout(F.leaky_relu(self.fc3(s)), p=self.args.dropout, training=self.training)   # batch_size x 512
        s = F.dropout(F.leaky_relu(self.fc4(s)), p=self.args.dropout, training=self.training)   # batch_size x 256

        pi = self.fc5(s)                                                                            # batch_size x action_size
        v = self.fc6(s)                                                                             # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
