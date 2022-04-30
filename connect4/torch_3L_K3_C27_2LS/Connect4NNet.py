import sys
sys.path.append('..')
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class Connect4NNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args


        super(Connect4NNet, self).__init__()

        self.flatten = nn.Flatten()

        # layers that are shared
        # 3x3x2 = 18 output layers, goal is for each layer to learn a single spot on the 3x3 for player/opponent
        self.board_discovery_layer = nn.Conv2d(1, 27, kernel_size=3, stride=1, padding=1) #sees if a spot could be considered a potential threat or score, k size 5x5, ouput is board size
        self.analysis_layer = nn.Linear(self.board_x * self.board_y * 27, self.board_x * self.board_y * 3) # should interperet the threat/score map and provide a signal about a spot's utility
        
        # layers used to generate action
        self.interpretation_layer = nn.Linear(self.board_x * self.board_y * 3, self.board_x * self.board_y * 3)
        self.planning_layer = nn.Linear(self.board_x * self.board_y * 3, self.board_x * self.board_y) # should plan out the next move based on the board analysis 
        self.action_layer = nn.Linear(self.board_x * self.board_y, self.action_size) #uses risky spots to determine output

        # layers used to score the board state
        self.status_layer = nn.Linear(self.board_x * self.board_y * 3, self.board_x * self.board_y) #should interpret threat/score map and provide a signal about the current state of the game
        self.assessment_layer = nn.Linear(self.board_x * self.board_y, 1)

        # batch normalizations
        self.board_discovery_layer_bn = nn.BatchNorm2d(27)
        self.analysis_layer_bn = nn.BatchNorm1d(self.board_x * self.board_y * 3)

        self.interpretation_layer_bn = nn.BatchNorm1d(self.board_x * self.board_y * 3)
        self.planning_layer_bn = nn.BatchNorm1d(self.board_x * self.board_y)

        self.status_layer_bn = nn.BatchNorm1d(self.board_x * self.board_y)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_y, self.board_x)                # batch_size x 1 x board_x x board_y
        
        # shared layers
        s = F.leaky_relu(self.flatten(self.board_discovery_layer_bn(self.board_discovery_layer(s))))
        s = F.dropout(self.analysis_layer_bn(F.leaky_relu(self.analysis_layer(s))), p=0.1)

        # action layers
        pi = F.dropout(self.interpretation_layer_bn(F.leaky_relu(self.interpretation_layer(s))), p=0.1)
        pi = F.dropout(self.planning_layer_bn(F.leaky_relu(self.planning_layer(pi))), p=0.1)
        pi = F.dropout(F.log_softmax(self.action_layer(pi), dim=1), p=0.1)

        # assessment layers
        v = F.dropout(self.status_layer_bn(F.leaky_relu(self.status_layer(s))), p=0.1)
        v = torch.tanh(self.assessment_layer(v))

        return pi, v
