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
        self.board_discovery_layer = nn.Linear(self.board_x * self.board_y, self.board_x * self.board_y) #sees if a spot could be considered a potential threat or score, k size 5x5, ouput is board size
        
        # layers used to generate action
        self.analysis_layer = nn.Linear(self.board_x * self.board_y, self.board_x * self.board_y) # should interperet the threat/score map and provide a signal about a spot's utility
        self.planning_layer = nn.Linear(self.board_x * self.board_y, self.board_x * self.board_y) # should plan out the next move based on the board analysis 
        self.action_layer = nn.Linear(self.board_x * self.board_y, self.action_size) #uses risky spots to determine output

        # layers used to score the board state
        self.status_layer = nn.Linear(self.board_x * self.board_y, self.board_x * self.board_y) #should interpret threat/score map and provide a signal about the current state of the game
        self.assessment_layer = nn.Linear(self.board_x * self.board_y, 1)

        # batch normalizations
        self.board_discovery_layer_bn = nn.BatchNorm1d(self.board_x * self.board_y)

        self.analysis_layer_bn = nn.BatchNorm1d(self.board_x * self.board_y)
        self.planning_layer_bn = nn.BatchNorm1d(self.board_x * self.board_y)

        self.status_layer_bn = nn.BatchNorm1d(self.board_x * self.board_y)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = self.flatten(s)
        
        # shared layers
        s = F.leaky_relu(self.board_discovery_layer_bn(self.board_discovery_layer(s)))

        # action layers
        pi = F.dropout(self.analysis_layer_bn(F.leaky_relu(self.analysis_layer(s))), p=0.1)
        pi = F.dropout(self.planning_layer_bn(F.leaky_relu(self.planning_layer(pi))), p=0.1)
        pi = F.log_softmax(self.action_layer(pi), dim=1)

        # assessment layers
        v = F.dropout(self.status_layer_bn(F.leaky_relu(self.status_layer(s))), p=0.1)
        v = torch.tanh(self.assessment_layer(v))

        return pi, v
