import Arena
from MCTS import MCTS
from connect4.Connect4Game import Connect4Game as Game
from connect4.Connect4Players import *
from connect4.torch_4_layerCNN.NNet import NNetWrapper as NNet4L
from connect4.torch_6_layerCNN.NNet import NNetWrapper as NNet6L
from connect4.torch_4_layerCNN_kernalSize5.NNet import NNetWrapper as NNet5K

from connect4.torch_4_layerMLP.NNet import NNetWrapper as MLP

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

#mini_othello = False  # Play in 6x6 instead of the normal 8x8.
#human_vs_cpu = True
'''
if mini_othello:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)
'''
g = Game()
# all players
rp = RandomPlayer(g).play
lap = OneStepLookaheadConnect4Player(g).play
hp = HumanConnect4Player(g).play



# nnet players
n1 = NNet4L(g)
'''
if mini_othello:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
else:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
'''
n1.load_checkpoint('./saved_checkpoints/4layerCNN', '44rounds.pth.tar')
args1 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = MLP(g)
n2.load_checkpoint('./saved_checkpoints/MLP', 'best.pth.tar')
mcts2 = MCTS(g, n2, args1)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

'''
if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.
'''
arena = Arena.Arena(hp, n2p, g, display=Game.display)

print(arena.playGames(2, verbose=True))
