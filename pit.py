import Arena
from MCTS import MCTS
from connect4.Connect4Game import Connect4Game as Game
from connect4.Connect4Players import *
from connect4.torch_3L_K3_C18_1LS.NNet import NNetWrapper as NNet
from connect4.torch_4_layerCNN.NNet import NNetWrapper as NNet4L
from main import args
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

args1 = args

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./saved_checkpoints/torch_3L_K3_C18_1LS_2', 'best.pth.tar')
mcts1 = MCTS(g, n1, args1)
n1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


n2 = NNet4L(g)
n2.load_checkpoint('./saved_checkpoints/4layerCNN', '44rounds.pth.tar')
mcts2 = MCTS(g, n2, args1)
n2 = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
'''
n2 = MLP(g)
n2.load_checkpoint('./saved_checkpoints/MLP', 'best.pth.tar')
mcts2 = MCTS(g, n2, args1)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
'''

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
arena = Arena.Arena(n1, hp, g, display=Game.display)

print(f"P1 Wins, P2 Wins, Draws: \n {arena.playGames(10, verbose=True)}")
