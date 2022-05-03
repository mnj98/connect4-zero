from connect4.torch_3L_K3_C18_1LS.NNet import NNetWrapper as NNet
MODEL = "torch_3L_K3_C18_1LS"

import subprocess
import os

import Arena
from MCTS import MCTS
from connect4.Connect4Game import Connect4Game as Game
from connect4.Connect4Players import *
from main import args
import numpy as np
from utils import *


def main():
    solver_dir = os.path.join(os.curdir, "connect4solver")
    solver_file = os.path.join(solver_dir, "c4solver")
    if not os.path.isfile(solver_file):
        subprocess.run(['make', '-C', solver_dir, 'clean'])
        subprocess.run(['make', '-C', solver_dir, 'c4solver'])
    
    book_file = os.path.join(solver_dir, "7x6.book")

    game = Game(args)
    solver = SolverPlayer(game, solver_file, book_file)
    solver_player = solver.play

    # maps out the performance of each checkpoint of a model vs the solver over 300 games (150 going first each x 2)

    # TODO: LOAD THE MODEL
    # TODO: MAKE MODEL PLAY AGAINST THE SOLVER OR AGAINST ITSELF AND HAVE THE SOLVER JUDGE PERFORMANCE
    models_path = os.path.join("saved_checkpoints", MODEL)
    MAX_MODEL_CHECKPOINTS = 200
    NUM_GAMES = 10
    results = []
    # nnet players
    
    for i in range(MAX_MODEL_CHECKPOINTS + 1):
        print(f"Evaluating Checkpoint {i}")
        n1 = NNet(game)
        checkpoint_file = "checkpoint_"+ str(i) + ".pth.tar"
        if os.path.isfile(os.path.join(models_path, checkpoint_file)):
            n1.load_checkpoint(models_path, checkpoint_file)
            mcts1 = MCTS(game, n1, args)
            n1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
            # The solver will always win the fist move so to test models we only need to play with the solver as Player 2.
            # Setting switch to false to achieve this
            arena = Arena.Arena(n1, solver_player, game, display=Game.display, solver=True, switch=False)
            n1_wins, solver_wins, draws = arena.playGames(NUM_GAMES, verbose=False)
            results.append((i, n1_wins, solver_wins, draws))
    print(results)
    f = open(os.path.join(models_path, "results.txt"), "w")
    f.write(str(results))
    f.close()
    
if __name__ == "__main__":
    main()