##IMPORT ALL THE MODEL


from connect4.torch_4L_K3_C18_1LS.NNet import NNetWrapper as torch_4L_K3_C18_1LS_NNet
#from connect4.torch_3L_K3_C27_2LS.NNet import NNetWrapper as torch_3L_K3_C27_2LS_NNet
#from connect4.torch_4L_K3_C27_1LS.NNet import NNetWrapper as torch_4L_K3_C27_2LS_NNet
MODELS = [
    ("torch_4L_K3_C18_1LS_FIXED", torch_4L_K3_C18_1LS_NNet)
]

import subprocess
import os
import fnmatch
import re

import Arena
from MCTS import MCTS
from connect4.Connect4Game import Connect4Game as Game
from connect4.Connect4Players import *
from main import args
import numpy as np
from utils import *
import pandas as pd

def get_results():
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

    for model in MODELS:
        MODEL = model[0]
        models_path = os.path.join("saved_checkpoints", MODEL)
        checkpoints = np.sort(np.array(fnmatch.filter(os.listdir(models_path), "checkpoint_*.pth.tar")))
        num_checkpoints = len(checkpoints)

        checkpoints = checkpoints[np.round(np.linspace(0, len(checkpoints) - 1, 25)).astype(int)]
        #print(checkpoints)
        MAX_MODEL_CHECKPOINTS = 200
        NUM_GAMES = 20
        results = []
        # nnet players

        for checkpoint in checkpoints:
            #print(f"Evaluating Checkpoint {i}")
            n1 = model[1](game)
            checkpoint_file = checkpoint #"checkpoint_"+ str(i) + ".pth.tar"
            if os.path.isfile(os.path.join(models_path, checkpoint_file)):
                n1.load_checkpoint(models_path, checkpoint_file)
                mcts1 = MCTS(game, n1, args)
                n1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
                # The solver will always win the fist move so to test models we only need to play with the solver as Player 2.
                # Setting switch to false to achieve this
                arena = Arena.Arena(n1, solver_player, game, display=Game.display, solver=True, switch=False)
                n1_wins, solver_wins, draws = arena.playGames(NUM_GAMES, verbose=False)


                index = re.findall(r'\d+', checkpoint)
                index = list(map(int, index))[0]
                results.append((index, n1_wins, draws))
        df = pd.DataFrame(results,columns=['iter','wins','draws'])
        #print(df)
        df.sort_values('iter')
        df.to_csv(MODEL + '_results.csv',index=False)
        graph_results(MODEL)

def graph_results(model):
    df = pd.read_csv(model + '_results.csv')

    fig = df.plot(kind='line',x='iter', title='Agent vs Solver',ylabel='number of games',xlabel='iteration number')
    fig.figure.savefig(model + '_results.png')

if __name__ == "__main__":
    #print(fnmatch.filter(os.listdir('./saved_checkpoints/' + MODELS[0]), "checkpoint_*.pth.tar"))
    get_results()
    #graph_results()
