import logging

import coloredlogs

from Coach import Coach
#from othello.OthelloGame import OthelloGame as Game
from connect4.Connect4Game import Connect4Game as Game
from connect4.torch_3L_K3_C18_1LS.NNet import NNetWrapper as nn
#from othello.pytorch.NNet import NNetWrapper as nn
from utils import *
import os
0
log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 500,
    'numEps': 300,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.505,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 150,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './saved_checkpoints/torch_3L_K3_C18_1LS_2',
    'load_model': True,
    'load_folder_file': ('./saved_checkpoints/torch_3L_K3_C18_1LS_2','temp.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'rebase_to_best_on_reject': 15, # rebase after this many rejections in a row, 1 to rebase immediately, 0 to disable rebasing
    'trim_examples': 1 # Trim examples this far back. Does nothing if set to 0
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(6)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        checkpoint_path = os.path.join(args.load_folder_file[0], args.load_folder_file[1])
        if os.path.isfile(checkpoint_path):
            log.info('Loading checkpoint "%s', checkpoint_path)
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        else:
            log.warning("Failed to load model checkpoint!")
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
