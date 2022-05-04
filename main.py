# CHANGE THESE TO MATCH THE MODEL
from connect4.torch_3L_K3_C18_1LS.NNet import NNetWrapper as nn
MODEL = "torch_3L_K3_C18_1LS_FIXED"

import logging
import coloredlogs
from AsyncCoach import Coach
#from othello.OthelloGame import OthelloGame as Game
from connect4.Connect4Game import Connect4Game as Game
#from othello.pytorch.NNet import NNetWrapper as nn
from utils import *
import os
0
log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

CHECKPOINTS = os.path.join('saved_checkpoints', MODEL)
args = dotdict({
    'numIters': 250,
    'numEps': 300,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.501,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 150,         # Number of games to play during arena play to determine if new net will be accepted. # If changing this update slef.arena_games and self.num_arenas in AsyncCoach
    'arena_execs': 10, #number of arena games to play per arena, arenaCompare must be divisible by this
    'cpuct': 1,

    'checkpoint': CHECKPOINTS,
    'load_model': True,
    'load_folder_file': [CHECKPOINTS, 'temp.pth.tar'],
    'numItersForTrainExamplesHistory': 20,
    'rebase_to_best_on_reject': 15, # rebase after this many rejections in a row, 1 to rebase immediately, 0 to disable rebasing
    'trim_examples': 1, # Trim examples this far back. Does nothing if set to 0
    'draw_penalty': 0.0, # Penalization for drawing, 0 is none, 1 is as much as a loss
    'training_draw_penalty': 0.01, # Penalization for drawing during training (CANNOT BE 0). If you want to reward for draws make this positive
    'num_selfplay_execs': 10, # Number of self-plays per exec, numEps must be divisible by this
    'async_mcts_procs': 4
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(args)

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
