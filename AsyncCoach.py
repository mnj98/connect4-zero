import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from AsyncArena import Arena
#from Arena import Arena
from MCTS import MCTS
import time
import multiprocessing
log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        #self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.iter = 1
        self.consecutive_rejections = 0
        self.arena_games = self.args.arena_execs
        self.num_arenas = self.args.arenaCompare // self.arena_games

    def arenaPlay(self):
        arena = Arena(lambda x: np.argmax(MCTS(self.game, self.pnet, self.args).getActionProb(x, temp=0)),
                      lambda x: np.argmax(MCTS(self.game, self.nnet, self.args).getActionProb(x, temp=0)), self.game, num_games = self.arena_games)
        return arena.playGames()

    def executeEpisodes(self):
        examples = []
        for _ in range(self.args.num_selfplay_execs):
            examples.extend(self.executeEpisode())
        return examples

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        curPlayer = 1
        episodeStep = 0
        mcts = MCTS(self.game, self.nnet, self.args)
        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            #print(f"player before game ended: {self.curPlayer}")
            r = self.game.getGameEnded(board, curPlayer)
            #print(f"playe after game ended: {self.curPlayer}")
            #print(f"r: {r}")
            # if the game ends when you check and it's -1, you lost
            #print(trainExamples)
            if r != 0:
                #print(f"GAME OVER ========")
                #print(f"train examples: {trainExamples}")
                if r == self.args.training_draw_penalty:
                    #print("DRAW")
                    #current player gets negative penalty because they went first
                    if self.args.draw_always_bad:
                        return [(x[0], x[2], -0.1) for x in trainExamples]
                    else:
                        return [(x[0], x[2], -r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]
                else:
                    return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(self.iter, self.args.numIters + 1):
            self.iter = i
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                #print("Starting Async Self Play")
                p = []
                #self_play_start = time.time()
                num_execs = self.args.numEps // self.args.num_selfplay_execs
                with multiprocessing.Pool(processes= self.args.async_mcts_procs) as pool:
                    for ep in range(num_execs):
                        #try to make this async, should work if we can make a copy of all the args
                        #and pass the mcts to executeEpisode
                        p.append(pool.apply_async(self.executeEpisodes))
                    for ep in tqdm(range(num_execs), desc="Async Self Play"):
                        iterationTrainExamples.extend(p[ep].get(timeout=180))
                    pool.close()
                    pool.join()
                
                '''
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    #try to make this async, should work if we can make a copy of all the args
                    #and pass the mcts to executeEpisode
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples.extend(self.executeEpisode())
                #print(f"Completed Async Self Play in {time.time() - self_play_start}s")
                '''
                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
                
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # remove unneccessary training examples storage
            
            if self.args.trim_examples != 0:
                self.trimExamples()
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            self.nnet.train(trainExamples)

            log.info("PITTING AGAINST PREVIOUS VERSION")

            p = []
            #self_eval_start = time.time()
            pwins, nwins, draws = 0, 0, 0
            with multiprocessing.Pool(processes= self.args.async_mcts_procs) as pool:
                for eval in range(self.num_arenas):
                    #try to make this async, should work if we can make a copy of all the args
                    #and pass the mcts to executeEpisode
                    p.append(pool.apply_async(self.arenaPlay))
                    
                for eval in tqdm(range(self.num_arenas), desc="Async Pitting"):
                    results = p[eval].get(timeout=180)
                    pwins += results[0]
                    nwins += results[1]
                    draws += results[2]

                pool.close()
                pool.join()

            #print(f"Completed Evaluations in {time.time() - self_eval_start}s")
            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            reject = pwins + nwins == 0 or float(nwins) / (pwins + nwins + (draws * self.args.draw_penalty)) < self.args.updateThreshold


            if reject:
                log.info('REJECTING NEW MODEL')
                self.consecutive_rejections += 1
                if self.args.rebase_to_best_on_reject != 0 and self.consecutive_rejections >= self.args.rebase_to_best_on_reject:
                    self.consecutive_rejections = 0
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                else:
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.consecutive_rejections = 0
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def getTrainExampleFile(self, examplesFile):
        log.info(f"File: {examplesFile}  with trainExamples found. Loading it...")
        with open(examplesFile, "rb") as f:
            self.trainExamplesHistory = Unpickler(f).load()
        log.info('Loading done!')
        # examples based on the model were already collected (loaded)
        self.skipFirstSelfPlay = True

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            log.warning(f'Please input checkpoint example data # to load. If starting new model, input 0.\nYou will lose the consecutive rejects count, but probably do want to do this:')
            self.iter = int(input()) + 2
            examplesFile = os.path.join(self.args.load_folder_file[0], "checkpoint_" + str(self.iter - 2) + ".pth.tar.examples")
            if not os.path.isfile(examplesFile):
                if self.iter != 2:
                    r = input(f'File "{examplesFile}" with trainExamples not found either. Continue? [y|n]')
                    if r != "y":
                        sys.exit()
                else:
                    self.iter = 1
            else:
                self.getTrainExampleFile(examplesFile) 
        else:
            self.getTrainExampleFile(examplesFile)

    def trimExamples(self):
        examplesFile = os.path.join(self.args.checkpoint, "checkpoint_" + str(self.iter - 1 - self.args.trim_examples) + ".pth.tar.examples")
        if os.path.isfile(examplesFile):
            log.info(f"Removing: {examplesFile} to trim examples.")
            os.remove(examplesFile)