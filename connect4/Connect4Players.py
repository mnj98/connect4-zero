import random
from matplotlib.pyplot import draw
import numpy as np
import sys
import subprocess

from connect4.Connect4Game import Connect4Game
from connect4.Connect4Logic import Board

class RandomPlayer():
    def __init__(self, game: Connect4Game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a

class SolverPlayer():
    def __init__(self, game: Connect4Game, solver_file, book_file):
        self.game = game
        self.solver_file = solver_file
        self.book_file = book_file

    def play(self, board):
        move = random.choice(self.best_move(board))
        #print(f'solver plays: {move}')
        return move

    def best_move(self, board):
        # may need to check the precoditions in the solvers main.cpp
        moves_made = ""
        for move in self.game.moves_made:
            moves_made += str(move+1)

        #print(f"moves made: {moves_made}")
        best_score = 23 #just needs to be an impossibly high score #goal is to minimize the opponents best score
        best_moves = []
        solver_scores = []
        solver_actions = []
        winners = []
        blockers = []
        drawers = []

        player_value = 1
        #print(f"player value: {player_value}")

        valid = self.game.getValidMoves(np.copy(board), player_value)
        for i in range(0, 7):
            if not valid[i]:
                continue
            
            if self.check_win(i, np.copy(board), player_value) == 1:
                # Winning move for other player, append to blockers
                winners.append(i)
            elif self.check_win(i, np.copy(board), -player_value) == 1:
                #print(f'{i} IS A WINNING MOVE for {-1 * player_value}, SHOULD BE TAKEN UNLESS WINNING MOVE FOR SELF IS FOUND')
                blockers.append(i)
            else:
                results = subprocess.run([self.solver_file, '-b', self.book_file, '', moves_made+str(i+1)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                try:

                    score = int(results.stdout)
                    solver_scores.append(score)
                    solver_actions.append(i)
                except:
                    print(f"ERROR received non-int score for: {i}")
                    print(results.stdout)
                    print(results.stderr)
                    print(f"check_win output: {self.check_win(i, board, player_value)}")
                    winners.append(i)
        '''
        print(f"moves_made: {moves_made}")
        print(f"solver_scores: {solver_scores}")
        print(f"solver_actions: {solver_actions}")
        print(f"solver_winners: {winners}")
        print(f"solver_blockers: {blockers}")
        '''

        for i in range(len(solver_scores)):
            if solver_scores[i] == best_score:
                best_moves.append(solver_actions[i])
            if solver_scores[i] < best_score:
                best_score = solver_scores[i]
                best_moves = [solver_actions[i]]
        
        if len(winners) > 0:
            return winners
        
        if len(blockers) > 0:
            return blockers

        if best_score > 0 and len(drawers) > 0:
            return drawers

        if len(best_moves) > 0:
            #print(f"solver_best: {best_moves}")
            return best_moves

        else:
            i = 0
            while not valid[i]:
                i += 1
            return [i]

    def check_win(self, action, board, player_value):
        b = Board(height=6, width=7, win_length=4, np_pieces=board)
        b.add_stone(action, player_value)
        winstate = b.get_win_state()
        if winstate.is_ended:
            if winstate.winner is None:
                # draw has very little negative value for player 1 and small positive if player 2.
                if player_value == -1:
                    return -0.5
                elif player_value == 1:
                    return 0.5
                else:
                    raise ValueError('Unexpected winstate found: ', winstate)
            elif winstate.winner == player_value:
                return +1
            elif winstate.winner == -player_value:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

class HumanConnect4Player():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, 1)
        print('\nMoves:', [i for (i, valid) in enumerate(valid_moves) if valid])

        while True:
            move = input()
            try:
                move = int(move)
                if valid_moves[move]: break
                else: print('Invalid move')
            except:
                print('Invalid move')
        return move


class OneStepLookaheadConnect4Player():
    """Simple player who always takes a win if presented, or blocks a loss if obvious, otherwise is random."""
    def __init__(self, game, verbose=True):
        self.game = game
        self.player_num = 1
        self.verbose = verbose

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, self.player_num)
        win_move_set = set()
        fallback_move_set = set()
        stop_loss_move_set = set()
        for move, valid in enumerate(valid_moves):
            if not valid: continue
            if self.player_num == self.game.getGameEnded(*self.game.getNextState(board, self.player_num, move)):
                win_move_set.add(move)
            if -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, -self.player_num, move)):
                stop_loss_move_set.add(move)
            else:
                fallback_move_set.add(move)

        if len(win_move_set) > 0:
            ret_move = np.random.choice(list(win_move_set))
            if self.verbose: print('Playing winning action %s from %s' % (ret_move, win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set))
            if self.verbose: print('Playing loss stopping action %s from %s' % (ret_move, stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            ret_move = np.random.choice(list(fallback_move_set))
            if self.verbose: print('Playing random action %s from %s' % (ret_move, fallback_move_set))
        else:
            raise Exception('No valid moves remaining: %s' % game.stringRepresentation(board))

        return ret_move
