import sys
import numpy as np
sys.path.append('..')
from Game import Game
from .Connect4Logic import Board
from utils import *

class Connect4Game(Game):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, args=None, height=None, width=None, win_length=None, np_pieces=None):
        Game.__init__(self)
        if args != None:
            self.args = args
        else:
            print("WARNING: No args supplied to the game, using defaults")
            self.args = dotdict({
                    'training_draw_penalty': 1e4 # Penalization for drawing during training (CANNOT BE 0). If you want to reward for draws make this positive
            })
        self._base_board = Board(height, width, win_length, np_pieces)
        self.moves_made = []

    def getInitBoard(self):
        return self._base_board.np_pieces

    def getBoardSize(self):
        return (self._base_board.height, self._base_board.width)

    def getActionSize(self):
        return self._base_board.width

    def getNextState(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = self._base_board.with_np_pieces(np_pieces=np.copy(board))
        b.add_stone(action, player)
        return b.np_pieces, -player

    def getValidMoves(self, board, player):
        "Any zero value in top row in a valid move"
        return self._base_board.with_np_pieces(np_pieces=board).get_valid_moves()

    def getGameEnded(self, board, player):
        b = self._base_board.with_np_pieces(np_pieces=board)
        winstate = b.get_win_state()
        if winstate.is_ended:
            if winstate.winner is None:
                # draw has very little negative value.
                return self.args.training_draw_penalty * -1
            elif winstate.winner == player:
                return +1
            elif winstate.winner == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board, player):
        # Flip player from 1 to -1
        return board * player

    def getSymmetries(self, board, pi):
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        def piece(p):
            if p == 1: return '🔴'
            if p == -1: return '🟡'
            return '⚪'
        print(" -----------------------")
        if len(board[0]) == 7:
            print('   0    1    2    3    4    5    6')
        else:
            print(''.join(map(str, range(len(board[0])))))
        print(np.array([piece(b) for b in board.flatten()]).reshape(board.shape))
        print(" -----------------------")
