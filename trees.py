from abc import ABC, abstractmethod
from torch import tensor
from __future__ import annotations
import chess
import numpy as np

"""
Stores trees of different games
"""

def board_to_tensor(board: chess.Board) -> np.ndarray:
    # Define a mapping from pieces to integers
    piece_to_int = {
        None: 0,
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }

    # Initialize an empty 8x8 tensor
    tensor = np.zeros((8, 8), dtype=int)

    # Fill the tensor with values based on the board state
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Determine the value to assign (positive for white, negative for black)
            value = piece_to_int[piece.piece_type]
            if piece.color == chess.BLACK:
                value = -value
            # Assign the value to the tensor
            row, col = divmod(square, 8)
            tensor[row, col] = value

    return tensor


class Node(ABC):

    def __init__(self, s=None, p: Node = None) -> None:

        self.state = s
        self.W = 0.0
        self.N = 0
        self.parent = p

        # Add the number of next moves possible
        self.policy = []
        for move in self.state.legal_moves:
            self.policy.append(Node(self.state.push(move), self))

    @abstractmethod
    def getNextState(self) -> Node:
        """
        Returns a node with a new state that is reachable from the current state
        and update children list with new state

        Updates policy based on action taken
        """

        pass

    @abstractmethod
    def updateWN(self, W: float = 0.0) -> None:
        """
        Update "Wins" and number of simulations (by default 1) during backpropagation
        """

        self.W += W
        self.N += 1

    @abstractmethod
    def __get__(self) -> tuple:
        """
        Returns a tuple of node attributes needed to generate training set,
        e.g. state and policy (action taken)
        """

        # FIXME: What kind of tuple you expect here?

        return (self.state, self.policy)

    @abstractmethod
    def explored(self) -> bool:
        """
        Returns if the node has been fully explored
        """

        # TODO: How do you determine this? Is a node ever fully explored?
        pass

    @abstractmethod
    def terminal(self) -> bool:
        """
        Returns if the node is a terminal node (no children)
        """

        return len(self.policy) == 0
