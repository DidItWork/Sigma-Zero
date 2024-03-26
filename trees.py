from abc import ABC, abstractmethod
from  copy import deepcopy
from torch import tensor
from __future__ import annotations
import chess
import numpy as np
import random


"""
Stores trees of different games
"""

class Node(ABC):

    def __init__(self, s=None, p: Node = None) -> None:

        self.state = s
        self.W = 0.0
        self.N = 0
        self.parent = p
        self.children = []
        self.explored_moves = []

        # TODO: Probably want to add the way that the state is added

        # Add the number of next moves possible
        self.policy = []
        # for move in self.state.legal_moves:
        #     self.policy.append(Node(self.state.push(move), self))

        self.visit_count = 0
        self.value_sum = 0.0

    @abstractmethod
    def getNextState(self) -> Node:
        """
        Returns a node with a random new state that is reachable from the current state
        that has not already been explored and update children list with new state

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
    def updateValue(self, value: float = 0.0) -> None:
        """
        Update policy value (default 0.0) during backpropagation
        """

        self.value_sum += value

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
        Returns if the node has been fully explored, no more children to be generated from getNextState
        """

        # TODO: How do you determine this? Is a node ever fully explored?
        # Node fully explored if all possible moves at 1 depth are exhausted? children == policy?
        return len(self.children) == len(self.policy)
        pass

    @abstractmethod
    def terminal(self) -> bool:
        """
        Returns if the node is a terminal node (no children)
        """

        return len(self.policy) == 0


class ChessNode(Node):

    def __init__(self, s=None, p: Node = None) -> None:
        super().__init__()

    def getNextState(self) -> Node:
        """
        Returns a node with a random new state that is reachable from the current state
        that has not already been explored and update children list with new state

        """
        unexplored_moves = [x for x in self.state.legal_moves if x not in self.explored_moves]
        explored_move = random.choice(unexplored_moves)
        # TODO: are explored Nodes propagated and backpropped as well to be in the state?
        self.explored_moves.append(explored_move)
        new_state = deepcopy(self.state)
        child = ChessNode(new_state.push(explored_move), self)
        self.chilren.append(child)
        return child
        pass

    def updateWN(self, W: float = 0.0) -> None:
        """
        Update "Wins" and number of simulations (by default 1) during backpropagation
        """

        self.W += W
        self.N += 1

    def updateValue(self, value: float = 0.0) -> None:
        """
        Update policy value (default 0.0) during backpropagation
        """

        self.value_sum += value

    def __get__(self) -> tuple:
        """
        Returns a tuple of node attributes needed to generate training set,
        e.g. state and policy (action taken)
        """

        # FIXME: What kind of tuple you expect here?

        return (self.state, self.policy)

    def explored(self) -> bool:
        """
        Returns if the node has been fully explored, no more children to be generated from getNextState
        """

        # TODO: How do you determine this? Is a node ever fully explored?
        # Node fully explored if all possible moves at 1 depth are exhausted? children == legal_moves?
        return len(self.children) == len(self.state.legal_moves)

    def terminal(self) -> chess.Color:
        """
        Returns if the node is a terminal node (no children) by giving winner's colour, else None if not terminal
        """
        # Can use Outcome to give the reason for termination, the winner or None, and the result
        outcome = self.state.outcome()
        if outcome is not None:
            outcome = outcome.winner
        return outcome
    