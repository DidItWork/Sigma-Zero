from abc import ABC, abstractmethod
from torch import tensor
from __future__ import annotations

"""

Stores trees of different games

"""

class Node(ABC):

    def __init__(self, s : tensor = None, p : Node = None) -> None:

        self.state = s
        self.W = 0.
        self.N = 0
        self.children = []
        self.policy = None
        self.parent = None
    
    @abstractmethod
    def getNextState(self) -> Node:
        """
        Returns a node with a new state that is reachable from the current state
        and update children list with new state

        Updates policy based on action taken
        """
        pass
    
    @abstractmethod
    def updateWN(self, W : float = 0., N : int = 1) -> None:
        """
        Update "Wins" and number of simulations (by default 1) during backpropagation
        """
        pass

    @abstractmethod
    def __get__(self) -> tuple:
        """
        Returns a tuple of node attributes needed to generate training set,
        e.g. state and policy (action taken)
        """

        pass

    @abstractmethod
    def explored(self) -> bool:
        """
        Returns if the node has been fully explored
        """
        pass

    @abstractmethod
    def terminal(self) -> bool:
        """
        Returns if the node is a terminal node (no children)
        """
        pass

