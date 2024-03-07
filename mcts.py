import numpy as np
import torch.nn as nn
from trees import Node

class MCTS0:
    """

    Class for Monte Carlo Tree Search Modified for Alpha Zero    
    
    """
    def __init__(self, 
                 model : nn.Module = None,
                 tree_root : Node = None,
                 ) -> None:
        
        self.root = tree_root

    def explore(self) -> None:
        """
        
        Implementation of the explore stage, traversing from the root node to a node
        which has not been fully explored or when a terminal node is reached.

        """
        pass

    def backpropagate(self, x : Node = None) -> None:
        """
        
        Backpropogate step: updating W and N values of each node from the current node
        back to the root node.

        """
        pass

    def search(self) -> None:
        """
        
        Perform one iteration of explore and backpropagation

        """
        pass

    def play(self) -> None:
        """
        
        Perform multiple searches until the maximum iterations is reached

        """
        pass        
