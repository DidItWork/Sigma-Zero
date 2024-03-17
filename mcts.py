import numpy as np
import torch.nn as nn
from trees import Node

C = 1.41  # Exploration parameter for UCB

class MCTS0:
    """

    Class for Monte Carlo Tree Search Modified for Alpha Zero    
    
    """
    def __init__(self, 
                 model : nn.Module = None,
                 tree_root : Node = None,
                 ) -> None:
        
        self.root = tree_root
        self.model = model

    def ucb_score(self, parent, child):
        """

        Calculate the UCB score for a child node.

        """
        if child.visits == 0:
            return float('inf')  # Encourage exploration of unvisited nodes
        return child.value / child.visits + C * np.sqrt(np.log(parent.visits) / child.visits)

    def explore(self) -> None:
        """
        
        Implementation of the explore stage, traversing from the root node to a node
        which has not been fully explored or when a terminal node is reached.

        Traverse the tree from the root to a leaf node, selecting children based on the highest UCB score.
        
        """
        current_node = self.root
        while not current_node.is_terminal() and current_node.children:
            current_node = max(current_node.children, key=lambda x: self.ucb_score(current_node, x))
        return current_node

    def backpropagate(self, node, result, x : Node = None) -> None:
        """
        
        Backpropogate step: updating W and N values of each node from the current node
        back to the root node.

        Backpropagate the result from a simulation up the tree, updating node statistics.
        
        """
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

    def simulate(self, node):
        """
        Simulate a game from the given node, returning the result.
        Placeholder for actual game simulation or neural network evaluation.
        """
        # Placeholder: this should be replaced with actual game simulation or model evaluation logic
        return np.random.rand()

    def search(self) -> None:
        """
        
        Perform one iteration of explore and backpropagation

        Perform one iteration of MCTS: explore, simulate, and backpropagate.
        
        """
        leaf_node = self.explore()
        if not leaf_node.is_terminal():
            result = self.simulate(leaf_node)
            self.backpropagate(leaf_node, result)

    def play(self, max_iterations=100) -> None:
        """
        
        Perform multiple searches until the maximum iterations is reached

        Perform MCTS to determine the next move.
        
        """
        for _ in range(max_iterations):
            self.search()
        # Select the move with the highest visit count from the root node
        return max(self.root.children, key=lambda x: x.visits)