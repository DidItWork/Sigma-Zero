import numpy as np
import math
import torch
import chess
import time

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, color=chess.WHITE, search_scope_game=None):
        self.game = game
        self.args = args
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.color = color
        self.children = []
        self.visit_count = 0
        self.value_sum = .0
        self.value = .0 #state value
    
    def is_fully_expanded(self):
        return len(self.children)
    
    def select(self):

        vc = torch.tensor([child.visit_count for child in self.children])
        vsum = torch.tensor([child.value_sum for child in self.children])
        prior = torch.tensor([child.prior for child in self.children])

        ucb = self.get_ucb(vc, vsum, prior)

        return self.children[torch.argmax(ucb).item()]
    
    def get_ucb(self, vc, vsum, prior):

        q_value = 1 - (vsum/(vc+1e-6) + 1) / 2

        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (vc + 1)) * prior
    
    def expand(self, policy):

        """
        Reeived policy as a list of tuples (action, prob)
        """
    
        for action, prob in policy:
            
            child = Node(game=None,
                            args=self.args,
                            state=None,
                            parent=self,
                            action_taken=action,
                            prior=prob.item(),
                            color=not self.color)
            self.children.append(child)
    
    def backpropagate(self, value):
        # print(value)
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)