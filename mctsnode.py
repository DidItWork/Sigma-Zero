import numpy as np
import math
import copy
import chess

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, color=chess.WHITE):
        self.game = game
        self.args = args
        self.state = copy.deepcopy(state)
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.color = color

        self.children = []
        
        self.visit_count = 0
        self.value_sum = 0
    
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        self.game.move_piece(best_child.action_taken)
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2 # The +1, /2, is to normalize, 
                                                                    # otherwise the game is designed 
                                                                    # such that it would return -1 to 1
                                                                    # 1 - at the start to choose the worst
                                                                    # next move because it is the opponent
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):

        """
        Reeived policy as a list of tuples (action, prob)
        """

        for action, prob in policy:
            # if prob > 0:
            # print(self.state)
            # print(action, prob)

            child_state = copy.deepcopy(self.state)

            child_state = self.game.get_next_state(child_state, action, 1)
            # child_state = self.game.change_perspective(child_state, player=-1)

            child = Node(self.game, self.args, child_state, self, action, prob, color=not self.color)
            self.children.append(child)
        return child
    
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)