import numpy as np
import math
import torch
import chess
import time

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, color=chess.WHITE, search_scope_game=None):
        self.game = game
        self.args = args
        # self.state = copy.deepcopy(state)
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.color = color
        # self.search_scope_game = search_scope_game

        #list of all children nodes
        self.children = []

        #list of all explored children nodes
        # self.explored = []
        
        self.visit_count = 0
        self.value_sum = .0
        self.value = .0 #state value
    
    def is_fully_expanded(self):
        # return len(self.children)>0 and len(self.children) == len(self.explored)
        return len(self.children)
    
    def select(self):

        # print("selecting", len(self.explored))

        vc = torch.tensor([child.visit_count for child in self.children])
        vsum = torch.tensor([child.value_sum for child in self.children])
        prior = torch.tensor([child.prior for child in self.children])

        # for i in self.explored:
        #     t1 = time.perf_counter()
        #     ucb = self.get_ucb(self.children[i])
        #     if ucb > best_ucb:
        #         best_child = i
        #         best_ucb = ucb
        # # best_child.game = self.game
        # # self.game.move_piece(best_child.action_taken)
        #     t2 = time.perf_counter()
        #     print(f"Selection loop time: {t2-t1}")

        ucb = self.get_ucb(vc, vsum, prior)

        # print(prior)
        # print(ucb)

        return self.children[torch.argmax(ucb).item()]
    
    def get_ucb(self, vc, vsum, prior):

        q_value = 1 - (vsum/(vc+1e-6) + 1) / 2
        # q_value = -vsum/(vc+1e-7)
        # print(q_value)
        # if child.visit_count == 0:
        #     q_value = 0
        # else:
        #     q_value = 1 - ((child.value_sum / (child.visit_count)) + 1) / 2 # The +1, /2, is to normalize, 
        #                                                             # otherwise the game is designed 
        #                                                             # such that it would return -1 to 1
        #                                                             # 1 - at the start to choose the worst
        #                                                             # next move because it is the opponent

        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (vc + 1)) * prior
    
    def expand(self, policy):

        """
        Reeived policy as a list of tuples (action, prob)
        """
        # if policy is not None:

        for action, prob in policy:
            # print(self.state)
            # print(action, prob)

            # child_state = copy.deepcopy(self.state)

            # child_state = self.game.get_next_state(child_state, action, 1)
            # child_state = self.game.change_perspective(child_state, player=-1)

            child = Node(game=None,
                            args=self.args,
                            state=None,
                            parent=self,
                            action_taken=action,
                            prior=prob.item(),
                            color=not self.color)
            self.children.append(child)


        # else:

        #     rand_child = np.random.choice([i for i in range(len(self.children)) if i not in self.explored])
        #     self.explored.append(rand_child)
    
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)