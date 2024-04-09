import numpy as np
import torch
from mctsnode import Node
from chess_tensor import ChessTensor, actionToTensor, tensorToAction, actionsToTensor
from network import policyNN
import chess
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"

# FOR MAC SILICON USERS!!!!!!!!
# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#             "built with MPS enabled.")
#     else:
#         print("MPS not available because the current MacOS version is not 12.3+ "
#             "and/or you do not have an MPS-enabled device on this machine.")

# else:
#     device = torch.device("mps")

class MCTS0:
    """

    Class for Monte Carlo Tree Search Modified for Alpha Zero    
    
    """
    def __init__(self,
                 game=None,
                 args=None,
                 model=None):
        self.game = game
        self.args = args # args = ['C', 'num_searches', 'num_iterations', 'num_selfPlay_iterations', 'num_epochs', 'batch_size']
        self.model = model

    @torch.no_grad()
    def search(self, state, verbose=True):

        # Define root node
        root = Node(self.game, self.args, state, color=chess.WHITE if state.turn else chess.BLACK)

        # Selection
        for search in range(self.args['num_searches']):

            root.game = copy.deepcopy(self.game)
            
            if verbose: print("Searching iteration ", search)
            node = root

            while node.is_fully_expanded():
                # print('selection occuring')
                node = node.select()
            
            # print(node.state)
            if verbose: print("Searching node")
            if verbose: print(node.game.board, node.color)
            
            value, is_terminal = node.game.get_value_and_terminated()
            # value = node.game.get_opponent_value(value)

            if not is_terminal:

                if len(node.children)==0:

                    valid_moves = node.game.get_valid_moves(node.game.board)

                    # print(valid_moves)

                    policy_mask, queen_promotion = actionsToTensor(valid_moves, node.color)

                    policy_mask = policy_mask.to(device)
                    
                    policy, value = self.model(
                        # stateTensor.unsqueeze(0).to(device),
                        node.game.get_representation().unsqueeze(0).to(device)
                    )

                    policy = policy.squeeze(0).detach() * policy_mask

                    policy /= torch.sum(policy)

                    policy = policy.squeeze(0)

                    value = value.item()

                    # print(policy.shape, policy, policy.nonzero())
                    # print(value)

                    valid_moves = tensorToAction(policy, node.color, queen_promotion=queen_promotion)

                    probs = policy[policy.nonzero()]

                    policy_list = list(zip(valid_moves, probs))
                
                else:

                    policy_list = None

                # print(policy_list)

                node.expand(policy_list)  # Expansion

            # Backpropagation
            node.backpropagate(value)

        # Return visit counts
        # If number of simulations is too low, none of the root's children may be visited
        action_probs = {}
        for i in root.explored:
            child = root.children[i]
            action_probs[child.action_taken] = child.visit_count
        sum_values = sum(action_probs.values())
        action_probs = {k: v / sum_values for k, v in action_probs.items()}
        return action_probs

if __name__ == "__main__":

    game = ChessTensor()
    config = dict()
    model = policyNN(config).to(device)
    model.eval()
    args = {
        'C': 2,
        'num_searches': 200,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 64
    }
    mcts = MCTS0(game=game,
                 args = args,
                 model=model)
    
    boardTensor = game.get_representation()
    
    print(mcts.search(game.board))
    