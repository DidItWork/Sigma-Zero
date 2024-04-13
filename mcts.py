import numpy as np
import torch
from mctsnode import Node
from chess_tensor import ChessTensor, actionToTensor, tensorToAction, actionsToTensor
from network import policyNN
import chess
import copy
import time

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
        self.model = model.to(device)
        self.noise_distribution = torch.distributions.dirichlet.Dirichlet

    @torch.no_grad()
    def search(self, state, verbose=True, learning=True):

        #Set model to evaluation mode
        self.model.eval()

        # Define root node
        root = Node(self.game, self.args, state, color=chess.WHITE if state.turn else chess.BLACK)
        
        #Optimisation to use piror the first time we expand
        root.visit_count = 1

        # Selection
        for search in range(self.args['num_searches']):
            
            if verbose: print("Searching iteration ", search)
            node = root

            # t1 = time.perf_counter()

            while node.is_fully_expanded():
                # print('selection occuring')
                node = node.select()

            # t2 = time.perf_counter()

            if node.parent is not None:
                node.game = copy.deepcopy(node.parent.game)
                node.game.move_piece(node.action_taken)
            
            # print(node.state)
            if verbose: print("Searching node")
            if verbose: print(node.game.board, node.color)
            
            value, is_terminal = node.game.get_value_and_terminated()

            # print(f"Selection time: {t2-t1}")

            if not is_terminal:


                # t2_1 = time.perf_counter()

                valid_moves = node.game.get_valid_moves(node.game.board)

                # print(valid_moves)

                policy_mask, queen_promotion = actionsToTensor(valid_moves, node.color)

                policy_mask = policy_mask
                
                policy, value = self.model(
                    # stateTensor.unsqueeze(0).to(device),
                    node.game.get_representation().float().unsqueeze(0).to(device),
                    inference=True
                )

                # print(torch.max(policy)-torch.min(policy), value)

                # print(policy.device, value.device, policy.requires_grad, value.requires_grad)

                policy = policy.squeeze(0).detach().cpu() * policy_mask

                policy /= torch.sum(policy)

                policy = policy.squeeze(0)

                node.value = value.item()

                # print(policy.shape, policy, policy.nonzero())
                # print(value)

                # t2_2 = time.perf_counter()

                valid_moves = tensorToAction(policy, node.color, queen_promotion=queen_promotion)

                probs = policy[policy.nonzero()]

                if learning:
                    n_moves = torch.full(probs.shape, 0.3)
                    noise = self.noise_distribution(n_moves).sample()
                    eps = 0.25

                    probs = (1-eps)*probs + eps*noise
                

                policy_list = list(zip(valid_moves, probs))

                # t2_3 = time.perf_counter()

                # print(f"Model time: {t2_3-t2}")
                

                # print(policy_list)

                node.expand(policy_list)  # Expansion
                
                # t2_4 = time.perf_counter()

                # print(f"Loop time: {t2_4-t2}")
            
            else:

                node.value = value

            # Backpropagation
            # print("Val", node.value)
            node.backpropagate(node.value)

            # t3 = time.perf_counter()

            # print(f"Search time: {t3-t1}")

        # Return visit counts
        # If number of simulations is too low, none of the root's children may be visited
        action_probs = {}
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count

        sum_values = sum(action_probs.values())

        # if learning:
        #     n_moves = torch.full((len(root.children),), 0.3)
        #     noise = self.noise_distribution(n_moves).sample()
        #     eps = 0.25
        #     # print(noise, torch.sum(noise))
            
        #     action_probs = {k: (1-eps)*v/(sum_values +1e-7) + eps*noise[i].item() for i, (k, v) in enumerate(action_probs.items())}
        
        # else:

        action_probs = {k: v / (sum_values +1e-7) for k, v in action_probs.items()}
        
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
    