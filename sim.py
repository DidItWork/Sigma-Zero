"""
TODO: Implement self-play to generate one batch of training data
"""
import chess
import numpy as np
from mcts import MCTS0
from network import policyNN
from chess_tensor import ChessTensor, actionsToTensor
import torch
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# FOR MAC SILICON GPU!!!!!!!!
# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#             "built with MPS enabled.")
#     else:
#         print("MPS not available because the current MacOS version is not 12.3+ "
#             "and/or you do not have an MPS-enabled device on this machine.")
# else:
#     device = torch.device("mps")

def initialize_empty_boards():
    # Initialize with 7 empty boards
    empty_boards = [torch.zeros((119, 8, 8)) for _ in range(7)]
    return empty_boards

def play_game(model, args, c960=False):
    """
    Simulates a single game of chess, with the AI playing against itself.
    Returns the game history, including states, actions, and rewards.
    """
    chess_tensor = ChessTensor(chess960=c960)
    print(chess_tensor.board)

    game_history = {
        'states': [],
        'actions': [],
        'rewards': [], # Rewards will be assigned later based on the game outcome
        'colours': [],
    }
    
    while not chess_tensor.board.is_game_over():

        # print(chess_tensor.board)

        t1 = time.perf_counter()

        # Create a new MCTS object for every turn
        mcts = MCTS0(game=chess_tensor, model=model, args=args)
    
        # Convert the current board state to a tensor
        state_tensor = chess_tensor.get_representation()

        state_tensor.requires_grad = False

        t2 = time.perf_counter()
        
        # Use MCTS to determine the best move
        action_probs = mcts.search(chess_tensor.board, verbose=False, learning=True)

        t3 = time.perf_counter()

        #Random Sampling proportional to probs in learning
        best_move = np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))

        # Save the state and action
        game_history['states'].append(state_tensor)
        game_history['actions'].append(action_probs)
        game_history["colours"].append(chess_tensor.board.turn)
        
        # Update the chess tensor with the new move
        chess_tensor.move_piece(best_move)

        t4 = time.perf_counter()

        #debug code
        # print(f"Init time: {t2-t1}\nSearch time: {t3-t2}\nMove time: {t4-t3}\nLoop time: {t4-t1}")
        
    print(chess_tensor.board)

    # Determine the game result and assign rewards
    result = chess_tensor.board.result()
    print("Result", result)
    reward = 0
    if result == '1-0':  # White wins
        reward = 1
    elif result == '0-1':  # Black wins
        reward = -1
    # Draw case is already handled with reward = 0
        
    # Assign rewards based on the player's turn
    for i in range(len(game_history['actions'])):
        game_history['rewards'].append(reward if i % 2 == 0 else -reward)

    return game_history


def generate_training_data(model, num_games=1, args=None, return_dict=None, c960=False):
    # training_data = []

    games_history = {
        'states': [],
        'actions': [],
        'rewards': [], # Rewards will be assigned later based on the game outcome
        'colours': [],
    }

    for i in range(num_games):
        print(f"Game {i+1}/{num_games}")
        game_history = play_game(model, args, c960=c960)

        for key,value in game_history.items():

            games_history[key] += value

    if return_dict is not None:
        return_dict[os.getpid()]=games_history

    return games_history

if __name__ == "__main__":

    config = dict()
    args = {
        'C': 2,
        'num_searches': 10,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 64
    }
    model = policyNN(config).to(device)
    
    training_data = generate_training_data(model, 1, args, None, True)