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

def play_game(model, args):
    """
    Simulates a single game of chess, with the AI playing against itself.
    Returns the game history, including states, actions, and rewards.
    """
    board = chess.Board()
    chess_tensor = ChessTensor()

    game_history = {
        'states': [],
        'actions': [],
        'rewards': [] # Rewards will be assigned later based on the game outcome
    }
    
    while not board.is_game_over():

        # print("ID of process running worker1: {}".format(os.getpid()))
        # print(board)

        t1 = time.perf_counter()

        mcts = MCTS0(game=chess_tensor, model=model, args=args)  # Create a new MCTS object for every turn
    
        # Convert the current board state to a tensor
        state_tensor = chess_tensor.get_representation()

        t2 = time.perf_counter()
        
        # Use MCTS to determine the best move
        # This function might need to be aware of whose turn it is
        # move = mcts.search(board) ############ IDK IF THIS FUNCTION ARGUMENT RECEIVES THE STATE TENSOR OR THE BOARD
        action_probs = mcts.search(board, verbose=False)

        t3 = time.perf_counter()

        best_move = max(action_probs, key=action_probs.get)

        action_tensor = actionsToTensor(action_probs, color=board.turn)[0]

        action_tensor.requires_grad = False

        # Save the state and action
        game_history['states'].append(state_tensor)
        game_history['actions'].append(action_tensor) ### SHOULD THIS BE THE MOVE OR THE BEST_MOVE?
        
        # Apply the move
        board.push(best_move)
        
        # Update the chess tensor with the new move
        chess_tensor.move_piece(best_move)

        t4 = time.perf_counter()

        #debug code
        # print(f"Init time: {t2-t1}\nSearch time: {t3-t2}\nMove time: {t4-t3}\nLoop time: {t4-t1}")
        
    print(board)

    # Determine the game result and assign rewards
    result = board.result()
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

# def generate_mini_batches(game_history, result):
#     mini_batches = []
#     total_moves = len(game_history['actions'])
    
#     # Initialize empty boards for padding
#     empty_boards = initialize_empty_boards()

#     for start_index in range(total_moves):
#         end_index = start_index + 8  # Define the window for 8 half-turns
        
#         if end_index > total_moves:  # Ensure we don't go beyond the game length
#             break
        
#         # Calculate how many empty boards we need to prepend
#         num_empty_boards_needed = 8 - (end_index - start_index)
#         states_slice = game_history['states'][start_index:end_index]
        
#         # Prepend the empty boards to the states_slice
#         padded_states = empty_boards[:num_empty_boards_needed] + states_slice
        
#         # Determine the reward based on the game outcome
#         reward = 0
#         if result == '1-0':  # White wins
#             reward = 1 if end_index % 2 == 0 else -1
#         elif result == '0-1':  # Black wins
#             reward = -1 if end_index % 2 == 0 else 1
#         # Draw case is already handled with reward = 0
        
#         mini_batches.append({
#             'states': padded_states,
#             'reward': reward
#         })
    
#     return mini_batches

def generate_training_data(model, num_games=1, args=None, return_dict=None):
    # training_data = []

    games_history = {
        'states': [],
        'actions': [],
        'rewards': [] # Rewards will be assigned later based on the game outcome
    }
    
    # for _ in range(num_games):
    #     game_history = play_game(model)
    #     result = game_history['board'].result()  # This should be corrected to access the result properly
    #     mini_batches = generate_mini_batches(game_history, result)
    #     training_data.extend(mini_batches)

    for i in range(num_games):
        print(f"Game {i+1}/{num_games}")
        game_history = play_game(model, args)
        # training_data.append(game_history)

        # # Print the game history for debugging
        # print('game number:', _)

        # print(len(game_history['states']))
        # print(len(game_history['actions']))
        # print(len(game_history['rewards']))
        # print(game_history['actions'])
        # print(game_history['rewards'])

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
    
    training_data = generate_training_data(model, num_games=1, args=args)
    # print(device)
    # Save or use the training data as needed