"""
TODO: Implement self-play to generate one batch of training data
"""
import chess
import numpy as np
from mcts import MCTS0
from network import policyNN
from chess_tensor import ChessTensor
import torch

def initialize_empty_boards():
    # Initialize with 7 empty boards
    empty_boards = [torch.zeros((119, 8, 8)) for _ in range(7)]
    return empty_boards

def play_game(model):
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
        mcts = MCTS0(model=model)  # Create a new MCTS object for every turn

        # Convert the current board state to a tensor
        state_tensor = chess_tensor.get_representation()
        
        # Use MCTS to determine the best move
        # This function might need to be aware of whose turn it is
        move = mcts.search(state_tensor) ############ IDK IF THIS FUNCTION ARGUMENT RECEIVES THE STATE TENSOR OR THE BOARD
        
        # Save the state and action
        game_history['states'].append(state_tensor)
        game_history['actions'].append(move)
        
        # Apply the move
        board.push(move)
        
        # Update the chess tensor with the new move
        chess_tensor.move_piece(move)

def generate_mini_batches(game_history, result):
    mini_batches = []
    total_moves = len(game_history['actions'])
    
    # Initialize empty boards for padding
    empty_boards = initialize_empty_boards()

    for start_index in range(total_moves):
        end_index = start_index + 8  # Define the window for 8 half-turns
        
        if end_index > total_moves:  # Ensure we don't go beyond the game length
            break
        
        # Calculate how many empty boards we need to prepend
        num_empty_boards_needed = 8 - (end_index - start_index)
        states_slice = game_history['states'][start_index:end_index]
        
        # Prepend the empty boards to the states_slice
        padded_states = empty_boards[:num_empty_boards_needed] + states_slice
        
        # Determine the reward based on the game outcome
        reward = 0
        if result == '1-0':  # White wins
            reward = 1 if end_index % 2 == 0 else -1
        elif result == '0-1':  # Black wins
            reward = -1 if end_index % 2 == 0 else 1
        # Draw case is already handled with reward = 0
        
        mini_batches.append({
            'states': padded_states,
            'reward': reward
        })
    
    return mini_batches

def generate_training_data(model, num_games=100):
    training_data = []
    
    for _ in range(num_games):
        game_history = play_game(model)
        result = game_history['board'].result()  # This should be corrected to access the result properly
        mini_batches = generate_mini_batches(game_history, result)
        training_data.extend(mini_batches)
    
    return training_data

if __name__ == "__main__":
    model = policyNN(config=None)  # Placeholder for model initialization
    
    training_data = generate_training_data(model, num_games=100)
    # Save or use the training data as needed