"""
TODO: Implement self-play to generate one batch of training data
"""
import chess
import chess.engine
import numpy as np
from mcts import MCTS0
from network import policyNN
from chess_tensor import ChessTensor, actionsToTensor, tensorToAction
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

@torch.no_grad()
def play_game(model, args):
    """
    Test AI model by playing against it or playing StockFish against it.
    """
    board = chess.Board()
    chess_tensor = ChessTensor()

    model.eval()
    
    while not board.is_game_over():

        if board.turn:

            #####Uncomment for user play#####

            # move = input("Please enter move: ")

            # best_move = chess.Move.from_uci(move.strip())

            #####Uncomment for SF AI play#####

            engine = chess.engine.SimpleEngine.popen_uci("/home/benluo/school/Sigma-Zero/stockfish/stockfish-ubuntu-x86-64-avx2")
            
            engine.configure({"Skill Level": 4})

            limit = chess.engine.Limit(time=2)

            result = engine.play(board, limit)

            best_move = result.move
        
        else:

            t1 = time.perf_counter()

            mcts = MCTS0(game=chess_tensor, model=model, args=args)  # Create a new MCTS object for every turn
        
            action_probs = mcts.search(board, verbose=False, learning=False)

            # print(action_probs)

            t2 = time.perf_counter()

            best_move = max(action_probs, key=action_probs.get)

            print("Search Time", t2-t1)

        print("Board", "White" if chess_tensor.board.turn else "Black", model(chess_tensor.get_representation().float().unsqueeze(0).to(device))[-1].item())

        print(chess_tensor.board)

        board.push(best_move)
        
        # Update the chess tensor with the new move
        chess_tensor.move_piece(best_move)

        
        #debug code
        # print(f"Init time: {t2-t1}\nSearch time: {t3-t2}\nMove time: {t4-t3}\nLoop time: {t4-t1}")
        
    print(board)

    # Determine the game result and assign rewards
    result = board.result()
    print("Result", result)
    
    return result

if __name__ == "__main__":

    config = dict()
    args = {
        'C': 2,
        'num_searches': 800,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4, 
        'batch_size': 64
    }

    model_weights = torch.load("/home/benluo/school/Sigma-Zero/saves/supervised_model_max_best_hlr.pt")

    model = policyNN(config).to(device)

    ct = ChessTensor()

    model.load_state_dict(model_weights)
    
    play_game(model=model, args=args)
