"""
TODO: Implement chess eval by limiting time, depth, skill according to online source for ELO estimate
"""
import chess
import chess.engine
import numpy as np
from mcts import MCTS0
from network import policyNN
from chess_tensor import ChessTensor
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
    Simulates a single game of chess, with the AI playing against itself.
    Returns the game history, including states, actions, and rewards.
    """

    # modified settings from Lichess fishnet distribution skill levels
    # level_num: (skill_level, time_limit, depth_limit)  
    # 2: (3, 0.02, 5) lost to this initial setting roughly 1.5k elo
    # 
    level = {1: (0, 0.01, 1),
             2: (1, 0.02, 2),
             3: (2, 0.03, 3),
             4: (3, 0.04, 4),
             5: (4, 0.05, 5),
             6: (5, 0.06, 6),
             7: (6, 0.07, 7),
             8: (20, 10, 50)}

    level_reached = 1
    while True:
        board = chess.Board()
        chess_tensor = ChessTensor()

        stockfish_path = "/home/benluo/school/Sigma-Zero/stockfish/stockfish-ubuntu-x86-64-avx2"
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        skill_level = level[level_reached][0]
        time_limit = level[level_reached][1]
        depth_limit = level[level_reached][2]      
        
        engine.configure({"Skill Level": skill_level})
 
        while not board.is_game_over():

            print(board)

            if board.turn:

                # move = input("Please enter move: ")

                # best_move = chess.Move.from_uci(move.strip())

                limit = chess.engine.Limit(time=time_limit, depth=depth_limit)

                result = engine.play(board, limit)

                best_move = result.move

                time.sleep(1)
            
            else:

                t1 = time.perf_counter()

                mcts = MCTS0(game=chess_tensor, model=model, args=args)  # Create a new MCTS object for every turn
            
                action_probs = mcts.search(board, verbose=False, learning=False)

                print(action_probs)

                t2 = time.perf_counter()

                best_move = max(action_probs, key=action_probs.get)

                print("Search Time", t2-t1)

            print("Board", chess_tensor.board.turn)

            board.push(best_move)
            
            # Update the chess tensor with the new move
            chess_tensor.move_piece(best_move)

            
            #debug code
            # print(f"Init time: {t2-t1}\nSearch time: {t3-t2}\nMove time: {t4-t3}\nLoop time: {t4-t1}")
            
        print(board)

        # Determine the game result and assign rewards
        result = board.result()
        print("Result", result)

        if result == "1-0":
            break
        elif result == "1/2-1/2":
            engine.quit()
        else:
            level_reached += 1
            engine.quit()
    engine.quit()

    return level_reached

if __name__ == "__main__":

    config = dict()
    args = {
        'C': 5,
        'num_searches': 800,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4, 
        'batch_size': 64
    }

    model_weights = torch.load("/home/benluo/school/Sigma-Zero/saves/supervised_model_15k_45.pt")

    model = policyNN(config).to(device)

    ct = ChessTensor()

    model.load_state_dict(model_weights)
    
    level_reached = play_game(model=model, args=args)

    print(f"Model reached level {level_reached}")
