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
import json

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
    # 2: (3, 0.02, 5) old model lost to this initial setting roughly 1.5k elo
    # 
    level = {1: (3, 1, 5),
             2: (4, 1, 5),
             3: (5, 1, 5),
             4: (6, 1, 5),
             5: (7, 1, 5),
             6: (8, 1, 6),
             7: (9, 2, 7),
             8: (20, 10, 50)}

    level_reached = 1
    while True:
        # track the score for this level
        model_score = 0
        for idx in range(5):  # 5 matches
            chess_tensor = ChessTensor(chess960=args['chess960'])
            board = chess_tensor.board

        stockfish_path = "/home/benluo/school/Sigma-Zero/stockfish/stockfish-ubuntu-x86-64-avx2"
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

            skill_level = level[level_reached][0]
            time_limit = level[level_reached][1]
            depth_limit = level[level_reached][2]      
            
            engine.configure({"Skill Level": skill_level})

            turn = 0

            if idx % 2 == 0:
                sf_white = True
            else:
                sf_white = False
            while not board.is_game_over():

                print(board)

                if (turn % 2 == 0 and sf_white) or (turn % 2 == 1 and not sf_white):

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
                
                # Update the chess tensor with the new move
                chess_tensor.move_piece(best_move)

                
                #debug code
                # print(f"Init time: {t2-t1}\nSearch time: {t3-t2}\nMove time: {t4-t3}\nLoop time: {t4-t1}")

                turn += 1

            print(board)

            # Determine the game result and assign rewards
            result = board.result()
            print("Result", result)

            if (result == "0-1" and sf_white) or (result == "1-0" and not sf_white):
                model_score += 1
        
            elif result == "1/2-1/2":
                model_score += 0.5

            engine.quit()

            with open("./logs/log.txt", "a") as f:
                f.write(f"Model at level {level_reached}, score {model_score} at iteration {idx}\n")

            if model_score > 2.5:  # model won 2.5 points on stockfish, proceed to next level
                break

        if model_score <= 2.5:
            break
        else:
            level_reached += 1
            print("Model at level", level_reached)
            with open("./logs/log.txt", "a") as f:
                f.write(f"\n")

    return level_reached

if __name__ == "__main__":

    config = dict()
    args = {
        'C': 5,
        'num_searches': 800,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4, 
        'batch_size': 64,
        'chess960': True
    } 
    model_path = "./supervised_model_15k_45.pt"

    start_time = time.time()

    with open("./logs/log.txt", "a") as f:
        f.write(f"\n-------------------------------------------------\n")
        f.write(f"\nModel {model_path}\nStart time:{start_time}\n{json.dumps(args, indent=2)}\n")

    model_weights = torch.load(model_path)

    model = policyNN(config).to(device)

    model.load_state_dict(model_weights)
    
    level_reached = play_game(model=model, args=args)

    with open("./logs/log.txt", "a") as f:
        f.write(f"\nElapsed time: {time.time() - start_time}\n")

    print(f"Model reached level {level_reached}")
