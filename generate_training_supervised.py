import torch
import chess
import chess.pgn
import bz2
import io
from tqdm import tqdm
from chess_tensor import ChessTensor
from time import sleep
import tracemalloc

@torch.no_grad()
def get_games(game_strings=None, num_games=0, return_dict=None):

    #Win-share statistics
    results = {
        "1-0":0,
        "0-1":0,
        "1/2-1/2":0,
    }

    games_history = {
        "states":[],
        "actions":[],
        "rewards":[],
        "colours":[],
    }

    #Counter for number of games
    num_white = 0
    num_black = 0
    num_draw = 0

    indexes = torch.arange(8).unsqueeze(0).unsqueeze(0)

    # Scanning results
    print("Scanning results")
    for game_string in tqdm(game_strings):
        chess_game = chess.pgn.read_game(io.StringIO(game_string.decode("utf-8")))
        
        if chess_game is not None:
            result = chess_game.headers["Result"]
            results[result] += 1
        
        if results["1/2-1/2"]+min(results["0-1"],results["1-0"])*2>=num_games and num_games!=0:
            break
    num_draw = results["1/2-1/2"]
    num_black = min(results["0-1"],results["1-0"])
    num_white = min(results["0-1"],results["1-0"])

    print("White", num_white, "Black", num_black, "Draw", num_draw)

    ct = ChessTensor()

    #Building dataset
    for game_string in tqdm(game_strings):  
        
        ct.start_board(chess960=False)

        chess_game = chess.pgn.read_game(io.StringIO(game_string.decode("utf-8")))

        if chess_game is None:
            continue

        #Ignore computer-played games
        # if "BlackIsComp" in chess_game.headers and chess_game.headers["BlackIsComp"] == "Yes":
        #     continue
            
        # if "WhiteIsComp" in chess_game.headers and chess_game.headers["WhiteIsComp"] == "Yes":
        #     continue

        result = chess_game.headers["Result"]

        if (num_white==0 and result=="1-0") or (num_black==0 and result=="0-1") or (num_draw==0 and result=="1/2-1/2"):
            continue

        play_count = int(chess_game.headers["PlyCount"])

        if result == "1-0":
            games_history["rewards"] += [-1 if i%2 else 1 for i in range(play_count)]
        elif result == "0-1":
            games_history["rewards"] += [1 if i%2 else -1 for i in range(play_count)]
        else:
            games_history["rewards"] += [0]*play_count

        for move in chess_game.mainline_moves():

            representation = ct.get_representation()

            #Compression

            compressed_representation = (representation.byte() << indexes).sum(dim=-1).to(dtype=torch.uint8)

            games_history["states"].append(compressed_representation)
            games_history["actions"].append([move])
            games_history["colours"].append(ct.board.turn)

            ct.move_piece(move)

        #Check if game has ended definitively
        # if ct.board.is_game_over():
        # for key in game_history:
        #     games_history[key] += game_history[key]
        if result == "1-0":
            num_white -= 1
        elif result == "0-1":
            num_black -= 1
        else:
            num_draw -= 1

        if num_white+num_black+num_draw == 0:
            break

        tracemalloc.get_traced_memory()

    torch.save(games_history, f"game_data_{num_games}.pt")


if __name__ == "__main__":

    #File path to pgn.bz2 file
    file_path = "/home/benluo/school/Sigma-Zero/games/ficsgamesdb_2015_standard2000_nomovetimes_390037.pgn.bz2"

    pgn_games = bz2.BZ2File(file_path, "r")

    game_strings = [b""]
    
    print("Reading from", file_path)
    for line in tqdm(pgn_games):
        if  line == b"\n":
            game_strings.append(b"")
        game_strings[-1] += line

    total_games = len(game_strings)
    
    print("Total number of games", total_games)

    #Number of games to generate in total (>0)
    #0 means the entire pgn file will be generated
    num_games = 60000

    tracemalloc.start()

    get_games(game_strings, num_games)

    tracemalloc.stop()
