import torch
import chess
import chess.pgn
import bz2
import io
from tqdm import tqdm
from chess_tensor import ChessTensor

@torch.no_grad()
def get_games(file_path:str):

    pgn_games = bz2.BZ2File(file_path, "r")

    game_strings = [b""]
    
    print("Reading from", file_path)
    for line in tqdm(pgn_games):
        if  line == b"\n":
            game_strings.append(b"")
        game_strings[-1] += line
    
    print("Total number of games", len(game_strings))

    game_history = {
        "states":[],
        "actions":[],
        "rewards":[],
        "colours":[],
    }

    for game_string in tqdm(game_strings[:15000]):

        ct = ChessTensor()

        chess_game = chess.pgn.read_game(io.StringIO(game_string.decode("utf-8")))

        result = chess_game.headers["Result"]

        play_count = int(chess_game.headers["PlyCount"])

        if result == "1-0":
            game_history["rewards"] += [-1 if i%2 else 1 for i in range(play_count)]
        elif result == "0-1":
            game_history["rewards"] += [1 if i%2 else -1 for i in range(play_count)]
        else:
            game_history["rewards"] += [0]*play_count

        for move in chess_game.mainline_moves():

            game_history["states"].append(ct.get_representation())
            game_history["actions"].append({move:1.0})
            game_history["colours"].append(ct.board.turn)

            ct.move_piece(move)

        

    torch.save(game_history, "train_set_15k.pt")

    return game_history

if __name__ == "__main__":

    #File path to pgn.bz2 file
    file_path = ""

    game_history = get_games(file_path)
