import torch
import chess
import chess.pgn
import bz2
import io
from tqdm import tqdm
from chess_tensor import ChessTensor, actionsToTensor, tensorToAction

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

    for game_string in tqdm(game_strings[:10000]):

        ct = ChessTensor()

        chess_game = chess.pgn.read_game(io.StringIO(game_string.decode("utf-8")))

        result = chess_game.headers["Result"]

        if result == "1-0":
            winner = True
        elif result == "0-1":
            winner = False
        else:
            winner = None

        for move in chess_game.mainline_moves():

            #Check move tensor
            # mt, queen_promotion = actionsToTensor([move], ct.board.turn)

            # move2 = tensorToAction(mt,ct.board.turn,queen_promotion=queen_promotion)

            # assert move==move2[0]

            game_history["states"].append(ct.get_representation())
            game_history["actions"].append({move:1.0})
            game_history["rewards"].append(winner==ct.board.turn if winner is not None else 0)
            game_history["colours"].append(ct.board.turn)

            ct.move_piece(move)

    torch.save(game_history, "train_set.pt")

    return game_history

if __name__ == "__main__":

    # try:
    #     game_history = torch.load("train_set.pt")
    
    # except:

    game_history = get_games("/home/benluo/school/Sigma-Zero/games/ficsgamesdb_2019_chess2000_nomovetimes_387382.pgn.bz2")