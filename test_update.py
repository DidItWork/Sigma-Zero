import chess
import torch
import math

from mcts import MCTS0
from network import policyNN
from play import PlayTensor

device = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    def __init__(self, network, game, args):
        self.network = network
        self.game = game
        self.args = args


    def get_best_move(self, board):
        mcts = MCTS0(game=self.game, model=self.network, args=self.args)  # Create a new MCTS object for every turn
        action_probs = mcts.search(board, verbose=False)
        best_move = max(action_probs, key=action_probs.get)
        return best_move


def update_model(current_model=None, new_model=None, matches=1) -> bool:

    network = policyNN(config=dict()).to(device)
    play = PlayTensor()

    # Load current model
    if current_model:
        network.load_state_dict(torch.load(current_model), strict=False)
    network.eval()

    # Load new_model
    new_network = policyNN(config=dict()).to(device)
    if new_model:
        new_network.load_state_dict(torch.load(new_model), strict=False)
    else:
        return False
    new_network.eval()

    new_model_wins = 0
    args = {
        'C': 2,
        'num_searches': 800,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4, 
        'batch_size': 64
    }

    # Play against new_model
    for _ in range(matches):
        for colour in range(2):
            play.start_new_game()
            if colour == 0:
                play.model = Agent(network, play.game, args)
                agent = Agent(new_network, play.game, args)
            else:
                play.model = Agent(new_network, play.game, args)
                agent = Agent(network, play.game, args)

            while play.check_if_end() is False:
                board = play.get_board()
                print("---------------")
                print(board)
                best_move = agent.get_best_move(board)
                play.play_move(best_move)
            
            if play.check_if_end == chess.WHITE and colour == 1:
                new_model_wins += 1
            elif play.check_if_end == chess.BLACK and colour == 0:
                new_model_wins += 1
            else:  # draw
                new_model_wins += 0.5

            print("new model wins", new_model_wins)
            if new_model_wins >=  math.ceil(0.55*matches):  # update model if performance is better
                torch.save(new_network.state_dict(), current_model)
                return True
    return False


def main():
    base = "./test3.pt"  # old trained weights
    new = "./supervised_model.pt"  # new trained weights, to be checked

    if update_model(current_model=base, new_model=new, matches=3):
        print(f"Model {base} updated")
    else:
        print("Model not updated")


if __name__ == "__main__":
    main()
