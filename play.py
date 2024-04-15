from chess_tensor import ChessTensor
import chess
import copy
import chess.svg
from network import policyNN
import torch
from mcts import MCTS0

class PlayTensor():
    def __init__(self, chess960=False):

        config = dict()
        self.args = {
            'C': 2,
            'num_searches': 800,
            'num_iterations': 3,
            'num_selfPlay_iterations': 500,
            'num_epochs': 4, 
            'batch_size': 64    
        }

        model_weights = torch.load("/Users/raymondharrison/Desktop/AI/Sigma-Zero/saves/supervised_model_15k_45.pt", map_location=torch.device('cpu'))
        print(model_weights.keys())
        self.model = policyNN(config)
        self.model.load_state_dict(model_weights)

        # self.model = None # TODO: Load model here

    def start_new_game(self, chess960=False, color=chess.WHITE):
        """ Restart a game from the start """
        if color == chess.WHITE:
            self.game = ChessTensor(chess960)
        else:
            self.game = ChessTensor(chess960)
            self.__generate_move()

    def get_move(self) -> list[chess.Move]:
        """ Generates all possible moves """
        move = self.game.get_moves()
        return move

    def play_move(self, move: chess.Move):
        """ Allow the user to play a move """
        self.game.move_piece(move)
        move = self.__generate_move()
        # NOTE: Depends how smooth you want the GUI gameplay to be
        return move
        
    def __generate_move(self) -> None:
        """ Let the model play moves """
        move = self.get_move()[0]

        # # Have the model play a move here
        mcts = MCTS0(game=self.game, model=self.model, args=self.args)  # Create a new MCTS object for every turn

        action_probs = mcts.search(self.game.board, verbose=False, learning=False)
        best_move = max(action_probs, key=action_probs.get)

        self.game.move_piece(best_move)

        return move
    
    def check_if_end(self) -> bool:
        """ Check if the game has ended """
        if self.game.board.outcome():
            return self.game.board.outcome().winner # NOTE: None (Draw), chess.White, chess.Black
        else:
            return 'game_not_over'
        
    def get_current_board_svg(self) -> bool:
        """ Get the current board image"""
        svg = chess.svg.board(self.game.board)
        with open("board.svg", "w") as f:
            f.write(svg)

        return True
    
    def get_previous_board_Svg(self, moves: int) -> bool:
        """ Get the old board image """
        # Make deep copy of the chess tensor
        old_board = copy.deepcopy(self.game)
        for _ in range(moves):
            old_board.undo_move()

        svg = chess.svg.board(old_board.board)
        with open("board.svg", "w") as f:
            f.write(svg)

        return True
        


    # def undo_move(self):
    #     """ Undo a move """
    #     # NOTE: We undo the move twice because the AI also played a move
    #     self.game.undo_move()
    #     self.game.undo_move()
        
        