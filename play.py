from chess_tensor import ChessTensor
import chess
import torch
from typing import List
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
            'num_searches': 2,
            'num_iterations': 3,
            'num_selfPlay_iterations': 500,
            'num_epochs': 4, 
            'batch_size': 64    
        }

        # CHANGGEGEEEEEEEE TODO TODO change path and map location
        model_weights = torch.load("./saves/supervised_model_15k_45.pt", map_location=torch.device('cpu'))
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
            
            # Have the model play a move here
            mcts = MCTS0(game=self.game, model=self.model, args=self.args)  # Create a new MCTS object for every turn
            action_probs = mcts.search(self.game.board, verbose=False, learning=False)
            best_move = max(action_probs, key=action_probs.get)
            self.game.move_piece(best_move)

    def get_board(self) -> chess.Board:
        """ Get the current board """
        return self.game.board

    def get_move(self) -> List[chess.Move]:
        """ Generates all possible moves """
        move = self.game.get_moves()
        return move

    def play_move(self, move: chess.Move) -> str:
        """ Allow the user to play a move """
        self.game.move_piece(move)
        
        # Have the model play a move here
        mcts = MCTS0(game=self.game, model=self.model, args=self.args)  # Create a new MCTS object for every turn
        action_probs = mcts.search(self.game.board, verbose=False, learning=False)
        best_move = max(action_probs, key=action_probs.get)
        self.game.move_piece(best_move)
        
        return str(best_move)
    
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
    
    def get_previous_board_svg(self, moves: int) -> bool:
        """ Get the old board image """
        # Make deep copy of the chess tensor
        old_board = copy.deepcopy(self.game.board)
        for _ in range(moves):
            old_board.pop()

        svg = chess.svg.board(old_board)
        with open("board.svg", "w") as f:
            f.write(svg)

        return True