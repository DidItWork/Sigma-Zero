from chess_tensor import ChessTensor
import chess

class PlayTensor():
    def __init__(self, chess960=False):
        self.model = None # TODO: Load model here

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
        self.__generate_move()
        # NOTE: Depends how smooth you want the GUI gameplay to be
        
    def __generate_move(self):
        """ Let the model play moves """
        # Have the model play a move here
        tensor = self.game.get_representation()
        move = self.model(tensor) # How do we run inference here? 
        self.game.move_piece(move)
    
    def check_if_end(self) -> bool:
        """ Check if the game has ended """
        if self.game.board.outcome():
            return self.game.board.outcome().winner # NOTE: None (Draw), chess.White, chess.Black
        else:
            return False
    
    def undo_move(self):
        """ Undo a move """
        # NOTE: We undo the move twice because the AI also played a move
        self.game.undo_move()
        self.game.undo_move()
        

        
        