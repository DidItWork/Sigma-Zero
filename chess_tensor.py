import chess
import torch


"""
MT + L

M = 6 + 6 + 2
6 pieces of each color
2 planes of repetition
    1 for 1 repetition within T moves
    1 for 2 repetitions within T moves
    The whole plane should be shown as 1 if the repetition happens
    
T = 8

L = 7
    1 for colour
    1 for Total move count
    1 for P1 castling
    1 for P2 castling
    1 for no-progress count
"""

class ChessTensor():
    def __init__(self):
        self.M = 14
        self.T = 8
        self.L = 7
        self.__start_board()

    # This is to get a single tensor representation of the board
    def __board_to_tensor(self) -> torch.Tensor:
        order = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        # 6 white + 6 black
        representation = torch.zeros(12, 8, 8)   

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                # Determine the value to assign (positive for white, negative for black)
                channel = order[piece.piece_type]
                if piece.color == chess.BLACK:
                    channel += 6

                row, col = divmod(square, 8)
                representation[channel, row, col] = 1

        return representation

    def __start_board(self):
        self.board = chess.Board()

        # Get board current state
        board_tensor = self.__board_to_tensor()
        repetition_tensor = torch.zeros(2, 8, 8)
        current_representation = torch.cat([board_tensor, repetition_tensor], 0)

        # print(current_representation)

        # Adding L channel
        color = torch.zeros(1, 8, 8)
        total_moves = torch.ones(1, 8, 8)
        white_king_castling = torch.ones(1, 8, 8)
        white_queen_castling = torch.ones(1, 8, 8)
        black_king_castling = torch.ones(1, 8, 8)
        black_queen_castling = torch.ones(1, 8, 8)
        no_progress = torch.zeros(1, 8, 8)
        L_tensor = torch.cat([color, total_moves, white_king_castling, white_queen_castling, black_king_castling, black_queen_castling, no_progress], 0)

        self.representation = torch.cat([current_representation, torch.zeros(self.M * (self.T - 1), 8, 8), L_tensor], 0)
    
    def move_piece(self, move: chess.Move) -> torch.Tensor:
        # Moving the board forward
        self.board.push(move)

        # Get board current state
        board_tensor = self.__board_to_tensor()

        # Add repetition tensor
        repetition_1 = False
        repetition_2 = False
        for i in range(0, 112, 14):
            repetition_1 = torch.all(board_tensor == self.representation[i: i + 12]) or repetition_1
            repetition_2 = (torch.all(board_tensor == self.representation[i: i + 12]) or repetition_2) and repetition_1

        repetition_1_tensor = torch.ones(1, 8, 8) if repetition_1 else torch.zeros(1, 8, 8)
        repetition_2_tensor = torch.ones(1, 8, 8) if repetition_2 else torch.zeros(1, 8, 8)
        
        # Add in board tensor and remove oldest updates
        start_channel = (self.M - 1) * self.T
        end_channel = start_channel + self.M
        self.representation = torch.cat([board_tensor, repetition_1_tensor, repetition_2_tensor, self.representation[:start_channel], self.representation[end_channel:]], 0)

        # Adding L channel
        color = 1 - self.representation[-7]
        color = color.reshape(1, 8, 8).expand(1, 8, 8)
        total_moves = self.representation[-6] + 1
        total_moves = total_moves.reshape(1, 8, 8).expand(1, 8, 8)
        white_king_castling = torch.ones(1, 8, 8) if self.board.has_kingside_castling_rights(chess.WHITE) else torch.zeros(1, 8, 8)
        white_queen_castling = torch.ones(1, 8, 8) if self.board.has_queenside_castling_rights(chess.WHITE) else torch.zeros(1, 8, 8)
        black_king_castling = torch.ones(1, 8, 8) if self.board.has_kingside_castling_rights(chess.BLACK) else torch.zeros(1, 8, 8)
        black_queen_castling = torch.ones(1, 8, 8) if self.board.has_queenside_castling_rights(chess.BLACK) else torch.zeros(1, 8, 8)
        no_progress = torch.Tensor([self.board.halfmove_clock]).reshape(1, 1, 1).expand(1, 8, 8)
        
        L_tensor = torch.cat([color, total_moves, white_king_castling, white_queen_castling, black_king_castling, black_queen_castling, no_progress], 0)

        # Replace L tensor at the back 7 planes
        self.representation = torch.cat([self.representation[:-self.L], L_tensor], 0)  

    def get_representation(self) -> torch.Tensor:

        # print(self.representation[0], self.representation[6], self.representation[14])
        # For white representation
        if self.representation[-7][0][0] == 0:
            return self.representation
        else:
            # Changing order of representation
            copy = self.representation.clone()

            # Swapping order of representation for P1
            for i in range(self.T):
                start_channel = i * self.M
                end_channel = start_channel + 6

                copy[start_channel: end_channel, :, :], copy[start_channel + 6: start_channel + 12, :, :] = copy[start_channel + 6: start_channel + 12, :, :].clone(), copy[start_channel: end_channel, :, :].clone()
                
            # Swapping order of representation for L
            copy[-5:-4, :, :], copy[-3:-2, :, :] = copy[-3:-2, :, :].clone(), copy[-5:-4, :, :].clone()

            # Flipping the board for black
            return torch.flip(copy, [1, 2])
        
    def get_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)
    

# chesser = ChessTensor()

# rep = chesser.get_representation()
# print(rep[0], rep[6], rep[14], rep[20])

# chesser.move_piece(chess.Move.from_uci('e2e4'))

# rep = chesser.get_representation()
# print(rep[0], rep[6], rep[14], rep[20])

