import chess
import chess
# from ai.inputReaderl import *
import torch
import torch.nn as nn
import ai.neural_net as ann
import ai.inputReaderl as inputRead


def gpt3_eval(board: chess.Board):
    if board.is_stalemate():
        return 0

    if board.is_checkmate():
        if board.turn == chess.BLACK:
            return 10000000000
        else:
            return -10000000000
        
    # Initialize the score
    score = 0

    # Piece values
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0 
    }

    # Evaluate each square on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        piece_value = piece_values[piece.piece_type]
        
        # Assign positive or negative values based on piece color (positive for white, negative for black)
        if piece.color:
            score += piece_value
        else:
            score -= piece_value
    
    return score

# Load the neural network model and move it to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ann.SimpleNN().to(device)
checkpoint = torch.load("model_checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# Assuming neural_net_input function returns the input features for the neural network
def nn_eval(board: chess.Board):
    if board.is_stalemate():
        return 0

    if board.is_checkmate():
        if board.turn == chess.BLACK:
            return 10000000000
        else:
            return -10000000000
    
    # Evaluate each square on the board
    net_inputs = inputRead.neural_net_input(board)
    inputs = torch.tensor(net_inputs, dtype=torch.float).to(device)

    with torch.no_grad():
        output = model(inputs)
        value = output.item()  # Extract the scalar value from the tensor

    return value

