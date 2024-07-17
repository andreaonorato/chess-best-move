import cv2
import numpy as np
from tensorflow.keras.models import load_model
import chess
import chess.engine
import os

# Preprocess image
def preprocess_image(image_path):
    """
    Load an image from file and preprocess it for further processing.
    Args:
    - image_path (str): Path to the image file.
    
    Returns:
    - image (ndarray): The original image in color.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")
    return image

# Segment board into squares
def segment_board(image):
    """
    Segment the chessboard image into 64 squares.
    Args:
    - image (ndarray): The original image.
    
    Returns:
    - squares (list): List of 64 segmented square images.
    """
    board_size = image.shape[0]
    square_size = board_size // 8
    squares = []
    for row in range(8):
        for col in range(8):
            x = col * square_size
            y = row * square_size
            square = image[y:y+square_size, x:x+square_size]
            squares.append(square)
    return squares

# Save image with a specific filename
def save_image(image, filename):
    """
    Save an image to disk with a given filename.
    Args:
    - image (ndarray): The image to save.
    - filename (str): The filename to save the image as.
    """
    cv2.imwrite(filename, image)

# Recognize piece in each square and save image
def recognize_and_save_piece(square_image, model, piece_labels, index):
    """
    Recognize the chess piece in a given square image and save the image with a filename corresponding to the piece label.
    Args:
    - square_image (ndarray): Image of a single chess square.
    - model (keras.Model): The trained model for piece recognition.
    - piece_labels (dict): Dictionary mapping class indices to piece labels.
    - index (int): Index of the square for naming.
    
    Returns:
    - piece_label (str): The label of the recognized piece.
    """
    img = cv2.resize(square_image, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    piece_label = piece_labels.get(str(predicted_class), 'empty')
    
    # Save the square image with the piece label as the filename
    save_image(square_image, f'predicted_piece_{index}_{piece_label}.png')
    
    return piece_label

# Convert board state to FEN
def board_state_to_fen(board_state):
    """
    Convert the board state to Forsyth-Edwards Notation (FEN).
    Args:
    - board_state (list): List of piece labels representing the board state.
    
    Returns:
    - fen (str): The FEN string representing the board state.
    """
    fen_rows = []
    for i in range(0, 64, 8):
        row = board_state[i:i+8]
        fen_row = ''
        empty_count = 0
        for piece in row:
            if piece == 'empty':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return '/'.join(fen_rows)

# Suggest best move using Stockfish
def suggest_best_move(fen, engine_path):
    """
    Suggest the best move using the Stockfish chess engine.
    Args:
    - fen (str): The FEN string representing the board state.
    - engine_path (str): Path to the Stockfish executable.
    
    Returns:
    - best_move (chess.Move): The best move suggested by Stockfish.
    """
    board = chess.Board(fen)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    result = engine.play(board, chess.engine.Limit(time=0.1))
    engine.quit()
    return result.move

# Load the trained model
model = load_model('chess_piece_recognizer.h5')

# Define your class labels (make sure this matches the labels used in training)
piece_labels = {
    '0': 'bb',       # Black bishop
    '1': 'bk',       # Black king
    '2': 'bn',       # Black knight
    '3': 'bp',       # Black pawn
    '4': 'bq',       # Black queen
    '5': 'br',       # Black rook
    '6': 'empty',    # Empty square
    '7': 'wb',       # White bishop
    '8': 'wk',       # White king
    '9': 'wn',       # White knight
    '10': 'wp',      # White pawn
    '11': 'wq',      # White queen
    '12': 'wr'       # White rook
}

# Main function
def main(image_path, engine_path):
    """
    Main function to run the piece recognition and move suggestion.
    Args:
    - image_path (str): Path to the chessboard image file.
    - engine_path (str): Path to the Stockfish executable.
    """
    try:
        # Preprocess the image to get original image
        original_image = preprocess_image(image_path)
        
        # Segment the board into 64 squares
        squares = segment_board(original_image)
        
        board_state = []  # List to store the recognized piece labels
        for index, square in enumerate(squares):
            # Recognize the piece in each square and save the image
            predicted_piece = recognize_and_save_piece(square, model, piece_labels, index)
            # Append the recognized piece to the board state
            board_state.append(predicted_piece)
        
        # Convert the board state to FEN notation
        board_fen = board_state_to_fen(board_state)
        print("Recognized FEN:", board_fen)  # Print the FEN string
        
        # Suggest the best move based on the FEN notation
        best_move = suggest_best_move(board_fen, engine_path)
        print("Suggested Best Move:", best_move)  # Print the best move

    except Exception as e:
        print(f"An error occurred: {e}")  # Print any error messages

# Run the main function
image_path = "chesscom_board2.png"  # Path to the chessboard image
engine_path = r'C:\Users\USER\workspace\python\personal_projects\AI_Chess_cheater\stockfish\stockfish-windows-x86-64-sse41-popcnt.exe'  # Path to Stockfish engine
main(image_path, engine_path)  # Execute the main function