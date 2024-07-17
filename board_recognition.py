from flask import Flask, request, render_template, send_file, send_from_directory, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import chess
import chess.engine
import os
import io

app = Flask(__name__)

# Preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")
    return image

# Segment board into squares
def segment_board(image):
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

# Draw a highlight on the chessboard image
def highlight_move(image, move, square_size):
    image_with_highlight = image.copy()
    from_square = move.from_square
    to_square = move.to_square
    from_row, from_col = divmod(from_square, 8)
    from_row = 7 - from_row
    to_row, to_col = divmod(to_square, 8)
    to_row = 7 - to_row
    highlight_color = (0, 255, 0)
    thickness = 2
    start_x = from_col * square_size
    start_y = from_row * square_size
    end_x = start_x + square_size
    end_y = start_y + square_size
    cv2.rectangle(image_with_highlight, (start_x, start_y), (end_x, end_y), highlight_color, thickness)
    start_x = to_col * square_size
    start_y = to_row * square_size
    end_x = start_x + square_size
    end_y = start_y + square_size
    cv2.rectangle(image_with_highlight, (start_x, start_y), (end_x, end_y), highlight_color, thickness)
    return image_with_highlight

# Save image with a specific filename
def save_image(image, filename):
    cv2.imwrite(filename, image)

# Recognize piece in each square
def recognize_and_save_piece(square_image, model, piece_labels, index):
    img = cv2.resize(square_image, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    piece_label = piece_labels.get(str(predicted_class), 'empty')
    return piece_label

def board_state_to_fen(board_state, custom_to_fen_map, is_white_turn):
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
                fen_row += custom_to_fen_map[piece]
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    fen = '/'.join(fen_rows)
    fen += ' ' + ('w' if is_white_turn else 'b')
    fen += ' KQkq - 0 1'
    return fen

def suggest_best_move(fen, engine_path):
    board = chess.Board(fen)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    result = engine.play(board, chess.engine.Limit(time=0.1))
    engine.quit()
    return result.move

# Load the trained model
model = load_model('chess_piece_recognizer.h5')

# Define your class labels (make sure this matches the labels used in training)
piece_labels = {
    '0': 'bb', '1': 'bk', '2': 'bn', '3': 'bp', '4': 'bq', '5': 'br',
    '6': 'empty', '7': 'wb', '8': 'wk', '9': 'wn', '10': 'wp', '11': 'wq', '12': 'wr'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        turn = request.form.get('turn', 'white')
        is_white_turn = (turn == 'white')
        if file:
            image_path = 'uploaded_image.png'
            file.save(image_path)
            original_image = preprocess_image(image_path)
            squares = segment_board(original_image)
            board_state = []
            for index, square in enumerate(squares):
                predicted_piece = recognize_and_save_piece(square, model, piece_labels, index)
                board_state.append(predicted_piece)
            custom_to_fen_map = {
                'bb': 'b', 'bk': 'k', 'bn': 'n', 'bp': 'p', 'bq': 'q', 'br': 'r',
                'wb': 'B', 'wk': 'K', 'wn': 'N', 'wp': 'P', 'wq': 'Q', 'wr': 'R', 'empty': '1'
            }
            board_fen = board_state_to_fen(board_state, custom_to_fen_map, is_white_turn=is_white_turn)
            best_move = suggest_best_move(board_fen, 'stockfish/stockfish-windows-x86-64-sse41-popcnt.exe')
            square_size = original_image.shape[0] // 8
            image_with_highlight = highlight_move(original_image, best_move, square_size)
            result_image_path = 'highlighted_move.png'
            save_image(image_with_highlight, result_image_path)
            return redirect(url_for('display_image', filename=result_image_path))

@app.route('/image/<filename>')
def display_image(filename):
    return render_template('display_image.html', image_url=url_for('serve_image', filename=filename))

@app.route('/serve_image/<filename>')
def serve_image(filename):
    return send_file(filename, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
