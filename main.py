import chess
import chess.engine

def print_board(board):
    print(board)

def player_move(board):
    while True:
        try:
            move = input("Enter your move (e.g., e2e4): ")
            move = chess.Move.from_uci(move)
            if move in board.legal_moves:
                return move
            else:
                print("Invalid move. Please try again.")
        except ValueError:
            print("Invalid move format. Please try again.")

def engine_move(board, engine):
    result = engine.play(board, chess.engine.Limit(time=0.1))
    return result.move

def main():
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\USER\workspace\python\personal_projects\AI_Chess_cheater\stockfish\stockfish-windows-x86-64-sse41-popcnt.exe")

    print_board(board)

    while not board.is_game_over():
        # Player move
        move = player_move(board)
        board.push(move)
        print_board(board)
        print("\n")

        if board.is_game_over():
            break

        # Engine move
        move = engine_move(board, engine)
        board.push(move)
        print_board(board)
        print("\n")

    engine.quit()

if __name__ == "__main__":
    main()
