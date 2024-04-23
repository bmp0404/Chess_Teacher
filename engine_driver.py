from ui.minimal_ui import ChessUI
import ai.ai as ai
from Game import ChessGame
import chess
import chess.pgn
from concurrent.futures import ProcessPoolExecutor
from Agents import *
import chess.svg
import threading
import queue
import time
import argparse
import Teach

USE_UI = True
LOG = False

from concurrent.futures import ProcessPoolExecutor
import time

def run_single_game(white_agent, black_agent):
    game = ChessGame(white_agent=white_agent, black_agent=black_agent)
    board_state = game.get_board()

    w_agent_times = []
    b_agent_times = []

    while not board_state.is_game_over():
        # print("\n",board_state)
        player = game.get_next_player()
        start_time = time.perf_counter()
        next_move = player.get_move(board_state)
        time_elapsed = time.perf_counter() - start_time
        # if USE_UI:
            # ui.update_board(board_state)

        if player == white_agent:
            w_agent_times.append(time_elapsed)
        else:
            b_agent_times.append(time_elapsed)
        
        board_state.push(next_move)

    winner = None
    if board_state.result() == "1-0":
        winner = (1, 0, 0)
    elif board_state.result() == "0-1":
        winner = (0, 1, 0)
    else:
        winner = (0, 0, 1)

    print("finished game!", board_state.result())
    return winner, w_agent_times, b_agent_times


def stockfish_ladder(ui, agent):
    sf_agent = StockfishAgent()

    print("stockfish as white", agent.__class__.__name__, "as black")
    for i in range(1, 21):
        sf_agent.set_skill(i)
        result = run_single_game(ui, agent, sf_agent)
        result = ("w wins as " + agent.__class__.__name__) if result[0][0] == 1 else "b wins as sf" if result[0][1] == 1 else "draw"
        print("level", i, "outcome:", result)
        if result == "b wins as sf":
            break


def test_games(white_agent = RandomAgent(), black_agent = RandomAgent(), game_count = 1):
    w_total_times = []
    b_total_times = []

    with ProcessPoolExecutor() as executor:
        future_games = [executor.submit(run_single_game, white_agent, black_agent) for _ in range(game_count)]
        results = [future.result() for future in future_games]

    # results = []
    # for _ in range(game_count):
    #     winner, w_times, b_times = run_single_game(white_agent, black_agent)
    #     results.append((winner, w_times, b_times))
        # print("finished game!", winner)

    w_wins = sum(w for (w, _, _), _, _ in results)
    b_wins = sum(b for (_, b, _), _, _ in results)
    draws  = sum(d for (_, _, d), _, _ in results)

    for _, w_times, b_times in results:
        w_total_times.extend(w_times)
        b_total_times.extend(b_times)

    print(f"\n{white_agent.__class__.__name__} wins as WHITE: {w_wins} ({w_wins/game_count*100:.2f}%)")
    print(f"{black_agent.__class__.__name__} wins as BLACK: {b_wins} ({b_wins/game_count*100:.2f}%)")
    print(f"Draws: {draws} ({draws/game_count*100:.2f}%)")
    print(f"\n{white_agent.__class__.__name__} Average time per move: {sum(w_total_times)/len(w_total_times):.2f}s")
    print(f"{black_agent.__class__.__name__} Average time per move: {sum(b_total_times)/len(b_total_times):.2f}s")

def moves_to_pgn(moves, result, white = "opponet", black = "opponet"):
    game = chess.pgn.Game()
    game.headers["Result"] = result
    game.headers["White"] = white
    game.headers["Black"] = black
    node = game
    evaluation = []
    board = chess.Board()
    for move in moves:
        move_obj = chess.Move.from_uci(move)
        board.push(move_obj)
        node = node.add_variation(move_obj)

        # Get evaluation after the move
        result = ai.getPositionEval(board.fen())
        evaluation.append(result)

        node.comment = f"Evaluation: {result}"

    return game

def save_pgn(game, filename):
    with open(filename, "w") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)

from queue import Queue

def main(ui, white, black):
    game = ChessGame(white_agent=white, black_agent=black)
    board_state_queue = Queue()  # Queue to pass board states from game loop to UI update loop

    name = game.white.__class__.__name__ 
    side = 0
    
    if game.white.__class__.__name__ == "MouseAgent":
        side = 0
        name = input("What is your name: ")
        game.white.__class__.__name__ = name
    elif game.black.__class__.__name__ == "MouseAgent":
        side = 1
        name = input("What is your name: ")
        game.black.__class__.__name__ = name

    print(f"W: {game.white.__class__.__name__} B: {game.black.__class__.__name__}")
    board_state = game.get_board()
    movelist = []

    # Define a function to run the game loop
    def game_loop():
        while not board_state.is_game_over():
            player = game.get_next_player()
            next_move = player.get_move(board_state)
            print(next_move)
            movelist.append(next_move.uci())
            board_state.push(next_move)

            if LOG:
                game.print_state()

            # Put the current board state in the queue
            board_state_queue.put(board_state.copy())

    # Start the game loop in a separate thread
    game_thread = threading.Thread(target=game_loop)
    game_thread.start()

    if USE_UI:
        # Define a function to continuously update the UI
        def update_ui_loop():
            while True:
                # Get the latest board state from the queue
                board_state = board_state_queue.get()
                ui.update_board(board_state)
                if board_state.is_game_over():
                    break  # Exit the loop if the game is over

        # Start the UI update loop in a separate thread
        ui_thread = threading.Thread(target=update_ui_loop)
        ui_thread.start()

    # Wait for the game loop thread to finish
    game_thread.join()

    if USE_UI:
        ui_thread.join()

    print("\nresult:", board_state.result())
    pgn = moves_to_pgn(movelist, board_state.result(), game.white.__class__.__name__, game.black.__class__.__name__)
    save_pgn(pgn, "game.pgn")
    Teach.main(side)
    ui.destroy()





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='robot arm that plays chess')
    parser.add_argument('-w', '--white', type=str, default='RandomAgent', metavar="PlayerAgent",
                        help='specify who plays white (H , R_ARM , or H_ARM)')
    parser.add_argument('-b', '--black', type=str, default='RandomAgent', metavar="PlayerAgent",
                        help='specify who plays black (H , R_ARM, or H_ARM)')
    parser.add_argument('-tc', '--time-control', type=str, default='NONE', metavar="10/2",
                    help='time control for the game in the format <minutes>/<increment> (e.g. 10/2 for 10 minutes with a 2 second increment). If not specified, the game will have no time control.')
    parser.add_argument('--no-ui', dest='ui', action='store_false')
    parser.add_argument('--no-logs', dest='logs', action='store_false')
    parser.add_argument('--tests-per-side', type=int, default=0, metavar="N")
    
    parser.set_defaults(ui=True, logs=True)

    args = parser.parse_args()

    app = None if not args.ui else ChessUI()
    agents = {Agent.__name__: Agent for Agent in Agent.__subclasses__()}

    USE_UI = args.ui
    LOG = args.logs


    if args.white not in agents:
        raise ValueError('white must be in ' + ', '.join(agents.keys()))
    if args.black not in agents:
        raise ValueError('black must be in ' + ', '.join(agents.keys()))
    
    white = agents[args.white]() if args.white != 'MouseAgent' else agents[args.white](app)
    black = agents[args.black]() if args.black != 'MouseAgent' else agents[args.black](app)

    if args.tests_per_side > 0:
        print("\ntesting with", args.tests_per_side * 2, "games...")
        test_games(white, black, args.tests_per_side)
        print("\nswitching sides...\n")
        test_games(black, white, args.tests_per_side)
        exit()

    # stockfish_ladder(agent=white)
    # thread = threading.Thread(target=stockfish_ladder, args=(app, white,))
    # thread.start()
    thread = threading.Thread(target=main, args=(app, white, black,))
    thread.start()


    if USE_UI:
        print("starting UI thread...")
        app.mainloop()