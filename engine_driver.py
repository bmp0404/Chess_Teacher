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

def moves_to_pgn(moves):
    print(moves)
    game = chess.pgn.Game()
    node = game

    board = chess.Board()

    for move in moves:
        print(move)
        move = chess.Move.from_uci(move)
        board.push(move)
        node = node.add_variation(move)

    return game

def save_pgn(game, filename):
    with open(filename, "w") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)

def main(ui, white, black):
    game = ChessGame(white_agent=white, black_agent=black)
    print(f"W: {game.white.__class__.__name__} B: {game.black.__class__.__name__}")
    board_state = game.get_board()
    movelist = []
    while not board_state.is_game_over():
        player = game.get_next_player()
        next_move = player.get_move(board_state)
        print(next_move)
        movelist.append((next_move))
        board_state.push(next_move)

        if USE_UI:
            ui.update_board(board_state)
        if LOG:
            game.print_state()

    print("\nresult:",board_state.result())
    print(movelist)
    moves_to_pgn(movelist)

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