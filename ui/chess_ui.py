import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import chess
import chess.svg
import io
import cairosvg

class ChessApp(tk.Tk):
    def __init__(self):
        """
        Initializes UI in tkinter with given title, size
        """
        super().__init__()

        self.title("Chess Arm UI")
        self.geometry("800x400")

        self.board = chess.Board()
        self.create_widgets()

    def create_widgets(self):
        """
        Creates the widgets in the UI's frame
        """
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Area for chessboard visualization
        self.chessboard_frame = ttk.Frame(self.main_frame, padding=(0, 0, 20, 0))
        self.chessboard_frame.grid(row=0, column=0, sticky=tk.NSEW)

        self.update_board()

        # Button container
        self.buttons_frame = ttk.Frame(self.main_frame)
        self.buttons_frame.grid(row=0, column=1, sticky=tk.NSEW)

        # Button (previous move)
        self.prev_move_button = ttk.Button(self.buttons_frame, text="Previous Move", command=self.previous_move)
        self.prev_move_button.pack(fill=tk.X, pady=(50, 10))

        # Button (next move)
        self.next_move_button = ttk.Button(self.buttons_frame, text="Next Move", command=self.next_move)
        self.next_move_button.pack(fill=tk.X)

    def update_board(self):
        # does some fancy chess svg stuff to display self.board
        svg_data = chess.svg.board(self.board)
        png_data = io.BytesIO()
        cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), write_to=png_data)
        img_data = Image.open(png_data)
        img_data = img_data.resize((400, 400), Image.ANTIALIAS)
        img_data = ImageTk.PhotoImage(img_data)

        # gets rid of existing board
        if hasattr(self, 'chessboard_label'):
            self.chessboard_label.destroy()

        # puts picture on UI
        self.chessboard_label = ttk.Label(self.chessboard_frame, image=img_data)
        self.chessboard_label.image = img_data
        self.chessboard_label.pack()

    def next_move(self):
        # gets first possible move, makes it
        if self.board.legal_moves:
            move = self.board.legal_moves.__iter__().__next__()
            self.board.push(move)
            self.update_board()

    def previous_move(self):
        # undoes last move
        if len(self.board.move_stack) > 0:
            self.board.pop()
            self.update_board()

    def close_app(self):
        self.destroy()


if __name__ == "__main__":
    app = ChessApp()
    app.mainloop()
