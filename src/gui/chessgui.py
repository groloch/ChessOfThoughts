import tkinter as tk
from tkinter import ttk
import chess
from PIL import Image, ImageTk
import os

class ChessGUI:
    def __init__(self, master, board):
        self.master = master
        self.master.title("Chess Board")
        self.master.resizable(False, False)
        
        self.board = board
        
        self.square_size = 64
        self.board_size = 8 * self.square_size
        
        self.light_color = "#F0D9B5"
        self.dark_color = "#B58863"
        
        self.canvas = tk.Canvas(
            master, 
            width=self.board_size, 
            height=self.board_size,
            highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=10)
        
        self.piece_images = {}
        self.load_piece_images()
        
        self.draw_board()
        self.draw_pieces()
        
        self.create_controls()
    
    def load_piece_images(self):
        """Load piece images from data directory"""
        piece_mapping = {
            'K': 'wk', 'Q': 'wq', 'R': 'wr', 'B': 'wb', 'N': 'wn', 'P': 'wp',
            'k': 'bk', 'q': 'bq', 'r': 'br', 'b': 'bb', 'n': 'bn', 'p': 'bp'
        }
        
        for piece_symbol, filename in piece_mapping.items():
            try:
                image_path = f"src/gui/data/{filename}.png"
                if os.path.exists(image_path):
                    # Load and resize image
                    img = Image.open(image_path)
                    img = img.resize((self.square_size, self.square_size), Image.Resampling.LANCZOS)
                    self.piece_images[piece_symbol] = ImageTk.PhotoImage(img)
                else:
                    print(f"Warning: Image not found: {image_path}")
            except Exception as e:
                print(f"Error loading {filename}.png: {e}")
    
    def draw_board(self):
        """Draw the chessboard squares"""
        self.canvas.delete("square")
        
        for rank in range(8):
            for file in range(8):
                x1 = file * self.square_size
                y1 = rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                # Determine square color
                if (rank + file) % 2 == 0:
                    color = self.light_color
                else:
                    color = self.dark_color
                
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, 
                    fill=color, 
                    outline=color,
                    tags="square"
                )
    
    def draw_pieces(self):
        """Draw pieces on the board"""
        self.canvas.delete("piece")
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                # Convert square to file/rank
                file = chess.square_file(square)
                rank = 7 - chess.square_rank(square)  # Flip rank for display
                
                # Calculate position
                x = file * self.square_size + self.square_size // 2
                y = rank * self.square_size + self.square_size // 2
                
                # Get piece symbol
                piece_symbol = piece.symbol()
                
                if piece_symbol in self.piece_images:
                    self.canvas.create_image(
                        x, y, 
                        image=self.piece_images[piece_symbol],
                        tags="piece"
                    )
                else:
                    # Fallback: draw text if image not available
                    self.canvas.create_text(
                        x, y,
                        text=piece_symbol,
                        font=("Arial", 24),
                        tags="piece"
                    )
    
    def create_controls(self):
        """Create control buttons"""
        control_frame = ttk.Frame(self.master)
        control_frame.pack(pady=5)
        
        ttk.Button(
            control_frame, 
            text="Reset Board", 
            command=self.reset_board
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame, 
            text="Refresh", 
            command=self.refresh_display
        ).pack(side=tk.LEFT, padx=5)
    
    def reset_board(self):
        """Reset board to starting position"""
        self.board.reset()
        self.refresh_display()
    
    def refresh_display(self):
        """Refresh the board display"""
        self.draw_board()
        self.draw_pieces()
    
    def set_position(self, fen):
        """Set board position from FEN string"""
        try:
            self.board.set_fen(fen)
            self.refresh_display()
        except ValueError as e:
            print(f"Invalid FEN: {e}")
    
    def get_board(self):
        """Get the chess board object"""
        return self.board
