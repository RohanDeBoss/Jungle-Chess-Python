import tkinter as tk
from tkinter import ttk
import time
import hashlib

# -----------------------------
# Global Constants
# -----------------------------
ROWS, COLS = 8, 8
SQUARE_SIZE = 65  # pixels
BOARD_COLOR_1 = "#D2B48C"
BOARD_COLOR_2 = "#8B5A2B"
HIGHLIGHT_COLOR = "#ADD8E6"

# -----------------------------
# Piece Base Class and Subclasses
# -----------------------------
class Piece:
    def __init__(self, color):
        self.color = color
        self.has_moved = False

    def clone(self):
        new_piece = self.__class__(self.color)
        new_piece.has_moved = self.has_moved
        return new_piece

    def save_state(self):
        return {'has_moved': self.has_moved}

    def restore_state(self, state):
        self.has_moved = state['has_moved']

    def symbol(self):
        return "?"

    def get_valid_moves(self, board, pos):
        return []

    def move(self, board, start, end):
        if board[end[0]][end[1]] is not None and board[end[0]][end[1]].color != self.color:
            board[end[0]][end[1]] = None
        board[end[0]][end[1]] = self
        board[start[0]][start[1]] = None
        self.has_moved = True
        return board


# ---- King ----
class King(Piece):
    def symbol(self):
        return "♔" if self.color == "white" else "♚"

    def get_valid_moves(self, board, pos):
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for d in directions:
            for step in [1, 2]:
                new_r = pos[0] + d[0] * step
                new_c = pos[1] + d[1] * step
                if 0 <= new_r < ROWS and 0 <= new_c < COLS:
                    if step == 2:
                        inter_r = pos[0] + d[0]
                        inter_c = pos[1] + d[1]
                        if board[inter_r][inter_c] is not None:
                            break
                    if board[new_r][new_c] is None or board[new_r][new_c].color != self.color:
                        moves.append((new_r, new_c))
                    if board[new_r][new_c] is not None:
                        break
        return moves

# ---- Queen ----
class Queen(Piece):
    def symbol(self):
        return "♕" if self.color == "white" else "♛"

    def get_valid_moves(self, board, pos):
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for d in directions:
            r, c = pos
            while True:
                r += d[0]
                c += d[1]
                if 0 <= r < ROWS and 0 <= c < COLS:
                    if board[r][c] is None:
                        moves.append((r, c))
                    else:
                        if board[r][c].color != self.color:
                            moves.append((r, c))
                        break
                else:
                    break
        return moves

    def move(self, board, start, end):
        if board[end[0]][end[1]] is not None and board[end[0]][end[1]].color != self.color:
            board[end[0]][end[1]] = None
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    r = end[0] + dr
                    c = end[1] + dc
                    if 0 <= r < ROWS and 0 <= c < COLS:
                        if board[r][c] is not None and board[r][c].color != self.color:
                            board[r][c] = None
            board[start[0]][start[1]] = None
        else:
            board = super().move(board, start, end)
        self.has_moved = True
        return board

# ---- Rook ----
class Rook(Piece):
    def symbol(self):
        return "♖" if self.color == "white" else "♜"

    def get_valid_moves(self, board, pos):
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for d in directions:
            enemy_encountered = False
            r, c = pos
            while True:
                r += d[0]
                c += d[1]
                if 0 <= r < ROWS and 0 <= c < COLS:
                    if not enemy_encountered:
                        if board[r][c] is None:
                            moves.append((r, c))
                        elif board[r][c].color != self.color:
                            moves.append((r, c))
                            enemy_encountered = True
                        else:
                            break
                    else:
                        if board[r][c] is None or board[r][c].color != self.color:
                            moves.append((r, c))
                        else:
                            break
                else:
                    break
        return moves

    def move(self, board, start, end):
        if start[0] == end[0]:
            d = (0, 1) if end[1] > start[1] else (0, -1)
        elif start[1] == end[1]:
            d = (1, 0) if end[0] > start[0] else (-1, 0)
        else:
            return board
        r, c = start
        path_positions = []
        while True:
            r += d[0]
            c += d[1]
            path_positions.append((r, c))
            if (r, c) == end:
                break
        enemy_encountered = any(
            board[r][c] is not None and board[r][c].color != self.color
            for (r, c) in path_positions
        )
        if enemy_encountered:
            for (r, c) in path_positions:
                if board[r][c] is not None and board[r][c].color != self.color:
                    board[r][c] = None
        else:
            if board[end[0]][end[1]] is not None and board[end[0]][end[1]].color != self.color:
                board[end[0]][end[1]] = None
        board[end[0]][end[1]] = self
        board[start[0]][start[1]] = None
        self.has_moved = True
        return board

def get_zigzag_moves(board, pos, color):
    """Calculate zigzag moves for the Bishop, alternating between two diagonal directions each step."""
    moves = set()
    r, c = pos
    direction_pairs = [
        # Up direction pairs
        ((-1, 1), (-1, -1)),
        ((-1, -1), (-1, 1)),
        # Down direction pairs
        ((1, 1), (1, -1)),
        ((1, -1), (1, 1)),
        # Right direction pairs
        ((-1, 1), (1, 1)),
        ((1, 1), (-1, 1)),
        # Left direction pairs
        ((-1, -1), (1, -1)),
        ((1, -1), (-1, -1)),
    ]
    for d1, d2 in direction_pairs:
        current_r, current_c = r, c
        current_dir = d1
        while True:
            new_r = current_r + current_dir[0]
            new_c = current_c + current_dir[1]
            # Check if new position is within the board boundaries
            if not (0 <= new_r < ROWS and 0 <= new_c < COLS):
                break
            # Check for pieces at the new position
            piece = board[new_r][new_c]
            if piece is not None:
                if piece.color != color:
                    moves.add((new_r, new_c))
                break  # Blocked by any piece
            else:
                moves.add((new_r, new_c))
            # Move to the new position and toggle direction for the next step
            current_r, current_c = new_r, new_c
            current_dir = d2 if current_dir == d1 else d1
    return list(moves)

def get_diagonal_moves(board, pos, color):
    """Calculate regular diagonal moves for the Bishop."""
    moves = []
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for d in directions:
        r, c = pos
        while True:
            r += d[0]
            c += d[1]
            # Check if new position is within the board boundaries
            if not (0 <= r < ROWS and 0 <= c < COLS):
                break
            # Check for pieces at the new position
            piece = board[r][c]
            if piece is not None:
                if piece.color != color:
                    moves.append((r, c))
                break  # Blocked by any piece
            else:
                moves.append((r, c))
    return moves

class Bishop(Piece):
    def symbol(self):
        return "♗" if self.color == "white" else "♝"

    def get_valid_moves(self, board, pos):
        # Combine zigzag and regular diagonal moves
        zigzag_moves = get_zigzag_moves(board, pos, self.color)
        diagonal_moves = get_diagonal_moves(board, pos, self.color)
        return list(set(zigzag_moves + diagonal_moves))  # Remove duplicates

    def move(self, board, start, end):
        if board[end[0]][end[1]] is not None and board[end[0]][end[1]].color != self.color:
            board[end[0]][end[1]] = None
        board[end[0]][end[1]] = self
        board[start[0]][start[1]] = None
        self.has_moved = True
        return board

class Knight(Piece):
    def symbol(self):
        return "♘" if self.color == "white" else "♞"

    def get_valid_moves(self, board, pos):
        moves = []
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                        (1, 2), (1, -2), (-1, 2), (-1, -2)]
        for d in knight_moves:
            r = pos[0] + d[0]
            c = pos[1] + d[1]
            if 0 <= r < ROWS and 0 <= c < COLS:
                # Check if the destination square is empty or contains an enemy piece
                if board[r][c] is None or board[r][c].color != self.color:
                    moves.append((r, c))
        return moves

    def move(self, board, start, end):
        # First, move the knight.
        board[end[0]][end[1]] = self
        board[start[0]][start[1]] = None

        # Then, from its new location, evaporate enemy pieces in all squares reachable via knight moves.
        self.evaporate(board, end)
        self.has_moved = True
        return board

    def evaporate(self, board, pos):
        """Evaporate all enemy pieces in the Knight's radius."""
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                        (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        # Keep track of any enemy knights we find
        enemy_knights = []
        
        # First, evaporate all pieces and note any enemy knights
        for d in knight_moves:
            r = pos[0] + d[0]
            c = pos[1] + d[1]
            if 0 <= r < ROWS and 0 <= c < COLS:
                piece_here = board[r][c]
                if piece_here is not None and piece_here.color != self.color:
                    if isinstance(piece_here, Knight):
                        enemy_knights.append((r, c))
                    board[r][c] = None
        
        # If we found any enemy knights, evaporate ourselves too
        if enemy_knights:
            board[pos[0]][pos[1]] = None
            
class Pawn(Piece):
    def symbol(self):
        return "♙" if self.color == "white" else "♟"

    def get_valid_moves(self, board, pos):
        moves = []
        direction = -1 if self.color == "white" else 1
        
        # Forward moves and captures (including two squares on first move)
        for steps in [1, 2]:
            # Only allow 2 steps if it's the first move
            if steps == 2 and self.has_moved:
                continue
                
            new_r = pos[0] + (direction * steps)
            new_c = pos[1]
            
            if 0 <= new_r < ROWS:
                # Allow either empty square or enemy piece
                if board[new_r][new_c] is None or (board[new_r][new_c] is not None and board[new_r][new_c].color != self.color):
                    moves.append((new_r, new_c))
                # Stop checking further steps if we hit any piece
                if board[new_r][new_c] is not None:
                    break
        
        # Sideways captures at current rank
        for dc in [-1, 1]:
            new_c = pos[1] + dc
            if 0 <= new_c < COLS:
                if board[pos[0]][new_c] is not None and board[pos[0]][new_c].color != self.color:
                    moves.append((pos[0], new_c))
        
        return moves

    def move(self, board, start, end):
        if board[end[0]][end[1]] is not None and board[end[0]][end[1]].color != self.color:
            board[end[0]][end[1]] = None
        board[end[0]][end[1]] = self
        board[start[0]][start[1]] = None
        self.has_moved = True
        return board

# -----------------------------
# Custom Board Copy Function
# -----------------------------
def copy_board(board):
    return [[piece.clone() if piece is not None else None for piece in row] for row in board]


# -----------------------------
# Board Setup and Game Over Check
# -----------------------------
def create_initial_board():
    board = [[None for _ in range(COLS)] for _ in range(ROWS)]
    board[0][0] = Rook("black")
    board[0][1] = Knight("black")
    board[0][2] = Bishop("black")
    board[0][3] = Queen("black")
    board[0][4] = King("black")
    board[0][5] = Bishop("black")
    board[0][6] = Knight("black")
    board[0][7] = Rook("black")
    for i in range(COLS):
        board[1][i] = Pawn("black")
    board[7][0] = Rook("white")
    board[7][1] = Knight("white")
    board[7][2] = Bishop("white")
    board[7][3] = Queen("white")
    board[7][4] = King("white")
    board[7][5] = Bishop("white")
    board[7][6] = Knight("white")
    board[7][7] = Rook("white")
    for i in range(COLS):
        board[6][i] = Pawn("white")
    return board

def check_game_over(board):
    white_king_found = False
    black_king_found = False
    for row in board:
        for piece in row:
            if piece is not None and isinstance(piece, King):
                if piece.color == "white":
                    white_king_found = True
                elif piece.color == "black":
                    black_king_found = True
    if not white_king_found:
        return "black"
    if not black_king_found:
        return "white"
    return None

def check_evaporation(board):
    """Check for evaporation after every move."""
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if isinstance(piece, Knight):
                piece.evaporate(board, (r, c))

    
def manhattan_distance(pos1, pos2):
    """Calculate the Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class ChessBot:
    search_depth = 3

    def __init__(self, board, color, app):
        self.board = board
        self.color = color
        self.tt = {}
        self.nodes_searched = 0
        self.app = app  # Store the EnhancedChessApp instance

    def evaluate_board(self, board, depth):
        piece_values = {
            Pawn: 100,
            Knight: 700,
            Bishop: 600,
            Rook: 500,
            Queen: 900,
            King: 100000
        }
        
        score = 0
        our_king_pos = None
        enemy_king_pos = None
        
        # First pass: find kings and calculate material score
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if piece is None:
                    continue
                    
                # Track king positions
                if isinstance(piece, King):
                    if piece.color == self.color:
                        our_king_pos = (r, c)
                    else:
                        enemy_king_pos = (r, c)
                    continue
                
                # Calculate piece value with bonuses
                value = piece_values[type(piece)]
                if isinstance(piece, Knight):
                    value += len(piece.get_valid_moves(board, (r, c))) * 10
                elif isinstance(piece, Queen):
                    atomic_threats = sum(1 for move in piece.get_valid_moves(board, (r, c))
                                    if any(isinstance(board[adj_r][adj_c], King)
                                        for adj_r, adj_c in self.get_adjacent_squares(move)))
                    value += atomic_threats * 20
                
                score += value if piece.color == self.color else -value

        # Check win/loss conditions
        if enemy_king_pos is None:
            return float('inf')
        if our_king_pos is None:
            return float('-inf')

        # Evaluate threats to our king
        threat_score = 0
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if piece and piece.color != self.color:
                    for move in piece.get_valid_moves(board, (r, c)):
                        if manhattan_distance(move, our_king_pos) <= 1:
                            threat_score += 200

        return int(score - threat_score)

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        self.nodes_searched += 1
        
        board_key = self.board_hash(board)
        if board_key in self.tt and self.tt[board_key][0] >= depth:
            return self.tt[board_key][1]

        if depth == 0:
            return self.evaluate_board(board, depth)

        moves = self.get_all_moves(board, 
                                 self.color if maximizing_player 
                                 else ("black" if self.color == "white" else "white"))
        if not moves:
            return self.evaluate_board(board, depth)

        best_move = None
        if maximizing_player:
            value = float('-inf')
            for move in moves:
                start, end = move
                new_board = self.simulate_move(board, start, end)
                eval_value = self.minimax(new_board, depth - 1, False, alpha, beta)
                if eval_value > value:
                    value = eval_value
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
        else:
            value = float('inf')
            for move in moves:
                start, end = move
                new_board = self.simulate_move(board, start, end)
                eval_value = self.minimax(new_board, depth - 1, True, alpha, beta)
                if eval_value < value:
                    value = eval_value
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break

        self.tt[board_key] = (depth, value, best_move)
        return value

    def make_move(self):
        best_move = None
        best_value = float('-inf')
        total_start = time.time()
        
        for current_depth in range(1, self.search_depth + 1):
            self.nodes_searched = 0
            iteration_start = time.time()
            
            moves = self.get_all_moves(self.board, self.color)
            if not moves:
                return False
                
            current_best_move = None
            current_best_value = float('-inf')
            
            for move in moves:
                start, end = move
                new_board = self.simulate_move(self.board, start, end)
                value = self.minimax(new_board, current_depth - 1, False, float('-inf'), float('inf'))
                if value > current_best_value:
                    current_best_value = value
                    current_best_move = move
                    
            iteration_time = time.time() - iteration_start
            # Flip the evaluation value if the bot is black to report from white's perspective
            reported_value = -current_best_value if self.color == 'black' else current_best_value
            print(f"Depth {current_depth}: {iteration_time:.3f}s, "
                f"nodes: {self.nodes_searched}, value: {reported_value}")
            
            # Update the evaluation bar in the UI
            self.app.master.after(0, lambda: self.app.draw_eval_bar(reported_value))
                
            best_move = current_best_move
            best_value = current_best_value
            
        print(f"Total time: {(time.time() - total_start):.3f}s")
        
        if best_move is not None:
            start, end = best_move
            moving_piece = self.board[start[0]][start[1]]
            self.board = moving_piece.move(self.board, start, end)
            check_evaporation(self.board)
            return True
        return False

    # Helper methods
    def board_hash(self, board):
        """Generate a unique hash for the board state."""
        board_str = ''.join(
            piece.symbol() if piece is not None else '.'
            for row in board for piece in row
        )
        return hashlib.md5(board_str.encode()).hexdigest()

    def get_adjacent_squares(self, pos):
        """Get all adjacent squares for a given position."""
        r, c = pos
        adjacent_squares = [
            (r-1, c-1), (r-1, c), (r-1, c+1),
            (r, c-1),           (r, c+1),
            (r+1, c-1), (r+1, c), (r+1, c+1)
        ]
        return [(adj_r, adj_c) for adj_r, adj_c in adjacent_squares 
                if 0 <= adj_r < ROWS and 0 <= adj_c < COLS]

    def simulate_move(self, board, start, end):
        """Simulate a move on the board and return the new board state."""
        new_board = copy_board(board)
        piece = new_board[start[0]][start[1]]
        new_board = piece.move(new_board, start, end)
        check_evaporation(new_board)  # Apply evaporation effect
        return new_board

    def is_in_check(self, board, color):
        """Check if the given color is in check."""
        king_pos = None
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if piece is not None and isinstance(piece, King) and piece.color == color:
                    king_pos = (r, c)
                    break
            if king_pos is not None:
                break

        if king_pos is None:
            return False

        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if piece is not None and piece.color != color:
                    if king_pos in piece.get_valid_moves(board, (r, c)):
                        return True
        return False

    def get_all_moves(self, board, color):
        """Get all legal moves for the given color."""
        moves = []
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if piece is not None and piece.color == color:
                    for move in piece.get_valid_moves(board, (r, c)):
                        test_board = self.simulate_move(board, (r, c), move)
                        check_evaporation(test_board)  # Apply evaporation effect
                        if not self.is_in_check(test_board, color):
                            moves.append(((r, c), move))
        
        # Reorder moves using the transposition table
        board_key = self.board_hash(board)
        if board_key in self.tt:
            tt_best_move = self.tt[board_key][2]
            if tt_best_move in moves:
                moves.remove(tt_best_move)
                moves.insert(0, tt_best_move)
        return moves

class EnhancedChessApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Enhanced Chess")
        
        # Get colors from setup_styles
        self.COLORS = self.setup_styles()
        self.master.configure(bg=self.COLORS['bg_dark'])

        # Window setup: Fill the screen (windowed fullscreen)
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.master.geometry(f"{screen_width}x{screen_height}+0+0")
        self.master.state('zoomed')  # Start maximized/fullscreen
        self.fullscreen = True  # Flag to track fullscreen state
        
        # Bind to configuration changes to detect when fullscreen is exited via window buttons
        self.master.bind("<Configure>", self.on_configure)

        # Main frame with padding
        self.main_frame = ttk.Frame(master, style='Left.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Left panel with increased width
        self.left_panel = ttk.Frame(self.main_frame, width=250, style='Left.TFrame')
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=15)
        self.left_panel.pack_propagate(False)

        # Add the evaluation bar to the top of the left panel
        self.eval_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        self.eval_frame.pack(side=tk.TOP, fill=tk.Y, padx=15, pady=15)

        # Add other elements (title, game mode, controls, etc.) below the evaluation bar
        ttk.Label(self.left_panel, text="CHESS", 
          style='Header.TLabel',
          font=('Helvetica', 24, 'bold')).pack(pady=(0, 20))
        # Game mode frame with subtle border
        self.game_mode_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        self.game_mode_frame.pack(fill=tk.X, pady=(0, 20))

        # Game mode selection
        self.game_mode = tk.StringVar(value="bot")
        ttk.Label(self.game_mode_frame, text="GAME MODE", 
                  style='Header.TLabel').pack(anchor=tk.W)
        ttk.Radiobutton(self.game_mode_frame, text="Human vs Bot", 
                        variable=self.game_mode,
                        value="bot", 
                        command=self.reset_game, 
                        style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(5, 3))
        ttk.Radiobutton(self.game_mode_frame, text="Human vs Human", 
                        variable=self.game_mode,
                        value="human", 
                        command=self.reset_game, 
                        style='Custom.TRadiobutton').pack(anchor=tk.W)

        # Controls frame
        self.controls_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        self.controls_frame.pack(fill=tk.X, pady=10)

        # Buttons with increased padding
        ttk.Button(self.controls_frame, text="NEW GAME", 
                   command=self.reset_game,
                   style='Control.TButton').pack(fill=tk.X, pady=5)
        # Replace FULLSCREEN button with BOT SETTINGS button
        ttk.Button(self.controls_frame, text="BOT SETTINGS", 
                   command=self.open_settings,
                   style='Control.TButton').pack(fill=tk.X, pady=5)
        ttk.Button(self.controls_frame, text="QUIT", 
                   command=self.master.quit,
                   style='Control.TButton').pack(fill=tk.X, pady=5)

        # Enhanced turn indicator
        self.turn_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        self.turn_frame.pack(fill=tk.X, pady=(20, 0))
        self.turn_label = ttk.Label(self.turn_frame, text="WHITE'S TURN", 
                                    style='Status.TLabel')
        self.turn_label.pack(fill=tk.X)

        # Right panel with improved canvas positioning
        self.right_panel = ttk.Frame(self.main_frame, style='Right.TFrame')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Canvas container with shadow effect
        self.canvas_container = ttk.Frame(self.right_panel, style='Canvas.TFrame')
        self.canvas_container.pack(expand=True)
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)

        # Canvas frame with border
        self.canvas_frame = ttk.Frame(self.canvas_container, style='Canvas.TFrame')
        self.canvas_frame.grid(row=0, column=0)

        # Enhanced chess board canvas
        self.canvas = tk.Canvas(self.canvas_frame,
                                width=COLS * SQUARE_SIZE,
                                height=ROWS * SQUARE_SIZE,
                                bg=self.COLORS['bg_light'],
                                highlightthickness=2,
                                highlightbackground=self.COLORS['accent'])
        self.canvas.pack()

        # Move eval_frame to the left panel
        self.eval_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        self.eval_frame.pack(side=tk.TOP, fill=tk.Y, padx=15, pady=15)

        # Now sized for a horizontal bar
        self.eval_bar_canvas = tk.Canvas(self.eval_frame, 
            width=300,
            height=30,
            bg=self.COLORS['bg_light'],
            highlightthickness=0)
        # If you want it truly at the "bottom left," you might do side=tk.BOTTOM here:
        self.eval_bar_canvas.pack(side=tk.BOTTOM, pady=(10, 5))


        # Label to display the evaluation score (optional, can be removed)
        self.eval_score_label = ttk.Label(self.eval_frame, 
                                        text="",  # Empty by default
                                        style='Status.TLabel')
        self.eval_score_label.pack()

        # Initialize the evaluation bar
        self.draw_eval_bar(0)  # Start with a neutral evaluation
        # Add a toggle for the evaluation bar in the bot settings
        self.eval_bar_visible = True

        # Rest of initialization
        self.board = create_initial_board()
        self.turn = "white"
        self.selected = None
        self.valid_moves = []
        self.game_over = False
        self.dragging = False
        self.drag_piece = None
        self.drag_start = None
        self.bot = ChessBot(self.board, "black", self)  # Pass self to ChessBot

        # Bind events
        self.canvas.bind("<Button-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)

        # Initial draw
        self.draw_board()
        
    def on_configure(self, event):
        # Handle window configuration changes (e.g., resizing)
        pass

    def open_settings(self):
        # Create a settings window (Toplevel)
        settings_win = tk.Toplevel(self.master)
        settings_win.title("Bot Settings")
        
        # Center the settings window (for example, 300x150)
        win_width, win_height = 300, 150
        screen_width = settings_win.winfo_screenwidth()
        screen_height = settings_win.winfo_screenheight()
        x = (screen_width - win_width) // 2
        y = (screen_height - win_height) // 2
        settings_win.geometry(f"{win_width}x{win_height}+{x}+{y}")
        
        # Label and Spinbox for bot search depth
        ttk.Label(settings_win, text="Bot Search Depth:",
                  font=('Helvetica', 12)).pack(pady=(20, 5))
        depth_var = tk.IntVar(value=ChessBot.search_depth)
        spin = ttk.Spinbox(settings_win, from_=1, to=5, textvariable=depth_var, width=5)
        spin.pack(pady=(0, 20))

        # Checkbox to toggle evaluation bar visibility
        eval_bar_var = tk.BooleanVar(value=self.eval_bar_visible)
        ttk.Checkbutton(settings_win, text="Show Evaluation Bar", 
                        variable=eval_bar_var).pack(pady=(0, 10))
        
        def apply_settings():
            # Update both the class default and the current bot instance
            new_depth = depth_var.get()
            ChessBot.search_depth = new_depth
            if hasattr(self, 'bot'):
                self.bot.search_depth = new_depth
            
            # Toggle evaluation bar visibility
            self.eval_bar_visible = eval_bar_var.get()
            if self.eval_bar_visible:
                self.eval_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=15, pady=15)
            else:
                self.eval_frame.pack_forget()
            
            settings_win.destroy()
        
        ttk.Button(settings_win, text="Apply", command=apply_settings).pack()

    def draw_eval_bar(self, eval_score):
        """
        Draw a horizontal evaluation bar along the bottom-left.
        """
        # Convert centipawn eval to pawn units.
        eval_score /= 100.0

        self.eval_bar_canvas.delete("all")

        bar_width = 235   # match red buttons
        bar_height = 30
        max_eval = 5.0
        neutral_zone = 0.2

        normalized_score = max(min(eval_score / max_eval, 1.0), -1.0)

        # Draw left-to-right gradient from black to white.
        for x in range(bar_width):
            ratio = x / float(bar_width)
            r = int(255 * ratio)
            g = int(255 * ratio)
            b = int(255 * ratio)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.eval_bar_canvas.create_line(x, 0, x, bar_height, fill=color)

        # Compute marker position based on eval.
        marker_x = int((normalized_score + 1) / 2 * bar_width)
        accent_color = self.COLORS.get('accent', '#e94560')
        marker_width = 2
        self.eval_bar_canvas.create_rectangle(
            marker_x - marker_width, 0,
            marker_x + marker_width, bar_height,
            fill=accent_color, outline=""
        )

        # Shift the middle marker slightly further left.
        middle_offset = 0  # increased from 3 to 5 pixels
        mid_x = (bar_width // 2) - middle_offset
        self.eval_bar_canvas.create_line(mid_x, 0, mid_x, bar_height, fill="#666666", width=1)

        # Update evaluation score label.
        if abs(eval_score) < neutral_zone:
            self.eval_score_label.config(text="Even", font=("Helvetica", 10))
        else:
            display_score = abs(eval_score)
            if eval_score > 0:
                self.eval_score_label.config(text=f"+{display_score:.2f}", font=("Helvetica", 10))
            else:
                self.eval_score_label.config(text=f"-{display_score:.2f}", font=("Helvetica", 10))

        self.master.update_idletasks()


    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Define color scheme
        COLORS = {
            'bg_dark': '#1a1a2e',  # Darker background
            'bg_medium': '#16213e',  # Medium background
            'bg_light': '#0f3460',   # Lighter background
            'accent': '#e94560',     # Accent color
            'text_light': '#ffffff',
            'text_dark': '#a2a2a2'
        }

        # Configure frame styles
        style.configure('Left.TFrame', background=COLORS['bg_dark'])
        style.configure('Right.TFrame', background=COLORS['bg_medium'])
        style.configure('Canvas.TFrame', background=COLORS['bg_medium'])

        # Enhanced header label style
        style.configure('Header.TLabel',
                        background=COLORS['bg_dark'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 14, 'bold'),
                        padding=(0, 10))

        # Modern status label style
        style.configure('Status.TLabel',
                        background=COLORS['bg_light'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 16, 'bold'),
                        padding=(15, 10),
                        relief='flat',
                        borderwidth=0)

        # Sleek button style
        style.configure('Control.TButton',
                        background=COLORS['accent'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 11, 'bold'),
                        padding=(15, 12),
                        borderwidth=0,
                        relief='flat')
        
        style.map('Control.TButton',
                  background=[('active', COLORS['accent']),
                              ('pressed', '#d13550')],
                  relief=[('pressed', 'flat'),
                          ('!pressed', 'flat')])

        # Modern radio button style
        style.configure('Custom.TRadiobutton',
                        background=COLORS['bg_dark'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 11),
                        padding=(5, 8))
        
        style.map('Custom.TRadiobutton',
                  background=[('active', COLORS['bg_dark'])],
                  foreground=[('active', COLORS['accent'])])
                 
        return COLORS

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.master.attributes('-fullscreen', self.fullscreen)
        if not self.fullscreen:
            self.master.geometry("800x600")

    def draw_board(self):
        self.canvas.delete("all")
    
        # Draw board squares
        for r in range(ROWS):
            for c in range(COLS):
                x1 = c * SQUARE_SIZE
                y1 = r * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE
                color = BOARD_COLOR_1 if (r + c) % 2 == 0 else BOARD_COLOR_2
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
            
                # Highlight valid moves
                if (r, c) in self.valid_moves:
                    self.canvas.create_oval(x1 + 19, y1 + 19, x2 - 19, y2 - 19,
                            fill="#1E90FF", outline="#1E90FF", width=3)
    
        # Draw pieces
        for r in range(ROWS):
            for c in range(COLS):
                self.draw_piece(r, c)
            
        # Draw dragged piece
        if self.dragging and self.drag_piece:
            x, y = self.drag_piece
            piece = self.board[self.drag_start[0]][self.drag_start[1]]
            self.canvas.create_text(x, y, text=piece.symbol(), 
                              font=("Arial", 36), tags="drag")

    def draw_piece(self, r, c):
        piece = self.board[r][c]
        if piece is not None and (r, c) != self.drag_start:
            x = c * SQUARE_SIZE + SQUARE_SIZE // 2
            y = r * SQUARE_SIZE + SQUARE_SIZE // 2
            symbol = piece.symbol()
    
            if piece.color == "white":
                # Add a shadow for white pieces
                shadow_offset = 2  # Adjust the shadow offset as needed
                shadow_color = "#444444"  # Dark gray for shadow
                self.canvas.create_text(x + shadow_offset, y + shadow_offset, text=symbol,
                                    font=("Arial", 39),
                                    fill=shadow_color,  # Shadow color
                                    tags="piece")
                self.canvas.create_text(x, y, text=symbol,
                        font=("Arial Unicode MS", 39),
                        fill="white" if piece.color == "white" else "black",
                        tags="piece")
            else:
                # Black pieces remain as before
                self.canvas.create_text(x, y, text=symbol,
                                    font=("Arial", 39),
                                    fill="black",
                                    tags="piece")

    def on_drag_start(self, event):
        if self.game_over:
            return
            
        col = event.x // SQUARE_SIZE
        row = event.y // SQUARE_SIZE
        piece = self.board[row][col]
        
        if piece is not None and piece.color == self.turn:
            self.dragging = True
            self.drag_start = (row, col)
            self.drag_piece = (event.x, event.y)
            self.valid_moves = piece.get_valid_moves(self.board, (row, col))
            self.draw_board()

    def on_drag_motion(self, event):
        if self.dragging:
            self.drag_piece = (event.x, event.y)
            self.draw_board()

    def on_drag_end(self, event):
        if not self.dragging:
            return
        
        col = event.x // SQUARE_SIZE
        row = event.y // SQUARE_SIZE
        end_pos = (row, col)
    
        if end_pos in self.valid_moves:
            moving_piece = self.board[self.drag_start[0]][self.drag_start[1]]
            self.board = moving_piece.move(self.board, self.drag_start, end_pos)
            check_evaporation(self.board)
        
            winner = check_game_over(self.board)
            if winner is not None:
                self.game_over = True
                # Update the turn_label to show the winner
                self.turn_label.config(text=f"{winner.capitalize()} wins!")
            else:
                self.turn = "black" if self.turn == "white" else "white"
                self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")
            
                # If playing against bot and it's bot's turn
                if self.game_mode.get() == "bot" and self.turn == "black":
                    self.master.after(500, self.make_bot_move)

        self.dragging = False
        self.drag_piece = None
        self.drag_start = None
        self.valid_moves = []
        self.draw_board()

    def make_bot_move(self):
        if self.game_over:
            return
    
        if self.bot.make_move():
            winner = check_game_over(self.board)
            if winner is not None:
                self.game_over = True
                self.turn_label.config(text=f"{winner.capitalize()} wins!")
            else:
                self.turn = "white"
                self.turn_label.config(text="Turn: White")
    
            self.draw_board()
        else:
            # If no legal moves are available, it's checkmate
            self.game_over = True
            self.turn_label.config(text="White wins!")

    def reset_game(self):
        self.board = create_initial_board()
        self.turn = "white"
        self.selected = None
        self.valid_moves = []
        self.game_over = False
        self.dragging = False
        self.drag_piece = None
        self.drag_start = None
        self.bot = ChessBot(self.board, "black", self)  # Pass self to ChessBot
        self.turn_label.config(text="Turn: White")
        self.draw_board()

def main():
    root = tk.Tk()
    app = EnhancedChessApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()