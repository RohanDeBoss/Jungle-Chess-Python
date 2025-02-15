import tkinter as tk
from tkinter import ttk
import time
import random

# -----------------------------
# Global Constants
# -----------------------------
ROWS, COLS = 8, 8
SQUARE_SIZE = 65
BOARD_COLOR_1 = "#D2B48C"
BOARD_COLOR_2 = "#8B5A2B"
HIGHLIGHT_COLOR = "#ADD8E6"
movedelay = 400 # milliseconds

DIRECTIONS = {
    'king': [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
    'queen': [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
    'rook': [(0, 1), (0, -1), (1, 0), (-1, 0)],
    'bishop': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    'knight': [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
}

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


class King(Piece):
    def symbol(self):
        return "♔" if self.color == "white" else "♚"

    def get_valid_moves(self, board, pos):
        moves = []
        for d in DIRECTIONS['king']:
            for step in (1, 2):
                new_r = pos[0] + d[0] * step
                new_c = pos[1] + d[1] * step
                if 0 <= new_r < ROWS and 0 <= new_c < COLS:
                    if step == 2:
                        inter_r, inter_c = pos[0] + d[0], pos[1] + d[1]
                        if board[inter_r][inter_c] is not None:
                            break
                    target = board[new_r][new_c]
                    if target is None or target.color != self.color:
                        moves.append((new_r, new_c))
                    if target is not None:
                        break
        return moves


# In the Queen class, update move() as follows:

class Queen(Piece):
    def symbol(self):
        return "♕" if self.color == "white" else "♛"

    def get_valid_moves(self, board, pos):
        moves = []
        for d in DIRECTIONS['queen']:
            r, c = pos
            while True:
                r += d[0]
                c += d[1]
                if not (0 <= r < ROWS and 0 <= c < COLS):
                    break
                target = board[r][c]
                if target is None:
                    moves.append((r, c))
                else:
                    if target.color != self.color:
                        moves.append((r, c))
                    break
        return moves

    def move(self, board, start, end):
        # If capturing an enemy piece, remove that piece and trigger explosion.
        if board[end[0]][end[1]] and board[end[0]][end[1]].color != self.color:
            # Remove the target piece.
            board[end[0]][end[1]] = None
            # Loop through the 8 surrounding squares (adjacent neighbors) and clear enemy pieces.
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r = end[0] + dr
                    c = end[1] + dc
                    if 0 <= r < ROWS and 0 <= c < COLS:
                        if board[r][c] and board[r][c].color != self.color:
                            board[r][c] = None
            # Remove the queen from her original square.
            board[start[0]][start[1]] = None
            # The queen "dies" after capturing and exploding, so do not place her at the destination.
            board[end[0]][end[1]] = None
        else:
            board = super().move(board, start, end)
        self.has_moved = True
        return board


class Rook(Piece):
    def symbol(self):
        return "♖" if self.color == "white" else "♜"

    def get_valid_moves(self, board, pos):
        moves = []
        for d in DIRECTIONS['rook']:
            r, c = pos
            enemy_encountered = False
            while True:
                r += d[0]
                c += d[1]
                if not (0 <= r < ROWS and 0 <= c < COLS):
                    break
                target = board[r][c]
                if not enemy_encountered:
                    if target is None:
                        moves.append((r, c))
                    else:
                        if target.color != self.color:
                            moves.append((r, c))
                            enemy_encountered = True
                        else:
                            break
                else:
                    if target is None or target.color != self.color:
                        moves.append((r, c))
                    else:
                        break
        return moves

    def move(self, board, start, end):
        if start[0] == end[0]:
            d = (0, 1) if end[1] > start[1] else (0, -1)
        else:
            d = (1, 0) if end[0] > start[0] else (-1, 0)
        r, c = start
        path = []
        while (r, c) != end:
            r += d[0]
            c += d[1]
            path.append((r, c))
        if any(board[r][c] and board[r][c].color != self.color for r, c in path):
            for r, c in path:
                if board[r][c] and board[r][c].color != self.color:
                    board[r][c] = None
        super().move(board, start, end)
        return board


def get_zigzag_moves(board, pos, color):
    moves = set()
    r, c = pos
    direction_pairs = [
        ((-1, 1), (-1, -1)), ((-1, -1), (-1, 1)),
        ((1, 1), (1, -1)), ((1, -1), (1, 1)),
        ((-1, 1), (1, 1)), ((1, 1), (-1, 1)),
        ((-1, -1), (1, -1)), ((1, -1), (-1, -1))
    ]
    for d1, d2 in direction_pairs:
        cr, cc, cd = r, c, d1
        while True:
            cr += cd[0]
            cc += cd[1]
            if not (0 <= cr < ROWS and 0 <= cc < COLS):
                break
            piece = board[cr][cc]
            if piece:
                if piece.color != color:
                    moves.add((cr, cc))
                break
            moves.add((cr, cc))
            cd = d2 if cd == d1 else d1
    return list(moves)


def get_diagonal_moves(board, pos, color):
    moves = []
    for d in DIRECTIONS['bishop']:
        r, c = pos
        while True:
            r += d[0]
            c += d[1]
            if not (0 <= r < ROWS and 0 <= c < COLS):
                break
            piece = board[r][c]
            if piece:
                if piece.color != color:
                    moves.append((r, c))
                break
            moves.append((r, c))
    return moves


class Bishop(Piece):
    def symbol(self):
        return "♗" if self.color == "white" else "♝"

    def get_valid_moves(self, board, pos):
        return list(set(get_zigzag_moves(board, pos, self.color) + get_diagonal_moves(board, pos, self.color)))

    def move(self, board, start, end):
        return super().move(board, start, end)


class Knight(Piece):
    def symbol(self):
        return "♘" if self.color == "white" else "♞"

    def get_valid_moves(self, board, pos):
        return [
            (pos[0] + dr, pos[1] + dc)
            for dr, dc in DIRECTIONS['knight']
            if 0 <= (pos[0] + dr) < ROWS and 0 <= (pos[1] + dc) < COLS
            and (not (piece := board[pos[0]+dr][pos[1]+dc]) or piece.color != self.color)
        ]

    def move(self, board, start, end):
        super().move(board, start, end)
        self.evaporate(board, end)
        return board

    def evaporate(self, board, pos):
        enemy_knights = []
        for dr, dc in DIRECTIONS['knight']:
            r, c = pos[0] + dr, pos[1] + dc
            if 0 <= r < ROWS and 0 <= c < COLS:
                piece = board[r][c]
                if piece and piece.color != self.color:
                    if isinstance(piece, Knight):
                        enemy_knights.append((r, c))
                    board[r][c] = None
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
            if steps == 2 and self.has_moved:
                continue
                
            new_r = pos[0] + (direction * steps)
            new_c = pos[1]
            
            if 0 <= new_r < ROWS:
                if board[new_r][new_c] is None or (board[new_r][new_c] is not None and board[new_r][new_c].color != self.color):
                    moves.append((new_r, new_c))
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
        # Handle capture if present.
        if board[end[0]][end[1]] is not None and board[end[0]][end[1]].color != self.color:
            board[end[0]][end[1]] = None
        board[end[0]][end[1]] = self
        board[start[0]][start[1]] = None
        self.has_moved = True
        
        # Promotion logic: white promotes when reaching row 0, black when reaching the last row.
        if (self.color == "white" and end[0] == 0) or (self.color == "black" and end[0] == ROWS - 1):
            board[end[0]][end[1]] = Queen(self.color)
        
        return board


def copy_board(board):
    return [[p.clone() if p else None for p in row] for row in board]

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
        self.tt = {}  # Transposition table
        self.nodes_searched = 0
        self.app = app  # Store the EnhancedChessApp instance
        

    def is_in_explosion_threat(self, board, color):
        """Check if the current player's king is under threat of explosion from an enemy queen's capture."""
        # Find the king's position of the given color
        king_pos = None
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if isinstance(piece, King) and piece.color == color:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        if not king_pos:
            return False  # King not found

        # Check all enemy queens' moves to squares adjacent to the king
        enemy_color = 'black' if color == 'white' else 'white'
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if isinstance(piece, Queen) and piece.color == enemy_color:
                    queen_moves = piece.get_valid_moves(board, (r, c))
                    for move in queen_moves:
                        # Check if move is adjacent to the king's position
                        if max(abs(move[0] - king_pos[0]), abs(move[1] - king_pos[1])) == 1:
                            # Check if the move captures a piece of 'color' (current player)
                            target_piece = board[move[0]][move[1]]
                            if target_piece and target_piece.color == color:
                                return True
        return False

    def is_in_knight_evaporation_threat(self, board, color):
        """Check if any of the player's knights are in positions where enemy knights can attack via evaporation."""
        enemy_color = 'black' if color == 'white' else 'white'
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if isinstance(piece, Knight) and piece.color == color:
                    # Check all squares a knight can move to (L-shaped) for enemy knights
                    for dr, dc in DIRECTIONS['knight']:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                            target_piece = board[nr][nc]
                            if isinstance(target_piece, Knight) and target_piece.color == enemy_color:
                                return True
        return False


    # Update the evaluate_board method to remove threefold check
    def evaluate_board(self, board, depth, current_turn):
        piece_values = {
            Pawn: 100,
            Knight: 700,
            Bishop: 600,
            Rook: 500,
            Queen: 900,
            King: 100000
        }
        
        score = 0
        our_king_pos = enemy_king_pos = None
        
        # First pass: find kings and calculate material score
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if not piece:
                    continue
                    
                # Track king positions
                if isinstance(piece, King):
                    if piece.color == self.color:
                        our_king_pos = (r, c)
                    else:
                        enemy_king_pos = (r, c)
                    continue
                
                # Calculate piece value with bonuses
                value = piece_values.get(type(piece), 0)
                if isinstance(piece, Knight):
                    value += len(piece.get_valid_moves(board, (r, c))) * 10
                elif isinstance(piece, Queen):
                    atomic_threats = sum(
                        1 for move in piece.get_valid_moves(board, (r, c))
                        if any(isinstance(board[adj_r][adj_c], King)
                            for adj_r, adj_c in self.get_adjacent_squares(move))
                    )
                    value += atomic_threats * 20
                
                score += value if piece.color == self.color else -value

        # Check win/loss conditions
        if not enemy_king_pos:
            return float('inf')
        if not our_king_pos:
            return float('-inf')

        # Evaluate explosion threats from enemy queens.
        explosion_threat = self.is_in_explosion_threat(board, self.color)

        # Evaluate threats to our king (existing approach).
        threat_score = sum(
            200 for r in range(ROWS) for c in range(COLS)
            if (piece := board[r][c]) and piece.color != self.color
            for move in piece.get_valid_moves(board, (r, c))
            if manhattan_distance(move, our_king_pos) <= 1
        )

        # Calculate original evaluation
        original_eval = int(score - threat_score - explosion_threat)

        return original_eval
    # Modify the ChessBot's minimax function:
    def minimax(self, board, depth, maximizing_player, alpha, beta):
        self.nodes_searched += 1
        
        # Determine current turn for this node
        current_turn = self.color if maximizing_player else ('black' if self.color == 'white' else 'white')
        # Generate current position key
        current_key = generate_position_key(board, current_turn)
        # Check if this position has occurred twice in the game history (third occurrence)
        count = self.app.position_history.count(current_key)
        if count >= 2:
            return 0  # Draw score
        
        board_key = self.board_hash(board)
        if board_key in self.tt and self.tt[board_key][0] >= depth:
            return self.tt[board_key][1]

        if depth == 0:
            return self.evaluate_board(board, depth, current_turn)

        moves = self.get_all_moves(board, current_turn)
        if not moves:
            return self.evaluate_board(board, depth, current_turn)

        best_move = None
        value = float('-inf') if maximizing_player else float('inf')
        for move in moves:
            start, end = move
            new_board = self.simulate_move(board, start, end)
            eval_value = self.minimax(new_board, depth - 1, not maximizing_player, alpha, beta)
            
            if (maximizing_player and eval_value > value) or (not maximizing_player and eval_value < value):
                value = eval_value
                best_move = move
            
            if maximizing_player:
                alpha = max(alpha, value)
            else:
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
            reported_value = -current_best_value if self.color == 'black' else current_best_value
            print(f"Depth {current_depth}: {iteration_time:.3f}s, "
                f"nodes: {self.nodes_searched}, value: {reported_value}")
            
            self.app.master.after(0, lambda: self.app.draw_eval_bar(reported_value))
                
            best_move = current_best_move
            best_value = current_best_value
            
        print(f"Total time: {(time.time() - total_start):.3f}s")
        
        if best_move:
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
            piece.symbol() if piece else '.'
            for row in board for piece in row
        )
        return hash(board_str)

    def get_adjacent_squares(self, pos):
        """Get all adjacent squares for a given position."""
        r, c = pos
        return [
            (adj_r, adj_c) for adj_r in range(r-1, r+2) for adj_c in range(c-1, c+2)
            if 0 <= adj_r < ROWS and 0 <= adj_c < COLS and (adj_r, adj_c) != (r, c)
        ]

    def simulate_move(self, board, start, end):
        """Simulate a move on the board and return the new board state."""
        new_board = copy_board(board)
        piece = new_board[start[0]][start[1]]
        new_board = piece.move(new_board, start, end)
        check_evaporation(new_board)  # Apply evaporation effect
        return new_board


    def get_all_moves(self, board, color):
        """Get all legal moves for the given color."""
        return [
            ((r, c), move)
            for r in range(ROWS) for c in range(COLS)
            if (piece := board[r][c]) and piece.color == color
            for move in piece.get_valid_moves(board, (r, c))
            if validate_move(board, color, (r, c), move)
        ]
        
# -----------------------------
# Threat Checking Utilities
# -----------------------------
def is_in_check(board, color):
    """Check if the given color is in check."""
    king_pos = next(
        ((r, c) for r in range(ROWS) for c in range(COLS)
        if isinstance(board[r][c], King) and board[r][c].color == color),
        None
    )
    if not king_pos:
        return False

    return any(
        king_pos in piece.get_valid_moves(board, (r, c))
        for r in range(ROWS) for c in range(COLS)
        if (piece := board[r][c]) and piece.color != color
    )

def is_in_explosion_threat(board, color):
    """Check if king is threatened by queen explosion capture."""
    king_pos = None
    for r in range(ROWS):
        for c in range(COLS):
            if isinstance(board[r][c], King) and board[r][c].color == color:
                king_pos = (r, c)
                break
        if king_pos:
            break
    if not king_pos:
        return False

    enemy_color = 'black' if color == 'white' else 'white'
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if isinstance(piece, Queen) and piece.color == enemy_color:
                for move in piece.get_valid_moves(board, (r, c)):
                    if max(abs(move[0] - king_pos[0]), abs(move[1] - king_pos[1])) == 1:
                        target = board[move[0]][move[1]]
                        if target and target.color == color:
                            return True
    return False

def validate_move(board, color, start, end):
    simulated = copy_board(board)
    piece = simulated[start[0]][start[1]]
    if not piece or piece.color != color:
        return False
    
    simulated = piece.move(simulated, start, end)
    check_evaporation(simulated)
    
    if is_in_check(simulated, color):
        return False
    
    if is_in_explosion_threat(simulated, color):
        return False
    
    if is_king_in_knight_evaporation_danger(simulated, color):
        return False
    
    # Only check knight attack if the king is moving.
    if isinstance(piece, King) and is_king_attacked_by_knight(simulated, color, end):
        return False
    
    return True

def is_king_attacked_by_knight(board, color, king_pos):
    """Check if the king is moving into a square attacked by an enemy knight."""
    enemy_color = 'black' if color == 'white' else 'white'
    
    # Check all enemy knights
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if isinstance(piece, Knight) and piece.color == enemy_color:
                # Check if the king's new position is a knight's move away from this knight
                for dr, dc in DIRECTIONS['knight']:
                    if (r + dr) == king_pos[0] and (c + dc) == king_pos[1]:
                        return True
    return False


def is_king_in_knight_evaporation_danger(board, color):
    """Check if the king is in a position where it could be evaporated on the next turn."""
    # Find the king's position
    king_pos = None
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if isinstance(piece, King) and piece.color == color:
                king_pos = (r, c)
                break
        if king_pos:
            break
    if not king_pos:
        return False  # King not found
    
    enemy_color = 'black' if color == 'white' else 'white'
    
    # Check all enemy knights
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if isinstance(piece, Knight) and piece.color == enemy_color:
                # Get all valid moves for this knight
                valid_moves = piece.get_valid_moves(board, (r, c))
                for move in valid_moves:
                    # Check if the king's position is a knight's move away from this move
                    for dr, dc in DIRECTIONS['knight']:
                        kr, kc = king_pos
                        if (move[0] + dr) == kr and (move[1] + dc) == kc:
                            return True
    return False

    # Add the helper function at the top level (outside any class)
def generate_position_key(board, turn):
    key_parts = []
    for row in board:
        for piece in row:
            if piece:
                key_parts.append(piece.symbol())
                key_parts.append('1' if piece.has_moved else '0')
            else:
                key_parts.append('..')  # Represent empty squares
    key_parts.append(turn)
    return ''.join(key_parts)

class EnhancedChessApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Enhanced Chess")
        self.COLORS = self.setup_styles()
        self.master.configure(bg=self.COLORS['bg_dark'])
        self.position_history = []  # Track positions for threefold repetition

        # Window setup
        screen_w = self.master.winfo_screenwidth()
        screen_h = self.master.winfo_screenheight()
        self.master.geometry(f"{screen_w}x{screen_h}+0+0")
        self.master.state('zoomed')
        self.fullscreen = True
        self.master.bind("<Configure>", self.on_configure)
        
        # Main frame setup
        self.main_frame = ttk.Frame(master, style='Left.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Left panel (sidebar)
        self.left_panel = ttk.Frame(self.main_frame, width=250, style='Left.TFrame')
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=(0,15))
        self.left_panel.pack_propagate(False)
        
        # Header label
        ttk.Label(self.left_panel, text="JUNGLE CHESS", style='Header.TLabel',
                font=('Helvetica', 24, 'bold')).pack(pady=(0,10))
        
        # Game mode frame setup
        self.game_mode = tk.StringVar(value="bot")
        game_mode_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        game_mode_frame.pack(fill=tk.X, pady=(0,10))
        ttk.Label(game_mode_frame, text="GAME MODE", style='Header.TLabel').pack(anchor=tk.W)
        ttk.Radiobutton(game_mode_frame, text="Human vs Bot", variable=self.game_mode,
                        value="bot", command=self.reset_game, style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(5,3))
        ttk.Radiobutton(game_mode_frame, text="Human vs Human", variable=self.game_mode,
                        value="human", command=self.reset_game, style='Custom.TRadiobutton').pack(anchor=tk.W)
        
        # Controls frame setup
        controls_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        controls_frame.pack(fill=tk.X, pady=5)
        ttk.Button(controls_frame, text="NEW GAME", command=self.reset_game,
                style='Control.TButton').pack(fill=tk.X, pady=5)
        ttk.Button(controls_frame, text="BOT SETTINGS", command=self.open_settings,
                style='Control.TButton').pack(fill=tk.X, pady=5)
        ttk.Button(controls_frame, text="SWAP SIDES", command=self.swap_sides,
                style='Control.TButton').pack(fill=tk.X, pady=5)
        ttk.Button(controls_frame, text="QUIT", command=self.master.quit,
                style='Control.TButton').pack(fill=tk.X, pady=5)
        
        # Turn display frame
        self.turn_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        self.turn_frame.pack(fill=tk.X, pady=(10,0))
        self.turn_label = ttk.Label(self.turn_frame, text="WHITE'S TURN", style='Status.TLabel')
        self.turn_label.pack(fill=tk.X)
        
        # Evaluation frame setup
        self.eval_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        self.eval_frame.pack(side=tk.TOP, fill=tk.Y, padx=15, pady=15)
        self.eval_bar_canvas = tk.Canvas(self.eval_frame, width=300, height=30,
                                        bg=self.COLORS['bg_light'], highlightthickness=0)
        self.eval_bar_canvas.pack(side=tk.BOTTOM, pady=(10, 5))
        self.eval_score_label = ttk.Label(self.eval_frame, text="", style='Status.TLabel')
        self.eval_score_label.pack()
        self.draw_eval_bar(0)
        self.eval_bar_visible = True

        # Right panel (main game board)
        self.right_panel = ttk.Frame(self.main_frame, style='Right.TFrame')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.canvas_container = ttk.Frame(self.right_panel, style='Canvas.TFrame')
        self.canvas_container.pack(expand=True)
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)
        self.canvas_frame = ttk.Frame(self.canvas_container, style='Canvas.TFrame')
        self.canvas_frame.grid(row=0, column=0)
        self.canvas = tk.Canvas(self.canvas_frame,
                                width=COLS * SQUARE_SIZE,
                                height=ROWS * SQUARE_SIZE,
                                bg=self.COLORS['bg_light'],
                                highlightthickness=2,
                                highlightbackground=self.COLORS['accent'])
        self.canvas.pack()

        # Initial game state initialization
        self.human_color = "white"
        self.board = create_initial_board()
        self.turn = "white"
        self.selected = None
        self.valid_moves = []
        self.game_over = False
        self.dragging = False
        self.drag_piece = None
        self.drag_start = None
        bot_color = "black"  # Opponent always plays opposite
        self.bot = ChessBot(self.board, bot_color, self)
        
        # Bind canvas events and initial drawing
        self.canvas.bind("<Button-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.draw_board()
    
    # Update EnhancedChessApp's get_position_key method
    def get_position_key(self):
        return generate_position_key(self.board, self.turn)

    def swap_sides(self):
        # Swap which color the human plays. After swapping, the board is redrawn from the new perspective.
        self.human_color = "black" if self.human_color == "white" else "white"
        # Force human to play the new color; bot gets the opposite.
        bot_color = "black" if self.human_color == "white" else "white"
        self.bot = ChessBot(self.board, bot_color, self)
        self.turn = self.human_color  # Let human start.
        self.turn_label.config(text=f"Turn: {self.human_color.capitalize()}")
        self.draw_board()

    # Coordinate conversion helpers:
    def board_to_canvas(self, r, c):
        # For a 180° rotated view when playing as black.
        if self.human_color == "white":
            x1 = c * SQUARE_SIZE
            y1 = r * SQUARE_SIZE
        else:
            x1 = (COLS - 1 - c) * SQUARE_SIZE
            y1 = (ROWS - 1 - r) * SQUARE_SIZE
        return x1, y1

    def canvas_to_board(self, x, y):
        # Convert canvas coordinates to board indices based on current perspective.
        if self.human_color == "white":
            row = y // SQUARE_SIZE
            col = x // SQUARE_SIZE
        else:
            row = (ROWS - 1) - (y // SQUARE_SIZE)
            col = (COLS - 1) - (x // SQUARE_SIZE)
        return row, col

    def on_configure(self, event):
        pass

    def open_settings(self):
        settings_win = tk.Toplevel(self.master)
        settings_win.title("Bot Settings")
        win_width, win_height = 300, 150
        screen_width = settings_win.winfo_screenwidth()
        screen_height = settings_win.winfo_screenheight()
        x = (screen_width - win_width) // 2
        y = (screen_height - win_height) // 2
        settings_win.geometry(f"{win_width}x{win_height}+{x}+{y}")
        ttk.Label(settings_win, text="Bot Search Depth:", font=('Helvetica', 12)).pack(pady=(20, 5))
        depth_var = tk.IntVar(value=ChessBot.search_depth)
        spin = ttk.Spinbox(settings_win, from_=1, to=6, textvariable=depth_var, width=5)
        spin.pack(pady=(0, 20))
        eval_bar_var = tk.BooleanVar(value=self.eval_bar_visible)
        ttk.Checkbutton(settings_win, text="Show Evaluation Bar", variable=eval_bar_var).pack(pady=(0, 10))
        
        def apply_settings():
            new_depth = depth_var.get()
            ChessBot.search_depth = new_depth
            if hasattr(self, 'bot'):
                self.bot.search_depth = new_depth
            self.eval_bar_visible = eval_bar_var.get()
            if self.eval_bar_visible:
                self.eval_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=15, pady=15)
            else:
                self.eval_frame.pack_forget()
            settings_win.destroy()
        
        ttk.Button(settings_win, text="Apply", command=apply_settings).pack()

    def draw_eval_bar(self, eval_score):
        eval_score /= 100.0
        self.eval_bar_canvas.delete("all")
        bar_width = 235
        bar_height = 30
        max_eval = 10.0
        neutral_zone = 0.2
        normalized_score = max(min(eval_score / max_eval, 1.0), -1.0)
        for x in range(bar_width):
            ratio = x / float(bar_width)
            r = int(255 * ratio)
            g = int(255 * ratio)
            b = int(255 * ratio)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.eval_bar_canvas.create_line(x, 0, x, bar_height, fill=color)
        marker_x = int((normalized_score + 1) / 2 * bar_width)
        accent_color = self.COLORS.get('accent', '#e94560')
        marker_width = 2
        self.eval_bar_canvas.create_rectangle(marker_x - marker_width, 0,
                                              marker_x + marker_width, bar_height,
                                              fill=accent_color, outline="")
        mid_x = (bar_width // 2)
        self.eval_bar_canvas.create_line(mid_x, 0, mid_x, bar_height, fill="#666666", width=1)
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
        COLORS = {
            'bg_dark': '#1a1a2e',
            'bg_medium': '#16213e',
            'bg_light': '#0f3460',
            'accent': '#e94560',
            'text_light': '#ffffff',
            'text_dark': '#a2a2a2'
        }
        style.configure('Left.TFrame', background=COLORS['bg_dark'])
        style.configure('Right.TFrame', background=COLORS['bg_medium'])
        style.configure('Canvas.TFrame', background=COLORS['bg_medium'])
        style.configure('Header.TLabel',
                        background=COLORS['bg_dark'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 14, 'bold'),
                        padding=(0, 10))
        style.configure('Status.TLabel',
                        background=COLORS['bg_light'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 16, 'bold'),
                        padding=(15, 10),
                        relief='flat',
                        borderwidth=0)
        # Adjusted button style: reduced padding for a skinnier red button look.
        style.configure('Control.TButton',
                        background=COLORS['accent'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 11, 'bold'),
                        padding=(10, 8),   # Reduced padding from (15, 12)
                        borderwidth=0,
                        relief='flat')
        style.map('Control.TButton',
                  background=[('active', COLORS['accent']),
                              ('pressed', '#d13550')],
                  relief=[('pressed', 'flat'),
                          ('!pressed', 'flat')])
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
        # Draw squares
        for r in range(ROWS):
            for c in range(COLS):
                x1, y1 = self.board_to_canvas(r, c)
                x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
                color = BOARD_COLOR_1 if (r + c) % 2 == 0 else BOARD_COLOR_2
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                # highlight valid moves if needed
                if (r, c) in self.valid_moves:
                    self.canvas.create_oval(x1 + 19, y1 + 19, x2 - 19, y2 - 19,
                                              fill="#1E90FF", outline="#1E90FF", width=3)
        # Draw pieces
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece is not None and (r, c) != self.drag_start:
                    x, y = self.board_to_canvas(r, c)
                    x_center = x + SQUARE_SIZE // 2
                    y_center = y + SQUARE_SIZE // 2
                    symbol = piece.symbol()
                    if piece.color == "white":
                        shadow_offset = 2
                        shadow_color = "#444444"
                        self.canvas.create_text(x_center + shadow_offset, y_center + shadow_offset,
                                                text=symbol, font=("Arial", 39), fill=shadow_color, tags="piece")
                        self.canvas.create_text(x_center, y_center,
                                                text=symbol, font=("Arial Unicode MS", 39),
                                                fill="white", tags="piece")
                    else:
                        self.canvas.create_text(x_center, y_center, text=symbol,
                                                font=("Arial", 39), fill="black", tags="piece")
        # Draw dragging piece if any
        if self.dragging and self.drag_piece:
            # Draw piece following the mouse (using raw event coordinates)
            piece = self.board[self.drag_start[0]][self.drag_start[1]]
            self.canvas.create_text(self.drag_piece[0], self.drag_piece[1],
                                    text=piece.symbol(), font=("Arial", 36), tags="drag")

    def draw_piece(self, r, c):
        piece = self.board[r][c]
        if piece is not None and (r, c) != self.drag_start:
            x = c * SQUARE_SIZE + SQUARE_SIZE // 2
            y = r * SQUARE_SIZE + SQUARE_SIZE // 2
            symbol = piece.symbol()
            if piece.color == "white":
                shadow_offset = 2
                shadow_color = "#444444"
                self.canvas.create_text(x + shadow_offset, y + shadow_offset, text=symbol,
                                        font=("Arial", 39), fill=shadow_color, tags="piece")
                self.canvas.create_text(x, y, text=symbol,
                                        font=("Arial Unicode MS", 39),
                                        fill="white", tags="piece")
            else:
                self.canvas.create_text(x, y, text=symbol,
                                        font=("Arial", 39),
                                        fill="black", tags="piece")

    def on_drag_start(self, event):
        if self.game_over:
            return
        row, col = self.canvas_to_board(event.x, event.y)
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

        row, col = self.canvas_to_board(event.x, event.y)
        end_pos = (row, col)

        if end_pos in self.valid_moves:
            # Validate move against game rules
            if validate_move(self.board, self.turn, self.drag_start, end_pos):
                moving_piece = self.board[self.drag_start[0]][self.drag_start[1]]
                self.board = moving_piece.move(self.board, self.drag_start, end_pos)
                check_evaporation(self.board)

                # Update position history
                self.position_history.append(self.get_position_key())

                winner = check_game_over(self.board)
                if winner is not None:
                    self.game_over = True
                    self.turn_label.config(text=f"{winner.capitalize()} wins!")
                else:
                    # Switch turns
                    # Update EnhancedChessApp's get_position_key method
                    self.turn = "black" if self.turn == "white" else "white"
                    # Then update position history
                    self.position_history.append(self.get_position_key())                    # If playing against the bot and it's the bot's turn, schedule the bot's move
                    if self.game_mode.get() == "bot" and self.turn != self.human_color:
                        self.master.after(movedelay, self.make_bot_move)
            else:
                print("Illegal move - would leave king in danger or knight under threat!")

        # Reset dragging state
        self.dragging = False
        self.drag_piece = None
        self.drag_start = None
        self.valid_moves = []
        self.draw_board()

    def make_bot_move(self):
        if self.game_over:
            return

        # Start timing for debug output
        start_time = time.time()

        # Make the bot's move
        if self.bot.make_move():
            # Update the board display immediately
            self.draw_board()

            # Check for a winner after the bot's move
            winner = check_game_over(self.board)
            if winner is not None:
                self.game_over = True
                self.turn_label.config(text=f"{winner.capitalize()} wins!")
            else:
                # Switch turns back to the human player
                self.turn = self.human_color  # Switch turns
                self.position_history.append(self.get_position_key())  # Update history

            # Debug output for timing
            elapsed_time = time.time() - start_time
            print(f"Bot move completed in {elapsed_time:.3f} seconds")
        else:
            # If the bot cannot make a move, the human wins
            self.game_over = True
            self.turn_label.config(text=f"{self.human_color.capitalize()} wins!")

    def swap_sides(self):
        self.human_color = "black" if self.human_color == "white" else "white"
        self.reset_game()

    def reset_game(self):
        self.board = create_initial_board()
        # Always start with white's turn.
        self.turn = "white"  
        self.selected = None
        self.valid_moves = []
        self.game_over = False
        self.dragging = False
        self.drag_piece = None
        self.drag_start = None
        # Bot takes the color that is not chosen by the human.
        bot_color = "black" if self.human_color == "white" else "white"
        self.bot = ChessBot(self.board, bot_color, self)
        self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")
        self.draw_board()
        # If playing in bot mode and it's not the human's turn, let the bot move.
        if self.game_mode.get() == "bot" and self.turn != self.human_color:
            self.master.after(500, self.make_bot_move)

def main():
    root = tk.Tk()
    app = EnhancedChessApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()