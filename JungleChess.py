import tkinter as tk
from tkinter import ttk
import time
import math

# -----------------------------
# Global Constants
# -----------------------------
ROWS, COLS = 8, 8
SQUARE_SIZE = 65
BOARD_COLOR_1 = "#D2B48C"
BOARD_COLOR_2 = "#8B5A2B"
HIGHLIGHT_COLOR = "#ADD8E6"
MOVEDELAY = 400  # milliseconds

DIRECTIONS = {
    'king': ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'queen': ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'rook': ((0, 1), (0, -1), (1, 0), (-1, 0)),
    'bishop': ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    'knight': ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
}

ADJACENT_DIRS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1),  (1, 0), (1, 1)
)


# -----------------------------
# Piece Base Class and Subclasses
# -----------------------------
class Piece:
    """Base class for all chess pieces"""
    def __init__(self, color):
        self.color = color
        self.has_moved = False

    def clone(self):
        """Create a deep copy of the piece"""
        new_piece = self.__class__(self.color)
        new_piece.has_moved = self.has_moved
        return new_piece

    def save_state(self):
        """Save current state for potential rollback"""
        return {'has_moved': self.has_moved}

    def restore_state(self, state):
        """Restore state from saved data"""
        self.has_moved = state['has_moved']

    def symbol(self):
        """Unicode character representing the piece"""
        return "?"

    def get_valid_moves(self, board, pos):
        """Return list of valid moves (to be implemented by subclasses)"""
        return []

    def move(self, board, start, end):
        """
        Execute move on the board
        Returns modified board state
        """
        if board[end[0]][end[1]] is not None and board[end[0]][end[1]].color != self.color:
            board[end[0]][end[1]] = None
        board[end[0]][end[1]] = self
        board[start[0]][start[1]] = None
        self.has_moved = True
        return board


class King(Piece):
    """King piece implementation"""
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


class Queen(Piece):
    """Queen piece implementation with explosion mechanics on capture"""
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
        # If the queen captures an enemy piece, trigger explosion
        if board[end[0]][end[1]] and board[end[0]][end[1]].color != self.color:
            # Remove the target piece
            board[end[0]][end[1]] = None
            # Loop through the 8 surrounding squares (adjacent neighbors) and clear enemy pieces
            for dr, dc in ADJACENT_DIRS:
                r = end[0] + dr
                c = end[1] + dc
                if 0 <= r < ROWS and 0 <= c < COLS:
                    if board[r][c] and board[r][c].color != self.color:
                        board[r][c] = None
            # Remove the queen from her original square
            board[start[0]][start[1]] = None
            # The queen "dies" after capturing and exploding, so do not place her at the destination
            board[end[0]][end[1]] = None
        else:
            # Normal move (no capture, no explosion)
            board = super().move(board, start, end)
        self.has_moved = True
        return board
        

class Rook(Piece):
    """Rook piece implementation with path clearing"""
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


class Bishop(Piece):
    """Bishop piece implementation with complex movement patterns"""
    def symbol(self):
        return "♗" if self.color == "white" else "♝"

    def get_valid_moves(self, board, pos):
        return list(set(get_zigzag_moves(board, pos, self.color) + get_diagonal_moves(board, pos, self.color)))

    def move(self, board, start, end):
        return super().move(board, start, end)


class Knight(Piece):
    """Knight piece implementation with evaporation mechanics"""
    def symbol(self):
        return "♘" if self.color == "white" else "♞"

    def get_valid_moves(self, board, pos):
        return [
            (pos[0] + dr, pos[1] + dc)
            for dr, dc in DIRECTIONS['knight']
            if 0 <= (pos[0] + dr) < ROWS 
            and 0 <= (pos[1] + dc) < COLS
            and (not (piece := board[pos[0]+dr][pos[1]+dc]) 
            or piece.color != self.color)
        ]

    def move(self, board, start, end):
        super().move(board, start, end)
        self.evaporate(board, end)
        return board

    def evaporate(self, board, pos):
        """Remove surrounding enemy pieces after move"""
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
    """Pawn piece implementation with variant movement rules"""
    def symbol(self):
        return "♙" if self.color == "white" else "♟"

    def get_valid_moves(self, board, pos):
        moves = []
        direction = -1 if self.color == "white" else 1
        starting_row = 6 if self.color == "white" else 1

        # Forward moves (1 or 2 squares based on rank)
        for steps in [1, 2]:
            if steps == 2 and pos[0] != starting_row:
                continue  # Allow 2-square move only from starting rank

            new_r = pos[0] + (direction * steps)
            new_c = pos[1]

            if 0 <= new_r < ROWS:
                # Forward capture: if an enemy is directly ahead
                if board[new_r][new_c] is not None and board[new_r][new_c].color != self.color:
                    moves.append((new_r, new_c))
                # Normal forward move: if square is empty
                if board[new_r][new_c] is None:
                    moves.append((new_r, new_c))
                if board[new_r][new_c] is not None:
                    break  # Block further movement if occupied

        # Sideways captures at the current rank
        for dc in [-1, 1]:
            new_c = pos[1] + dc
            if 0 <= new_c < COLS:
                if board[pos[0]][new_c] is not None and board[pos[0]][new_c].color != self.color:
                    moves.append((pos[0], new_c))
        
        return moves

    def move(self, board, start, end):
        board = super().move(board, start, end)
        # Determine promotion rank: 1st row for white, last row for black
        promotion_rank = 0 if self.color == "white" else (ROWS - 1)
        if end[0] == promotion_rank:
            # Promote pawn to queen
            board[end[0]][end[1]] = Queen(self.color)
        return board
        
# -----------------------------
# Helper Functions
# -----------------------------
def get_zigzag_moves(board, pos, color):
    """Calculate bishop's special zig-zag pattern moves"""
    moves = set()
    r, c = pos
    direction_pairs = (
        ((-1, 1), (-1, -1)), ((-1, -1), (-1, 1)),
        ((1, 1), (1, -1)), ((1, -1), (1, 1)),
        ((-1, 1), (1, 1)), ((1, 1), (-1, 1)),
        ((-1, -1), (1, -1)), ((1, -1), (-1, -1))
    )
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
    """Standard diagonal moves calculation"""
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


def copy_board(board):
    """Create a deep copy of the game board"""
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

def has_legal_moves(board, color):
    """Check if the player has any legal moves."""
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece and piece.color == color:
                moves = piece.get_valid_moves(board, (r, c))
                for move in moves:
                    if validate_move(board, color, (r, c), move):
                        return True
    return False

def check_game_over(board):
    """Check if the game is over due to checkmate, stalemate, or king capture."""
    # Check if kings are still on the board
    white_king_found = False
    black_king_found = False
    for row in board:
        for piece in row:
            if piece is not None and isinstance(piece, King):
                if piece.color == "white":
                    white_king_found = True
                elif piece.color == "black":
                    black_king_found = True
    
    # If a king is missing, the other player wins
    if not white_king_found:
        return "black"
    if not black_king_found:
        return "white"
    
    # Check for checkmate
    for color in ["white", "black"]:
        if is_in_check(board, color) and not has_legal_moves(board, color):
            return "checkmate", "black" if color == "white" else "white"
    
    # Check for stalemate
    for color in ["white", "black"]:
        if not has_legal_moves(board, color) and not is_in_check(board, color):
            return "stalemate", None
    
    return None, None

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

    # ====================================================
    # Threat Checking Methods
    # ====================================================
    def is_in_explosion_threat(self, board, color):
        """Check if the current player's king is under threat of explosion from an enemy queen's capture."""
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
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if isinstance(piece, Queen) and piece.color == enemy_color:
                    queen_moves = piece.get_valid_moves(board, (r, c))
                    for move in queen_moves:
                        if max(abs(move[0] - king_pos[0]), abs(move[1] - king_pos[1])) == 1:
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
                    for dr, dc in DIRECTIONS['knight']:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                            target_piece = board[nr][nc]
                            if isinstance(target_piece, Knight) and target_piece.color == enemy_color:
                                return True
        return False

    # ====================================================
    # Position Evaluation Methods
    # ====================================================
    def evaluate_board(self, board, depth, current_turn):
        """Evaluate the board state with early termination for winning/losing conditions."""
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

        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if not piece:
                    continue

                if isinstance(piece, King):
                    if piece.color == self.color:
                        our_king_pos = (r, c)
                    else:
                        enemy_king_pos = (r, c)
                    continue

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

        if not enemy_king_pos:
            return float('inf')
        if not our_king_pos:
            return float('-inf')

        explosion_threat = self.is_in_explosion_threat(board, self.color)
        threat_score = 0
        if our_king_pos:
            for r in range(ROWS):
                for c in range(COLS):
                    piece = board[r][c]
                    if piece and piece.color != self.color:
                        if our_king_pos in piece.get_valid_moves(board, (r, c)):
                            threat_score += 200  # Direct threat to king

        return int(score - threat_score - explosion_threat)

    # ====================================================
    # Move Ordering Methods
    # ====================================================
    def evaluate_move(self, board, move):
        """Heuristic to evaluate the quality of a move for move ordering."""
        start, end = move
        piece = board[start[0]][start[1]]
        target = board[end[0]][end[1]]

        # Base score for captures
        if target:
            # Prioritize capturing higher-value pieces
            piece_values = {
                Pawn: 100,
                Knight: 700,
                Bishop: 600,
                Rook: 500,
                Queen: 900,
                King: 100000
            }
            capture_score = piece_values.get(type(target), 0)
            return 1000 + capture_score  # Base capture bonus + piece value

        # Bonus for moving to central squares
        center_squares = {(3, 3), (3, 4), (4, 3), (4, 4)}
        if end in center_squares:
            return 50  # Small bonus for central control

        # Bonus for pawn advancement
        if isinstance(piece, Pawn):
            if piece.color == "white":
                return end[0]  # Higher score for advancing white pawns
            else:
                return 7 - end[0]  # Higher score for advancing black pawns

        return 0  # Default score for non-captures

    def order_moves(self, board, moves, maximizing_player=True):
        """
        Order moves using a heuristic evaluation.
        Optimized to avoid unnecessary sorting when possible.
        """
        if not moves:
            return moves

        # Evaluate all moves in one pass
        scored_moves = [(self.evaluate_move(board, move), move) for move in moves]

        # Sort only if there are multiple moves with different scores
        if len(set(score for score, _ in scored_moves)) > 1:
            scored_moves.sort(reverse=maximizing_player, key=lambda x: x[0])

        # Return only the moves (without scores)
        return [move for _, move in scored_moves]

    # ====================================================
    # Search Methods (Minimax & Move Selection)
    # ====================================================
    def minimax(self, board, depth, maximizing_player, alpha, beta):
        """Optimized minimax with alpha-beta pruning, null move pruning, LMR, and PVS."""
        self.nodes_searched += 1

        current_turn = self.color if maximizing_player else ('black' if self.color == 'white' else 'white')
        
        # Threefold repetition check
        current_key = generate_position_key(board, current_turn)
        if self.app.position_history.count(current_key) >= 2:
            return 0  # Evaluate repeated position as draw

        if is_stalemate(board, current_turn):
            return 0

        board_key = self.board_hash(board)
        if board_key in self.tt and self.tt[board_key][0] >= depth:
            return self.tt[board_key][1]

        if depth == 0:
            return self.evaluate_board(board, depth, current_turn)

        # -------- Null Move Pruning --------
        # Only apply if not in check and depth is sufficiently high
        if depth >= 3 and not is_in_check(board, current_turn):
            null_move_reduction = 2  # Reduction factor for null moves
            if maximizing_player:
                null_value = self.minimax(board, depth - 1 - null_move_reduction, False, alpha, beta)
                if null_value >= beta:
                    return null_value
            else:
                null_value = self.minimax(board, depth - 1 - null_move_reduction, True, alpha, beta)
                if null_value <= alpha:
                    return null_value
        # ------------------------------------

        moves = self.get_all_moves(board, current_turn)
        if not moves:
            if is_in_check(board, current_turn):
                return float('-inf') if maximizing_player else float('inf')
            else:
                return 0

        moves = self.order_moves(board, moves, maximizing_player)

        best_value = float('-inf') if maximizing_player else float('inf')
        first_move = True

        for i, move in enumerate(moves):
            # -------- Late Move Reductions (LMR) --------
            reduction = 0
            if i >= 3 and depth >= 4:  # For later moves when depth is sufficient (should be higher than null move)
                reduction = 1  # Reduction factor (tweakable based on experiments)
            new_depth = depth - 1 - reduction
            # ----------------------------------------------

            start, end = move
            new_board = self.simulate_move(board, start, end)

            if first_move:
                # For the first move, search with full window
                score = self.minimax(new_board, new_depth, not maximizing_player, alpha, beta)
                first_move = False
            else:
                # Principal Variation Search (PVS) with LMR applied:
                if maximizing_player:
                    score = self.minimax(new_board, new_depth, not maximizing_player, alpha, alpha + 1)
                    if score > alpha and score < beta:
                        score = self.minimax(new_board, new_depth, not maximizing_player, score, beta)
                else:
                    score = self.minimax(new_board, new_depth, not maximizing_player, beta - 1, beta)
                    if score < beta and score > alpha:
                        score = self.minimax(new_board, new_depth, not maximizing_player, alpha, score)

            if maximizing_player:
                if score > best_value:
                    best_value = score
                alpha = max(alpha, best_value)
            else:
                if score < best_value:
                    best_value = score
                beta = min(beta, best_value)

            if beta <= alpha:
                break  # Beta cut-off

        self.tt[board_key] = (depth, best_value)
        return best_value



    def make_move(self):
        """Make the best move found by the bot."""
        best_move = None
        best_value = float('-inf')
        total_start = time.time()

        for current_depth in range(1, self.search_depth + 1):
            self.nodes_searched = 0
            iteration_start = time.time()

            moves = self.get_all_moves(self.board, self.color)
            if not moves:
                return False

            moves = self.order_moves(self.board, moves, maximizing_player=True)
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
            print(f"Depth {current_depth}: {iteration_time:.3f}s, nodes: {self.nodes_searched}, value: {reported_value}")

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

    # ====================================================
    # Helper Methods
    # ====================================================
    def board_hash(self, board):
        board_str = ''.join(
            piece.symbol() if piece else '.' for row in board for piece in row
        )
        return hash(board_str)

    def get_adjacent_squares(self, pos):
        r, c = pos
        return [
            (adj_r, adj_c)
            for adj_r in range(r - 1, r + 2)
            for adj_c in range(c - 1, c + 2)
            if 0 <= adj_r < ROWS and 0 <= adj_c < COLS and (adj_r, adj_c) != (r, c)
        ]

    def simulate_move(self, board, start, end):
        new_board = copy_board(board)
        piece = new_board[start[0]][start[1]]
        new_board = piece.move(new_board, start, end)
        check_evaporation(new_board)
        return new_board

    def get_all_moves(self, board, color):
        return [
            ((r, c), move)
            for r in range(ROWS) for c in range(COLS)
            if (piece := board[r][c]) and piece.color == color
            for move in piece.get_valid_moves(board, (r, c))
            if validate_move(board, color, (r, c), move)
        ]

# -----------------------------
# Threat Checking Utilities (External)
# -----------------------------
def is_in_check(board, color):
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
    
    # NEW: Ensure the king still exists after the move (i.e. wasn't evaporated)
    king_exists = any(
        isinstance(simulated[r][c], King) and simulated[r][c].color == color 
        for r in range(ROWS) for c in range(COLS)
    )
    if not king_exists:
        return False

    if is_in_check(simulated, color):
        return False
    
    if is_in_explosion_threat(simulated, color):
        return False
    
    if is_king_in_knight_evaporation_danger(simulated, color):
        return False
    
    if isinstance(piece, King) and is_king_attacked_by_knight(simulated, color, end):
        return False
    
    return True

def is_king_attacked_by_knight(board, color, king_pos):
    enemy_color = 'black' if color == 'white' else 'white'
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if isinstance(piece, Knight) and piece.color == enemy_color:
                for dr, dc in DIRECTIONS['knight']:
                    if (r + dr) == king_pos[0] and (c + dc) == king_pos[1]:
                        return True
    return False

def is_king_in_knight_evaporation_danger(board, color):
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
        return False
    
    enemy_color = 'black' if color == 'white' else 'white'
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if isinstance(piece, Knight) and piece.color == enemy_color:
                valid_moves = piece.get_valid_moves(board, (r, c))
                for move in valid_moves:
                    for dr, dc in DIRECTIONS['knight']:
                        kr, kc = king_pos
                        if (move[0] + dr) == kr and (move[1] + dc) == kc:
                            return True
    return False

def generate_position_key(board, turn):
    key_parts = []
    for row in board:
        for piece in row:
            if piece:
                key_parts.append(piece.symbol())
                key_parts.append('1' if piece.has_moved else '0')
            else:
                key_parts.append('..')
    key_parts.append(turn)
    return ''.join(key_parts)

def is_stalemate(board, color):
    if is_in_check(board, color):
        return False
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece and piece.color == color:
                moves = piece.get_valid_moves(board, (r, c))
                for move in moves:
                    if validate_move(board, color, (r, c), move):
                        return False
    return True


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
        game_mode_frame.pack(fill=tk.X, pady=(0,9))
        ttk.Label(game_mode_frame, text="GAME MODE", style='Header.TLabel').pack(anchor=tk.W)
        ttk.Radiobutton(game_mode_frame, text="Human vs Bot", variable=self.game_mode,
                        value="bot", command=self.reset_game, style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(5,3))
        ttk.Radiobutton(game_mode_frame, text="Human vs Human", variable=self.game_mode,
                        value="human", command=self.reset_game, style='Custom.TRadiobutton').pack(anchor=tk.W)
        
       # Controls frame setup
        controls_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        controls_frame.pack(fill=tk.X, pady=4)
        ttk.Button(controls_frame, text="NEW GAME", command=self.reset_game,
                   style='Control.TButton').pack(fill=tk.X, pady=5)
        # Remove the BOT SETTINGS button
        ttk.Button(controls_frame, text="SWAP SIDES", command=self.swap_sides,
                   style='Control.TButton').pack(fill=tk.X, pady=5)
        ttk.Button(controls_frame, text="QUIT", command=self.master.quit,
                   style='Control.TButton').pack(fill=tk.X, pady=5)
# Inline Bot settings with a Bot Depth slider
        ttk.Label(controls_frame, text="Bot Depth:", style='Header.TLabel').pack(anchor=tk.W, pady=(10,0))
        self.bot_depth_slider = tk.Scale(controls_frame, from_=1, to=6, orient=tk.HORIZONTAL,
                                         command=self.update_bot_depth,
                                         bg=self.COLORS['bg_dark'], fg=self.COLORS['text_light'],
                                         highlightthickness=0)
        self.bot_depth_slider.set(ChessBot.search_depth)
        self.bot_depth_slider.pack(fill=tk.X, pady=(0,4))
        
        # Add an Instant Move checkmark
        self.instant_move = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls_frame, text="Instant Move", variable=self.instant_move,
                        style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(3,3))
        
        # Turn display frame
        self.turn_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        self.turn_frame.pack(fill=tk.X, pady=(9,0))
        self.turn_label = ttk.Label(self.turn_frame, text="WHITE'S TURN", style='Status.TLabel')
        self.turn_label.pack(fill=tk.X)
        
        # Updated Evaluation frame setup
        self.eval_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        # Remove horizontal padding so it matches the other elements
        self.eval_frame.pack(side=tk.TOP, fill=tk.X, padx=0, pady=(5,5))

        self.eval_score_label = ttk.Label(self.eval_frame, text="Even", style='Status.TLabel', anchor="center")
        # Pack without extra fill or vertical padding
        self.eval_score_label.pack(pady=(7,5))

        # The eval canvas remains the same:
        self.eval_bar_canvas = tk.Canvas(self.eval_frame, height=26,
                                        bg=self.COLORS['bg_light'], highlightthickness=0)
        self.eval_bar_canvas.pack(side=tk.BOTTOM, fill=tk.X, expand=True)
        self.eval_bar_canvas.bind("<Configure>", lambda event: self.draw_eval_bar(0))
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
        # In the __init__ method of EnhancedChessApp, change the canvas creation:
        self.canvas = tk.Canvas(self.canvas_frame,
                                width=COLS * SQUARE_SIZE,
                                height=ROWS * SQUARE_SIZE,
                                bg=self.COLORS['bg_light'],
                                highlightthickness=0)  # Remove built-in highlight border
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

    def update_bot_depth(self, value):
        new_depth = int(value)
        ChessBot.search_depth = new_depth
        self.bot.search_depth = new_depth
    
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
        bar_width = self.eval_bar_canvas.winfo_width() or 235
        bar_height = 30
        max_eval = 10.0
        # Picking a scaling factor such that tanh(62/scaling) is nearly 1.
        # For example, tanh(62/23.4) ~ 0.99.
        scaling = 23.4
        normalized_score = math.tanh(eval_score / scaling)
        # Clamp to the interval [-1, 1]
        normalized_score = max(min(normalized_score, 1.0), -1.0)
    
        for x in range(bar_width):
            ratio = x / float(bar_width)
            r = int(255 * ratio)
            g = int(255 * ratio)
            b = int(255 * ratio)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.eval_bar_canvas.create_line(x, 0, x, bar_height, fill=color)
    
        marker_x = int((normalized_score + 1) / 2 * bar_width)
        accent_color = self.COLORS.get('accent', '#e94560')
        marker_width = 1  # Reduced marker outline thickness
        self.eval_bar_canvas.create_rectangle(marker_x - marker_width, 0,
                                              marker_x + marker_width, bar_height,
                                              fill=accent_color, outline=accent_color)
        mid_x = (bar_width // 2)
        self.eval_bar_canvas.create_line(mid_x, 0, mid_x, bar_height, fill="#666666", width=1)
    
        if abs(eval_score) < 0.2:
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
        # Reduced padding makes the label box slimmer
        style.configure('Status.TLabel',
                        background=COLORS['bg_light'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 16, 'bold'),
                        padding=(11, 4),   # Reduced from (18, 10)
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

        # Highlight kings if in check or checkmated
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece and isinstance(piece, King):
                    if is_in_check(self.board, piece.color):
                        highlight_color = "darkred" if not has_legal_moves(self.board, piece.color) else "red"
                        x1, y1 = self.board_to_canvas(r, c)
                        x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
                        self.canvas.create_rectangle(x1, y1, x2, y2, outline=highlight_color, width=3)

        # Draw pieces (existing code)
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
                                                text=symbol, font=("Arial", 39),
                                                fill=shadow_color, tags="piece")
                        self.canvas.create_text(x_center, y_center,
                                                text=symbol, font=("Arial Unicode MS", 39),
                                                fill="white", tags="piece")
                    else:
                        self.canvas.create_text(x_center, y_center, text=symbol,
                                                font=("Arial", 39), fill="black", tags="piece")

        # Draw dragging piece if any
        if self.dragging and self.drag_piece and self.drag_start is not None:
            piece = self.board[self.drag_start[0]][self.drag_start[1]]
            if piece is not None:
                self.canvas.create_text(self.drag_piece[0], self.drag_piece[1],
                                        text=piece.symbol(), font=("Arial", 36), tags="drag")
        
        # Draw an even border around the chess board
        board_width = COLS * SQUARE_SIZE
        board_height = ROWS * SQUARE_SIZE
        self.canvas.create_rectangle(0, 0, board_width, board_height, outline=self.COLORS['accent'], width=4)

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
            if validate_move(self.board, self.turn, self.drag_start, end_pos):
                moving_piece = self.board[self.drag_start[0]][self.drag_start[1]]
                self.board = moving_piece.move(self.board, self.drag_start, end_pos)
                check_evaporation(self.board)
                
                # Redraw board and force an update so the capture animation completes
                self.draw_board()
                self.master.update_idletasks()  # Ensure canvas refresh

                # Check for game over conditions
                result, winner = check_game_over(self.board)
                if result == "checkmate":
                    self.game_over = True
                    self.turn_label.config(text=f"Checkmate! {winner.capitalize()} wins!")
                elif result == "stalemate":
                    self.game_over = True
                    self.turn_label.config(text="Stalemate! It's a draw.")
                else:
                    # Switch turns and record position history
                    self.turn = "black" if self.turn == "white" else "white"
                    self.position_history.append(self.get_position_key())
                    
                    # Use a minimal delay (20ms) to let board animation finalize if instant move is on
                    if self.game_mode.get() == "bot" and self.turn != self.human_color:
                        delay = 20 if self.instant_move.get() else 500
                        self.master.after(delay, self.make_bot_move)
            else:
                print("Illegal move!")
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

            # Check for game over conditions
            result, winner = check_game_over(self.board)
            if result == "checkmate":
                self.game_over = True
                self.turn_label.config(text=f"Checkmate! {winner.capitalize()} wins!")
            elif result == "stalemate":
                self.game_over = True
                self.turn_label.config(text="Stalemate! It's a draw.")
            else:
                # Switch turns back to the human player
                self.turn = self.human_color
                # Update position history after switching turns
                self.position_history.append(self.get_position_key())

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