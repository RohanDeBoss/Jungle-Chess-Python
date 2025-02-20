# game_logic.py

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
# Game Logic
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
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece and piece.color == color:
                moves = piece.get_valid_moves(board, (r, c))
                for move in moves:
                    if validate_move(board, color, (r, c), move):
                        print(f"Legal move found for {color}: {piece.symbol()} from {(r, c)} to {move}")
                        return True
    print(f"No legal moves found for {color}")
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

    enemy_color = 'black' if color == 'white' else 'white'
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece and piece.color == enemy_color:
                # Check standard threats
                if king_pos in piece.get_valid_moves(board, (r, c)):
                    return True
                # Check explosion threats
                if isinstance(piece, Queen):
                    for move in piece.get_valid_moves(board, (r, c)):
                        if max(abs(move[0] - king_pos[0]), abs(move[1] - king_pos[1])) == 1:
                            target = board[move[0]][move[1]]
                            if target and target.color == color:
                                return True
                # Check evaporation threats
                if isinstance(piece, Knight):
                    valid_moves = piece.get_valid_moves(board, (r, c))
                    for move in valid_moves:
                        for dr, dc in DIRECTIONS['knight']:
                            kr, kc = king_pos
                            if (move[0] + dr) == kr and (move[1] + dc) == kc:
                                return True
    return False

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

    # Check if the move is in the piece's valid moves
    valid_moves = piece.get_valid_moves(board, start)
    if end not in valid_moves:
        return False

    # Simulate the move
    simulated = piece.move(simulated, start, end)
    check_evaporation(simulated)

    # Ensure the king still exists after the move (e.g., wasn't evaporated)
    king_exists = any(
        isinstance(simulated[r][c], King) and simulated[r][c].color == color 
        for r in range(ROWS) for c in range(COLS)
    )
    if not king_exists:
        return False

    # Check if the move leaves the king in check
    if is_in_check(simulated, color):
        return False

    # Check for immediate threats from special mechanics
    if is_in_explosion_threat(simulated, color):
        return False

    if is_king_in_knight_evaporation_danger(simulated, color):
        return False

    if isinstance(piece, King) and is_king_attacked_by_knight(simulated, color, end):
        return False

    # Allow the move, even if it leads to checkmate on the next turn
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
