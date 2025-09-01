# v27 (Patch for rook movements)
import copy

# -----------------------------
# Global Constants
# -----------------------------
ROWS, COLS = 8, 8
SQUARE_SIZE = 75
BOARD_COLOR_1 = "#D2B48C"
BOARD_COLOR_2 = "#8B5A2B"

DIRECTIONS = {
    'king': ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'queen': ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'rook': ((0, 1), (0, -1), (1, 0), (-1, 0)),
    'bishop': ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    'knight': ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
}
ADJACENT_DIRS = DIRECTIONS['king']

# --- Pre-computation Maps for Performance ---
KNIGHT_ATTACKS_FROM = { (r, c): {(r+dr, c+dc) for dr, dc in DIRECTIONS['knight'] if 0 <= r+dr < ROWS and 0 <= c+dc < COLS} for r in range(ROWS) for c in range(COLS) }
ADJACENT_SQUARES_MAP = { (r, c): {(r+dr, c+dc) for dr, dc in ADJACENT_DIRS if 0 <= r+dr < ROWS and 0 <= c+dc < COLS} for r in range(ROWS) for c in range(COLS) }

# ---------------------------------------------------
# PIECE CLASSES: THE SINGLE SOURCE OF TRUTH
# ---------------------------------------------------
class Piece:
    def __init__(self, color):
        self.color = color
        self.opponent_color = "black" if color == "white" else "white"
        self.pos = None

    def clone(self):
        new_piece = self.__class__(self.color)
        new_piece.pos = self.pos
        return new_piece

    def symbol(self): return "?"

    def get_valid_moves(self, board, pos):
        """Returns a list of squares the piece can legally move to."""
        return []

    def get_threats(self, board, pos):
        """Returns a set of squares the piece threatens."""
        # By default, a piece threatens the squares it can move to.
        return set(self.get_valid_moves(board, pos))

class King(Piece):
    def symbol(self): return "♔" if self.color == "white" else "♚"
    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['king']:
            r1, c1 = r_start + dr, c_start + dc
            if 0 <= r1 < ROWS and 0 <= c1 < COLS and (board.grid[r1][c1] is None or board.grid[r1][c1].color == self.opponent_color):
                moves.append((r1, c1))
                # Can move two squares if the first square is empty
                if board.grid[r1][c1] is None:
                    r2, c2 = r1 + dr, c1 + dc
                    if 0 <= r2 < ROWS and 0 <= c2 < COLS and (board.grid[r2][c2] is None or board.grid[r2][c2].color == self.opponent_color):
                        moves.append((r2, c2))
        return moves

class Queen(Piece):
    def symbol(self): return "♕" if self.color == "white" else "♛"
    def get_valid_moves(self, board, pos):
        moves = []
        grid = board.grid # Cache the grid lookup
        for dr, dc in DIRECTIONS['queen']:
            r, c = pos[0] + dr, pos[1] + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = grid[r][c] # Access the faster local variable
                if target is None:
                    moves.append((r, c))
                else:
                    if target.color != self.color: moves.append((r, c))
                    break
                r += dr; c += dc
        return moves

    def get_threats(self, board, pos):
        """
        A Queen threatens squares she can move to. If a move is a capture,
        she ALSO threatens the adjacent squares (AoE).
        """
        threats = set()
        valid_moves = self.get_valid_moves(board, pos)
        for move in valid_moves:
            # She always threatens the square she can move to.
            threats.add(move)
            
            # The AoE threat is ONLY projected around a potential capture.
            target_piece = board.grid[move[0]][move[1]]
            if target_piece is not None and target_piece.color == self.opponent_color:
                threats.update(ADJACENT_SQUARES_MAP.get(move, set()))
        return threats

class Rook(Piece):
    def symbol(self): return "♖" if self.color == "white" else "♜"
    
    def get_valid_moves(self, board, pos):
        """
        A Jungle Chess Rook can move through enemy pieces, but is blocked
        by friendly pieces. This is the corrected move generation.
        """
        moves = []
        grid = board.grid
        for dr, dc in DIRECTIONS['rook']:
            r, c = pos[0] + dr, pos[1] + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = grid[r][c]
                if target and target.color == self.color:
                    break # Stop at friendly pieces
                
                moves.append((r, c))
                
                # If we hit an enemy piece, we add the move but continue searching,
                # as the Rook can move through them.
                
                r += dr; c += dc
        return moves
        
    def get_threats(self, board, pos):
        """
        A Rook's threat PIERCES through enemy pieces, but is BLOCKED by friendly pieces.
        This corrected version is board-aware and stops BEFORE a friendly piece.
        """
        threats = set()
        grid = board.grid
        for dr, dc in DIRECTIONS['rook']:
            r, c = pos[0] + dr, pos[1] + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = grid[r][c]
                # CRITICAL FIX: Check for a friendly blocker FIRST.
                if target and target.color == self.color:
                    break # The threat ends here, before this square.
                
                threats.add((r, c))
                
                # Unlike get_valid_moves, the threat ray continues even after hitting
                # an enemy piece, but it is still blocked by a friendly one.
                r += dr; c += dc
        return threats
        

class Bishop(Piece):
    def symbol(self): return "♗" if self.color == "white" else "♝"
    def get_valid_moves(self, board, pos):
        moves = set()
        r_start, c_start = pos
        # 1. Standard diagonal moves
        for dr, dc in DIRECTIONS['bishop']:
            r, c = r_start + dr, c_start + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = board.grid[r][c]
                if target:
                    if target.color != self.color: moves.add((r, c))
                    break
                moves.add((r, c)); r += dr; c += dc
        # 2. Zig-zag moves
        direction_pairs = (((-1, 1), (-1, -1)), ((-1, -1), (-1, 1)), ((1, 1), (1, -1)), ((1, -1), (1, 1)), ((-1, 1), (1, 1)), ((1, 1), (-1, 1)), ((-1, -1), (1, -1)), ((1, -1), (-1, -1)))
        for d1, d2 in direction_pairs:
            cr, cc, cd = r_start, c_start, d1
            while True:
                cr += cd[0]; cc += cd[1]
                if not (0 <= cr < ROWS and 0 <= cc < COLS): break
                target = board.grid[cr][cc]
                if target:
                    if target.color != self.color: moves.add((cr, cc))
                    break
                moves.add((cr, cc)); cd = d2 if cd == d1 else d1
        return list(moves)

class Knight(Piece):
    def symbol(self): return "♘" if self.color == "white" else "♞"
    def get_valid_moves(self, board, pos):
        moves = []
        for dr, dc in DIRECTIONS['knight']:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and (board.grid[nr][nc] is None or board.grid[nr][nc].color != self.color):
                moves.append((nr,nc))
        return moves

    def get_threats(self, board, pos):
        # A Knight threatens its landing squares AND the AoE squares around them.
        threats = set()
        valid_moves = self.get_valid_moves(board, pos)
        for move in valid_moves:
            threats.add(move)
            threats.update(KNIGHT_ATTACKS_FROM.get(move, set()))
        return threats

class Pawn(Piece):
    def __init__(self, color):
        super().__init__(color)
        self.direction = -1 if self.color == "white" else 1
        self.starting_row = 6 if self.color == "white" else 1

    def symbol(self): return "♙" if self.color == "white" else "♟"

    def get_valid_moves(self, board, pos):
        moves = []
        r, c = pos
        # 1. Forward move (which can be a capture)
        one_r = r + self.direction
        if 0 <= one_r < ROWS and (board.grid[one_r][c] is None or board.grid[one_r][c].color == self.opponent_color):
            moves.append((one_r, c))
        # 2. Initial two-step move (also a capture if it lands on a piece)
        if r == self.starting_row and board.grid[one_r][c] is None:
            two_r = r + (2 * self.direction)
            if 0 <= two_r < ROWS and (board.grid[two_r][c] is None or board.grid[two_r][c].color == self.opponent_color):
                moves.append((two_r, c))
        # 3. Sideways captures
        for dc_offset in [-1, 1]:
            new_c = c + dc_offset
            if 0 <= new_c < COLS and board.grid[r][new_c] is not None and board.grid[r][new_c].color == self.opponent_color:
                moves.append((r, new_c))
        return moves
        
    def get_threats(self, board, pos):
        """
        A pawn's threats are only the squares it can legally capture on.
        This corrected version is board-aware to prevent friendly fire.
        """
        threats = set()
        r, c = pos
        
        # Forward capture threat
        one_r = r + self.direction
        if 0 <= one_r < ROWS and board.grid[one_r][c] is not None and board.grid[one_r][c].color == self.opponent_color:
            threats.add((one_r, c))
            
        # Sideways capture threats
        for dc_offset in [-1, 1]:
            new_c = c + dc_offset
            if 0 <= new_c < COLS and board.grid[r][new_c] is not None and board.grid[r][new_c].color == self.opponent_color:
                threats.add((r, new_c))

        return threats

# ---------------------------------------------
# Board Class
# ---------------------------------------------
class Board:
    def __init__(self, setup=True):
        self.grid = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.white_king_pos = None
        self.black_king_pos = None
        self.white_pieces = []
        self.black_pieces = []
        if setup: self._setup_initial_board()

    def _setup_initial_board(self):
        pieces = {0: [(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King), (5, Bishop), (6, Knight), (7, Rook)], 1: [(i, Pawn) for i in range(8)], 6: [(i, Pawn) for i in range(8)], 7: [(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King), (5, Bishop), (6, Knight), (7, Rook)]}
        for r, piece_list in pieces.items():
            color = "black" if r < 2 else "white"
            for c, piece_class in piece_list: self.add_piece(piece_class(color), r, c)

    def add_piece(self, piece, r, c):
        if self.grid[r][c] is not None: self.remove_piece(r, c)
        self.grid[r][c] = piece
        piece.pos = (r, c)
        if piece.color == 'white': self.white_pieces.append(piece)
        else: self.black_pieces.append(piece)
        if isinstance(piece, King):
            if piece.color == "white": self.white_king_pos = (r, c)
            else: self.black_king_pos = (r, c)
            
    def remove_piece(self, r, c):
        piece = self.grid[r][c]
        if not piece: return
        
        if piece.color == 'white':
            self.white_pieces.remove(piece)
        else:
            self.black_pieces.remove(piece)
            
        if isinstance(piece, King):
            if piece.color == "white":
                self.white_king_pos = None
            else:
                self.black_king_pos = None
                
        # CRITICAL FIX: Decommission the piece by nullifying its position.
        piece.pos = None 
        
        self.grid[r][c] = None

    def move_piece(self, start, end):
        piece = self.grid[start[0]][start[1]]
        if not piece: return
        piece.pos = end
        if isinstance(piece, King):
            if piece.color == "white": self.white_king_pos = end
            else: self.black_king_pos = end
        self.grid[start[0]][start[1]] = None
        self.grid[end[0]][end[1]] = piece

    def find_king_pos(self, color): return self.white_king_pos if color == 'white' else self.black_king_pos

    def clone(self):
        new_board = Board(setup=False)
        for piece in self.white_pieces + self.black_pieces:
            p_clone = piece.clone()
            new_board.add_piece(p_clone, p_clone.pos[0], p_clone.pos[1])
        return new_board

    def make_move(self, start, end):
        moving_piece = self.grid[start[0]][start[1]]
        if not moving_piece: return
        target_piece = self.grid[end[0]][end[1]]
        is_capture = target_piece is not None
        
        if isinstance(moving_piece, Rook): self._apply_rook_piercing(start, end, moving_piece.color)
        if is_capture: self.remove_piece(end[0], end[1])
        self.move_piece(start, end)
        
        if isinstance(moving_piece, Queen) and is_capture:
            self._apply_queen_aoe(end, moving_piece.color)
        elif isinstance(moving_piece, Knight):
            self._apply_knight_aoe(end)
        elif isinstance(moving_piece, Pawn):
            promotion_rank = 0 if moving_piece.color == "white" else (ROWS - 1)
            if end[0] == promotion_rank:
                self.remove_piece(end[0], end[1])
                self.add_piece(Queen(moving_piece.color), end[0], end[1])
        
        self._check_all_knight_evaporation()

    def _apply_queen_aoe(self, pos, queen_color):
        if self.grid[pos[0]][pos[1]]: self.remove_piece(pos[0], pos[1]) # Remove the queen herself
        for r, c in ADJACENT_SQUARES_MAP.get(pos, set()):
            adj_piece = self.grid[r][c]
            if adj_piece and adj_piece.color != queen_color: self.remove_piece(r, c)

    def _apply_rook_piercing(self, start, end, rook_color):
        dr = 0 if start[0] == end[0] else 1 if end[0] > start[0] else -1
        dc = 0 if start[1] == end[1] else 1 if end[1] > start[1] else -1
        cr, cc = start[0] + dr, start[1] + dc
        while (cr, cc) != end:
            target = self.grid[cr][cc]
            if target and target.color != rook_color: self.remove_piece(cr, cc)
            cr += dr; cc += dc

    def _apply_knight_aoe(self, knight_pos):
        knight_instance = self.grid[knight_pos[0]][knight_pos[1]]
        if not knight_instance: return
        to_remove, enemy_knights_destroyed = [], False
        for r, c in KNIGHT_ATTACKS_FROM.get(knight_pos, set()):
            target = self.grid[r][c]
            if target and target.color != knight_instance.color:
                to_remove.append((r, c))
                if isinstance(target, Knight): enemy_knights_destroyed = True
        for r,c in to_remove: self.remove_piece(r,c)
        if enemy_knights_destroyed: self.remove_piece(knight_pos[0], knight_pos[1])

    def _check_all_knight_evaporation(self):
        all_knights = [p for p in (self.white_pieces + self.black_pieces) if isinstance(p, Knight)]
        for knight in all_knights:
            if self.grid[knight.pos[0]][knight.pos[1]] is knight:
                self._apply_knight_aoe(knight.pos)

# ----------------------------------------------------
# GLOBAL GAME LOGIC: ROBUST & CENTRALIZED
# ----------------------------------------------------
def is_square_attacked(board, r, c, attacking_color):
    """Checks if a square is attacked by relying on each piece's get_threats method."""
    attacking_pieces = board.white_pieces if attacking_color == 'white' else board.black_pieces
    for piece in attacking_pieces:
        if (r, c) in piece.get_threats(board, piece.pos):
            return True
    return False

def is_in_check(board, color):
    """Determines if a player is in check."""
    king_pos = board.find_king_pos(color)
    if not king_pos: return True
    opponent_color = "black" if color == "white" else "white"
    return is_square_attacked(board, king_pos[0], king_pos[1], opponent_color)

def generate_legal_moves_generator(board, color, yield_boards=False):
    """A generator that yields all legal moves for a given color."""
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    opponent_color = "black" if color == "white" else "white"
    for piece in piece_list:
        start_pos = piece.pos
        for end_pos in piece.get_valid_moves(board, start_pos):
            sim_board = board.clone()
            sim_board.make_move(start_pos, end_pos)
            king_pos = sim_board.find_king_pos(color)
            if king_pos and not is_square_attacked(sim_board, king_pos[0], king_pos[1], opponent_color):
                if yield_boards:
                    yield (start_pos, end_pos), sim_board
                else:
                    yield (start_pos, end_pos)

def get_all_legal_moves(board, color):
    """Returns a list of all legal moves for a given color."""
    return list(generate_legal_moves_generator(board, color))

def get_all_pseudo_legal_moves(board, color):
    """Returns all moves a piece can make, without checking for self-check."""
    moves = []
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    for piece in piece_list:
        moves.extend([(piece.pos, end_pos) for end_pos in piece.get_valid_moves(board, piece.pos)])
    return moves

def has_legal_moves(board, color):
    """Efficiently checks if any legal moves exist."""
    try:
        next(generate_legal_moves_generator(board, color))
        return True
    except StopIteration:
        return False
        
def is_insufficient_material(board):
    """Checks for endgames that are automatic draws."""
    total_pieces = len(board.white_pieces) + len(board.black_pieces)
    if total_pieces > 3: return False
    if total_pieces == 2: return True
    if total_pieces == 3:
        major_side = board.white_pieces if len(board.white_pieces) == 2 else board.black_pieces
        piece_types = {type(p) for p in major_side}
        if King in piece_types and (Rook in piece_types or Bishop in piece_types or Knight in piece_types):
            return True
    return False

def get_game_state(board, turn_to_move, position_counts, ply_count, max_moves):
    """Determines the current state of the game (ongoing, checkmate, stalemate, etc.)."""
    if not has_legal_moves(board, turn_to_move):
        winner = 'black' if turn_to_move == 'white' else 'white'
        return ("checkmate", winner) if is_in_check(board, turn_to_move) else ("stalemate", None)
    if is_insufficient_material(board):
        return "insufficient_material", None
    try:
        from AI import board_hash
        current_hash = board_hash(board, turn_to_move)
        if position_counts.get(current_hash, 0) >= 3:
            return "repetition", None
    except ImportError: # Failsafe for when running without the AI file
        pass
    if ply_count >= max_moves:
        return "move_limit", None
    return "ongoing", None

def calculate_material_swing(board, move, value_func):
    """Calculates the change in material after a move, accounting for all AoE effects."""
    moving_piece = board.grid[move[0][0]][move[0][1]]
    if not moving_piece: return 0
    sim_board = board.clone()
    original_opponent_pieces = {p.pos for p in (sim_board.black_pieces if moving_piece.color == 'white' else sim_board.white_pieces)}
    original_friendly_pieces = {p.pos for p in (sim_board.white_pieces if moving_piece.color == 'white' else sim_board.black_pieces)}
    sim_board.make_move(move[0], move[1])
    final_opponent_pieces = {p.pos for p in (sim_board.black_pieces if moving_piece.color == 'white' else sim_board.white_pieces)}
    final_friendly_pieces = {p.pos for p in (sim_board.white_pieces if moving_piece.color == 'white' else sim_board.black_pieces)}
    
    swing = 0
    # Re-create piece objects from positions to get their value
    original_board_pieces = {p.pos: p for p in board.white_pieces + board.black_pieces}
    for pos in original_opponent_pieces - final_opponent_pieces:
        swing += value_func(original_board_pieces[pos], sim_board)
    for pos in original_friendly_pieces - final_friendly_pieces:
        swing -= value_func(original_board_pieces[pos], sim_board)
    return swing

def is_draw(board, turn_to_move, position_counts, ply_count, max_moves):
    """Helper function to check if the current position is a draw."""
    state, _ = get_game_state(board, turn_to_move, position_counts, ply_count, max_moves)
    return state in ["stalemate", "insufficient_material", "repetition", "move_limit"]

def is_rook_piercing_capture(board, move):
    """
    Checks if a Rook move will capture any pieces along its path,
    even if it lands on an empty square. This is a fast, clone-free check.
    """
    start, end = move
    moving_piece = board.grid[start[0]][start[1]]

    # This special check only applies to Rooks.
    if not isinstance(moving_piece, Rook):
        return False

    # A direct capture is already handled, this is for piercing to an empty square.
    if board.grid[end[0]][end[1]] is not None:
        return False

    dr = 0 if start[0] == end[0] else 1 if end[0] > start[0] else -1
    dc = 0 if start[1] == end[1] else 1 if end[1] > start[1] else -1
    cr, cc = start[0] + dr, start[1] + dc

    while (cr, cc) != end:
        target = board.grid[cr][cc]
        if target and target.color != moving_piece.color:
            return True # Found an enemy piece in the path; it's a piercing capture.
        cr += dr
        cc += dc
        
    return False

def generate_all_captures(board, color):
    """
    An optimized generator that yields only pseudo-legal captures and promotions.
    This is much faster than generating all moves and then filtering.
    """
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    opponent_color = "black" if color == "white" else "white"

    for piece in piece_list:
        start_pos = piece.pos
        # Promotions are always tactical, so we can check the piece type first
        if isinstance(piece, Pawn):
            promotion_rank = 0 if piece.color == "white" else (ROWS - 1)
            for end_pos in piece.get_valid_moves(board, start_pos):
                if board.grid[end_pos[0]][end_pos[1]] is not None or end_pos[0] == promotion_rank:
                    yield (start_pos, end_pos)
        # Rooks are special because a piercing move to an empty square is a capture
        elif isinstance(piece, Rook):
             for end_pos in piece.get_valid_moves(board, start_pos):
                if board.grid[end_pos[0]][end_pos[1]] is not None or is_rook_piercing_capture(board, (start_pos, end_pos)):
                    yield (start_pos, end_pos)
        # For other pieces, a capture is only when they land on an enemy piece
        else:
            for end_pos in piece.get_valid_moves(board, start_pos):
                if board.grid[end_pos[0]][end_pos[1]] is not None:
                    yield (start_pos, end_pos)

def is_quiet_knight_evaporation(board, move):
    """
    A fast, clone-free check to see if a quiet Knight move will evaporate
    any enemy pieces from its destination square.
    """
    start_pos, end_pos = move
    moving_piece = board.grid[start_pos[0]][start_pos[1]]
    
    # This only applies to Knights making a non-capturing move.
    if not isinstance(moving_piece, Knight) or board.grid[end_pos[0]][end_pos[1]] is not None:
        return False
        
    # Check the future evaporation zone from the destination square.
    for r, c in KNIGHT_ATTACKS_FROM.get(end_pos, set()):
        target = board.grid[r][c]
        if target and target.color == moving_piece.opponent_color:
            return True # An enemy piece will be evaporated.
            
    return False