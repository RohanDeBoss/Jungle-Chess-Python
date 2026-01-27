# v35 Optimised the passive Knight Evaporation logic in Board._apply_knight_aoe (and its call in make_move)

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
# Change the inner {} to [] to preserve order
KNIGHT_ATTACKS_FROM = { (r, c): [(r+dr, c+dc) for dr, dc in DIRECTIONS['knight'] if 0 <= r+dr < ROWS and 0 <= c+dc < COLS] for r in range(ROWS) for c in range(COLS) }
ADJACENT_SQUARES_MAP = { (r, c): [(r+dr, c+dc) for dr, dc in ADJACENT_DIRS if 0 <= r+dr < ROWS and 0 <= c+dc < COLS] for r in range(ROWS) for c in range(COLS) }

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
        return []

    def get_threats(self, board, pos):
        return self.get_valid_moves(board, pos)

class King(Piece):
    def symbol(self): return "♔" if self.color == "white" else "♚"
    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['king']:
            r1, c1 = r_start + dr, c_start + dc
            if 0 <= r1 < ROWS and 0 <= c1 < COLS and (board.grid[r1][c1] is None or board.grid[r1][c1].color == self.opponent_color):
                moves.append((r1, c1))
                if board.grid[r1][c1] is None:
                    r2, c2 = r1 + dr, c1 + dc
                    if 0 <= r2 < ROWS and 0 <= c2 < COLS and (board.grid[r2][c2] is None or board.grid[r2][c2].color == self.opponent_color):
                        moves.append((r2, c2))
        return moves

class Queen(Piece):
    def symbol(self): return "♕" if self.color == "white" else "♛"
    def get_valid_moves(self, board, pos):
        moves = []
        grid = board.grid
        for dr, dc in DIRECTIONS['queen']:
            r, c = pos[0] + dr, pos[1] + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = grid[r][c]
                if target is None:
                    moves.append((r, c))
                else:
                    if target.color != self.color: moves.append((r, c))
                    break
                r += dr; c += dc
        return moves

    def get_threats(self, board, pos):
        threats = set()
        valid_moves = self.get_valid_moves(board, pos)
        for move in valid_moves:
            threats.add(move)
            target_piece = board.grid[move[0]][move[1]]
            if target_piece is not None and target_piece.color == self.opponent_color:
                threats.update(ADJACENT_SQUARES_MAP.get(move, set()))
        return threats

class Rook(Piece):
    def symbol(self): return "♖" if self.color == "white" else "♜"
    def get_valid_moves(self, board, pos):
        moves = []
        grid = board.grid
        for dr, dc in DIRECTIONS['rook']:
            r, c = pos[0] + dr, pos[1] + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = grid[r][c]
                if target and target.color == self.color:
                    break
                moves.append((r, c))
                r += dr; c += dc
        return moves
        
    def get_threats(self, board, pos):
        threats = set()
        grid = board.grid
        for dr, dc in DIRECTIONS['rook']:
            r, c = pos[0] + dr, pos[1] + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = grid[r][c]
                if target and target.color == self.color:
                    break
                threats.add((r, c))
                r += dr; c += dc
        return threats

class Bishop(Piece):
    def symbol(self): return "♗" if self.color == "white" else "♝"
    def get_valid_moves(self, board, pos):
        moves = {} # Dict keys preserve order and ensure uniqueness
        r_start, c_start = pos
        
        # 1. Normal Diagonal
        for dr, dc in DIRECTIONS['bishop']:
            r, c = r_start + dr, c_start + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = board.grid[r][c]
                if target:
                    if target.color != self.color: moves[(r, c)] = None
                    break
                moves[(r, c)] = None; r += dr; c += dc
                
        # 2. Zig-Zag Movement
        # (Moved this tuple out of the loop for tiny speedup too)
        direction_pairs = (((-1, 1), (-1, -1)), ((-1, -1), (-1, 1)), ((1, 1), (1, -1)), ((1, -1), (1, 1)), ((-1, 1), (1, 1)), ((1, 1), (-1, 1)), ((-1, -1), (1, -1)), ((1, -1), (-1, -1)))
        for d1, d2 in direction_pairs:
            cr, cc, cd = r_start, c_start, d1
            while True:
                cr += cd[0]; cc += cd[1]
                if not (0 <= cr < ROWS and 0 <= cc < COLS): break
                target = board.grid[cr][cc]
                if target:
                    if target.color != self.color: moves[(cr, cc)] = None
                    break
                moves[(cr, cc)] = None; cd = d2 if cd == d1 else d1
                
        return list(moves)
    
class Knight(Piece):
    def symbol(self): return "♘" if self.color == "white" else "♞"
    def get_valid_moves(self, board, pos):
        return [(r, c) for r, c in KNIGHT_ATTACKS_FROM[pos] 
                if (p := board.grid[r][c]) is None or p.color != self.color]

    def get_threats(self, board, pos):
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
        direction = self.direction
        grid = board.grid
        
        # 1. Forward Moves
        one_r = r + direction
        if 0 <= one_r < ROWS:
            target1 = grid[one_r][c]
            # Check 1st square (Move or Capture)
            if target1 is None or target1.color == self.opponent_color:
                moves.append((one_r, c))
                
                # 2. Forward 2 (Only if 1st was empty, not a capture)
                if r == self.starting_row and target1 is None:
                    two_r = r + (2 * direction)
                    target2 = grid[two_r][c]
                    if target2 is None or target2.color == self.opponent_color:
                        moves.append((two_r, c))

        # 3. Sideways Captures (Unrolled for speed)
        if c > 0:
            target = grid[r][c-1]
            if target and target.color == self.opponent_color:
                moves.append((r, c-1))
        if c < COLS - 1:
            target = grid[r][c+1]
            if target and target.color == self.opponent_color:
                moves.append((r, c+1))
                
        return moves
        

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
        
        try:
            if piece.color == 'white': self.white_pieces.remove(piece)
            else: self.black_pieces.remove(piece)
        except ValueError:
            pass
            
        if isinstance(piece, King):
            if piece.color == "white": self.white_king_pos = None
            else: self.black_king_pos = None
                
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
        
        # Clone lists directly to avoid add_piece() function call overhead
        new_board.white_pieces = [p.clone() for p in self.white_pieces]
        new_board.black_pieces = [p.clone() for p in self.black_pieces]
        
        # Populate grid and king positions directly
        for p in new_board.white_pieces:
            new_board.grid[p.pos[0]][p.pos[1]] = p
            if isinstance(p, King): new_board.white_king_pos = p.pos
            
        for p in new_board.black_pieces:
            new_board.grid[p.pos[0]][p.pos[1]] = p
            if isinstance(p, King): new_board.black_king_pos = p.pos
            
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
        elif isinstance(moving_piece, Pawn):
            promotion_rank = 0 if moving_piece.color == "white" else (ROWS - 1)
            if end[0] == promotion_rank:
                self.remove_piece(end[0], end[1])
                self.add_piece(Queen(moving_piece.color), end[0], end[1])

        self._apply_knight_aoe(end, is_active_move=isinstance(moving_piece, Knight))

    def _apply_queen_aoe(self, pos, queen_color):
        if self.grid[pos[0]][pos[1]]: self.remove_piece(pos[0], pos[1])
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

    def _apply_knight_aoe(self, pos, is_active_move):
        grid = self.grid
        
        # 1. Active Evaporation (The Knight itself moved)
        if is_active_move:
            knight_instance = grid[pos[0]][pos[1]]
            if not knight_instance: return
            
            # Use the precomputed list (optimized in v30)
            to_remove, enemy_knights_destroyed = [], False
            for r, c in KNIGHT_ATTACKS_FROM[pos]:
                target = grid[r][c]
                if target and target.color != knight_instance.color:
                    to_remove.append(target)
                    if isinstance(target, Knight):
                        enemy_knights_destroyed = True
            
            for piece in to_remove:
                self.remove_piece(piece.pos[0], piece.pos[1])

            if enemy_knights_destroyed:
                self.remove_piece(pos[0], pos[1])
                
        # 2. Passive Evaporation (Another piece moved into a Knight's range)
        else:
            # We only need to check if the specific square 'pos' is targeted by an enemy Knight.
            # KNIGHT_ATTACKS_FROM is symmetric: if Knight at A hits B, a Knight at B hits A.
            victim = grid[pos[0]][pos[1]]
            if not victim: return
            
            for r, c in KNIGHT_ATTACKS_FROM[pos]:
                potential_killer = grid[r][c]
                # Check if there is a Knight here, and if it is an enemy
                if (potential_killer and 
                    isinstance(potential_killer, Knight) and 
                    potential_killer.color != victim.color):
                    
                    self.remove_piece(pos[0], pos[1])
                    return # Piece is gone, stop checking

    def _get_rook_piercing_captures(self, start, end, rook_color):
        captured = []
        dr = 0 if start[0] == end[0] else 1 if end[0] > start[0] else -1
        dc = 0 if start[1] == end[1] else 1 if end[1] > start[1] else -1
        cr, cc = start[0] + dr, start[1] + dc
        while (cr, cc) != end:
            target = self.grid[cr][cc]
            if target and target.color != rook_color:
                captured.append(target)
            cr += dr; cc += dc
        return captured

    def _get_queen_aoe_captures(self, pos, queen_color):
        captured = []
        for r, c in ADJACENT_SQUARES_MAP.get(pos, set()):
            adj_piece = self.grid[r][c]
            if adj_piece and adj_piece.color != queen_color:
                captured.append(adj_piece)
        return captured
    
    def _get_knight_aoe_outcome(self, knight_pos, knight_color):
        captured, self_evaporates = [], False
        for r, c in KNIGHT_ATTACKS_FROM.get(knight_pos, set()):
            target = self.grid[r][c]
            if target and target.color != knight_color:
                captured.append(target)
                if isinstance(target, Knight):
                    self_evaporates = True
        return captured, self_evaporates

    def get_move_outcome(self, move):
        start_pos, end_pos = move
        moving_piece = self.grid[start_pos[0]][start_pos[1]]
        if not moving_piece:
            return set(), set(), None
            
        friendly_lost, opponent_captured, promotion_type = set(), set(), None
        target_piece = self.grid[end_pos[0]][end_pos[1]]
        is_capture = target_piece is not None

        if is_capture:
            opponent_captured.add(target_piece)

        if isinstance(moving_piece, Rook):
            opponent_captured.update(self._get_rook_piercing_captures(start_pos, end_pos, moving_piece.color))
        elif isinstance(moving_piece, Queen) and is_capture:
            friendly_lost.add(moving_piece)
            opponent_captured.update(self._get_queen_aoe_captures(end_pos, moving_piece.color))
        elif isinstance(moving_piece, Knight):
            captures, self_evaporates = self._get_knight_aoe_outcome(end_pos, moving_piece.color)
            opponent_captured.update(captures)
            if self_evaporates: friendly_lost.add(moving_piece)
        elif isinstance(moving_piece, Pawn):
            promotion_rank = 0 if moving_piece.color == "white" else (ROWS - 1)
            if end_pos[0] == promotion_rank:
                promotion_type = Queen
                friendly_lost.add(moving_piece)
        
        return friendly_lost, opponent_captured, promotion_type

# ----------------------------------------------------
# GLOBAL GAME LOGIC: ROBUST & CENTRALIZED
# ----------------------------------------------------
def is_square_attacked(board, r, c, attacking_color):
    attacking_pieces = board.white_pieces if attacking_color == 'white' else board.black_pieces
    for piece in attacking_pieces:
        if piece.pos and (r, c) in piece.get_threats(board, piece.pos):
            return True
    return False

def is_in_check(board, color):
    king_pos = board.find_king_pos(color)
    if not king_pos: return True
    opponent_color = "black" if color == "white" else "white"
    return is_square_attacked(board, king_pos[0], king_pos[1], opponent_color)

def generate_legal_moves_generator(board, color, yield_boards=False):
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    opponent_color = "black" if color == "white" else "white"
    for piece in piece_list: 
        start_pos = piece.pos
        if start_pos is None: continue
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
    return list(generate_legal_moves_generator(board, color))

def get_all_pseudo_legal_moves(board, color):
    moves = []
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    for piece in piece_list:
        if piece.pos is not None:
            moves.extend([(piece.pos, end_pos) for end_pos in piece.get_valid_moves(board, piece.pos)])
    return moves

def has_legal_moves(board, color):
    try:
        next(generate_legal_moves_generator(board, color))
        return True
    except StopIteration:
        return False
        
# --- UPDATE: CORRECTED INSUFFICIENT MATERIAL CHECKS ---
def is_insufficient_material(board):
    """Checks for endgames that are automatic draws."""
    white_pieces = board.white_pieces
    black_pieces = board.black_pieces
    total_pieces = len(white_pieces) + len(black_pieces)

    if total_pieces > 4: return False # Fast exit for most games

    # If anyone has a Pawn or Queen, it's NOT a draw.
    for p in white_pieces:
        if isinstance(p, (Pawn, Queen)): return False
    for p in black_pieces:
        if isinstance(p, (Pawn, Queen)): return False

    # 2. K vs K
    if total_pieces == 2: return True

    # 3. K vs K + 1 piece
    # In this variant:
    # - Knight vs King is a WIN (False).
    # - Rook vs King is a DRAW (True).
    # - Bishop vs King is a DRAW (True).
    if total_pieces == 3:
        all_pieces = white_pieces + black_pieces
        for p in all_pieces:
            if isinstance(p, Knight): return False
        return True

    # 4. K+1 vs K+1
    if total_pieces == 4:
        w_has_rook = any(isinstance(p, Rook) for p in white_pieces)
        b_has_rook = any(isinstance(p, Rook) for p in black_pieces)
        w_has_bishop = any(isinstance(p, Bishop) for p in white_pieces)
        b_has_bishop = any(isinstance(p, Bishop) for p in black_pieces)
        w_has_knight = any(isinstance(p, Knight) for p in white_pieces)
        b_has_knight = any(isinstance(p, Knight) for p in black_pieces)
        
        # R vs R (Draw)
        if w_has_rook and b_has_rook: return True
        # R vs B (Draw)
        if (w_has_rook and b_has_bishop) or (w_has_bishop and b_has_rook): return True
        # B vs B (Draw)
        if w_has_bishop and b_has_bishop: return True
        # N vs N (Draw)
        if w_has_knight and b_has_knight: return True
        
    return False

def get_game_state(board, turn_to_move, position_counts, ply_count, max_moves):
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
    except ImportError:
        pass
    if ply_count >= max_moves:
        return "move_limit", None
    return "ongoing", None

def calculate_material_swing(board, move, tapered_vals_by_type):
    friendly_lost, opponent_captured, promotion_type = board.get_move_outcome(move)
    if not friendly_lost and not opponent_captured and promotion_type is None: return 0
    swing = 0
    for piece in opponent_captured:
        swing += tapered_vals_by_type.get(type(piece), 0)
    for piece in friendly_lost:
        swing -= tapered_vals_by_type.get(type(piece), 0)
    if promotion_type is not None:
        swing += tapered_vals_by_type.get(promotion_type, 0)
    return swing

def is_draw(board, turn_to_move, position_counts, ply_count, max_moves):
    state, _ = get_game_state(board, turn_to_move, position_counts, ply_count, max_moves)
    return state in ["stalemate", "insufficient_material", "repetition", "move_limit"]

def is_rook_piercing_capture(board, move):
    start, end = move
    moving_piece = board.grid[start[0]][start[1]]
    if not isinstance(moving_piece, Rook): return False
    if board.grid[end[0]][end[1]] is not None: return False
    dr = 0 if start[0] == end[0] else 1 if end[0] > start[0] else -1
    dc = 0 if start[1] == end[1] else 1 if end[1] > start[1] else -1
    cr, cc = start[0] + dr, start[1] + dc
    while (cr, cc) != end:
        target = board.grid[cr][cc]
        if target and target.color != moving_piece.color:
            return True
        cr += dr; cc += dc
    return False

def generate_all_captures(board, color):
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    for piece in piece_list:
        start_pos = piece.pos
        if start_pos is None: continue
        if isinstance(piece, Pawn):
            promotion_rank = 0 if piece.color == "white" else (ROWS - 1)
            for end_pos in piece.get_valid_moves(board, start_pos):
                if board.grid[end_pos[0]][end_pos[1]] is not None or end_pos[0] == promotion_rank:
                    yield (start_pos, end_pos)
        elif isinstance(piece, Rook):
             for end_pos in piece.get_valid_moves(board, start_pos):
                if board.grid[end_pos[0]][end_pos[1]] is not None or is_rook_piercing_capture(board, (start_pos, end_pos)):
                    yield (start_pos, end_pos)
        else:
            for end_pos in piece.get_valid_moves(board, start_pos):
                if board.grid[end_pos[0]][end_pos[1]] is not None:
                    yield (start_pos, end_pos)

def is_quiet_knight_evaporation(board, move):
    start_pos, end_pos = move
    moving_piece = board.grid[start_pos[0]][start_pos[1]]
    if not isinstance(moving_piece, Knight) or board.grid[end_pos[0]][end_pos[1]] is not None:
        return False
    for r, c in KNIGHT_ATTACKS_FROM.get(end_pos, set()):
        target = board.grid[r][c]
        if target and target.color == moving_piece.opponent_color:
            return True
    return False

def generate_all_tactical_moves(board, color):
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    
    for piece in piece_list:
        start_pos = piece.pos
        if start_pos is None: continue

        for end_pos in piece.get_valid_moves(board, start_pos):
            is_capture = board.grid[end_pos[0]][end_pos[1]] is not None
            is_promotion = isinstance(piece, Pawn) and (end_pos[0] == 0 or end_pos[0] == ROWS - 1)
            
            if is_capture or is_promotion:
                yield (start_pos, end_pos)
                continue
            
            if is_rook_piercing_capture(board, (start_pos, end_pos)):
                yield (start_pos, end_pos)
            elif is_quiet_knight_evaporation(board, (start_pos, end_pos)):
                yield (start_pos, end_pos)

def format_move(move):
    """Converts a move tuple to a human-readable algebraic string."""
    if not move: return "None"
    (r1, c1), (r2, c2) = move
    return f"{'abcdefgh'[c1]}{'87654321'[r1]}-{'abcdefgh'[c2]}{'87654321'[r2]}"