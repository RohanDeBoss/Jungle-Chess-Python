# v28.4 (Final Audit & Bug Fixes)
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
        return []

    def get_threats(self, board, pos):
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
        moves = set()
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['bishop']:
            r, c = r_start + dr, c_start + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = board.grid[r][c]
                if target:
                    if target.color != self.color: moves.add((r, c))
                    break
                moves.add((r, c)); r += dr; c += dc
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
        one_r = r + self.direction
        if 0 <= one_r < ROWS and (board.grid[one_r][c] is None or board.grid[one_r][c].color == self.opponent_color):
            moves.append((one_r, c))
        if r == self.starting_row and board.grid[one_r][c] is None:
            two_r = r + (2 * self.direction)
            if 0 <= two_r < ROWS and (board.grid[two_r][c] is None or board.grid[two_r][c].color == self.opponent_color):
                moves.append((two_r, c))
        for dc_offset in [-1, 1]:
            new_c = c + dc_offset
            if 0 <= new_c < COLS and board.grid[r][new_c] is not None and board.grid[r][new_c].color == self.opponent_color:
                moves.append((r, new_c))
        return moves
        
    def get_threats(self, board, pos):
        threats = set()
        r, c = pos
        one_r = r + self.direction
        if 0 <= one_r < ROWS and board.grid[one_r][c] is not None and board.grid[one_r][c].color == self.opponent_color:
            threats.add((one_r, c))
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
        
        try:
            if piece.color == 'white': self.white_pieces.remove(piece)
            else: self.black_pieces.remove(piece)
        except ValueError:
            # This is safe. It just means the piece was already removed by another
            # effect in the same turn (e.g., Rook pierce + Knight AoE).
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
        elif isinstance(moving_piece, Pawn):
            promotion_rank = 0 if moving_piece.color == "white" else (ROWS - 1)
            if end[0] == promotion_rank:
                self.remove_piece(end[0], end[1])
                self.add_piece(Queen(moving_piece.color), end[0], end[1])

        # Your original, tested knight logic is preserved and centralized here.
        self._apply_knight_aoe(end if isinstance(moving_piece, Knight) else None)

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

    def _apply_knight_aoe(self, moved_knight_pos=None):
        """
        Handles all knight evaporation logic for a turn.
        If a knight moved, its AoE takes priority. Otherwise, all passive AoEs are resolved.
        """
        # Case 1: A knight just moved. Resolve only its AoE.
        if moved_knight_pos:
            knight_instance = self.grid[moved_knight_pos[0]][moved_knight_pos[1]]
            if not knight_instance: return
            
            to_remove, enemy_knights_destroyed = [], False
            for r, c in KNIGHT_ATTACKS_FROM.get(moved_knight_pos, set()):
                target = self.grid[r][c]
                if target and target.color != knight_instance.color:
                    to_remove.append(target)
                    if isinstance(target, Knight):
                        enemy_knights_destroyed = True
                        
            for piece in to_remove:
                if piece.pos: self.remove_piece(piece.pos[0], piece.pos[1])

            if enemy_knights_destroyed:
                self.remove_piece(moved_knight_pos[0], moved_knight_pos[1])
            return # IMPORTANT: Do not proceed to passive checks

        # Case 2: A non-knight moved. Resolve all passive AoEs simultaneously.
        knights_on_board = [p for p in (self.white_pieces + self.black_pieces) if isinstance(p, Knight)]
        if not knights_on_board: return

        evaporation_map, knights_to_be_removed = {}, set()

        for knight in knights_on_board:
            evaporation_map[knight] = set()
            for r_aoe, c_aoe in KNIGHT_ATTACKS_FROM.get(knight.pos, set()):
                target = self.grid[r_aoe][c_aoe]
                if target and target.color != knight.color:
                    evaporation_map[knight].add(target)

        for targets in evaporation_map.values():
            for target in targets:
                if isinstance(target, Knight):
                    knights_to_be_removed.add(target)

        pieces_to_remove = set()
        for targets in evaporation_map.values():
            pieces_to_remove.update(target for target in targets if not isinstance(target, Knight))

        for piece in pieces_to_remove:
            if piece.pos: self.remove_piece(piece.pos[0], piece.pos[1])
        for knight in knights_to_be_removed:
            if knight.pos: self.remove_piece(knight.pos[0], knight.pos[1])

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
    for piece in list(piece_list): # Iterate over a copy
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
        
# --- REVERTED: Insufficient Material Logic back to your tested version ---
def is_insufficient_material(board):
    """Checks for endgames that are automatic draws."""
    total_pieces = len(board.white_pieces) + len(board.black_pieces)
    if total_pieces > 3: return False
    if total_pieces == 2: return True
    if total_pieces == 3:
        major_side = board.white_pieces if len(board.white_pieces) == 2 else board.black_pieces
        piece_types = {type(p) for p in major_side}
        if King in piece_types and (Bishop in piece_types or Knight in piece_types):
            return True
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
    """
    An optimized generator that yields all pseudo-legal tactical moves.
    This includes captures, promotions, quiet rook skewers, and quiet knight evaporations.
    """
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    
    for piece in piece_list:
        start_pos = piece.pos
        if start_pos is None: continue

        for end_pos in piece.get_valid_moves(board, start_pos):
            is_capture = board.grid[end_pos[0]][end_pos[1]] is not None
            is_promotion = isinstance(piece, Pawn) and (end_pos[0] == 0 or end_pos[0] == ROWS - 1)
            
            if is_capture or is_promotion:
                yield (start_pos, end_pos)
                continue # Move has been yielded, no need for further checks
            
            # Check for special quiet tactical moves
            if is_rook_piercing_capture(board, (start_pos, end_pos)):
                yield (start_pos, end_pos)
            elif is_quiet_knight_evaporation(board, (start_pos, end_pos)):
                yield (start_pos, end_pos)