# gamelogic.py (v4.0 - Optimized Move Validation)
# - Replaced the inefficient 'validate_move' function with a generator-based approach.
# - The new `_generate_legal_moves` function now clones the board only ONCE per move
#   during validation, providing a major performance boost for the AI.

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

# -----------------------------
# Piece Classes
# -----------------------------
class Piece:
    def __init__(self, color):
        self.color = color
        self.has_moved = False
        self.opponent_color = "black" if color == "white" else "white"
    def clone(self):
        new_piece = self.__class__(self.color); new_piece.has_moved = self.has_moved; return new_piece
    def symbol(self): return "?"
    def get_valid_moves(self, board, pos): return []

class King(Piece):
    def symbol(self): return "♔" if self.color == "white" else "♚"
    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['king']:
            # --- Check 1-step move ---
            r1, c1 = r_start + dr, c_start + dc
            if 0 <= r1 < ROWS and 0 <= c1 < COLS:
                target1 = board.grid[r1][c1]
                if target1 is None or target1.color == self.opponent_color:
                    moves.append((r1, c1))
            
            # --- Check 2-step move (The Leap) ---
            # This is only possible if the intermediate square (r1, c1) is empty.
            # We must re-check if (r1, c1) is on the board before accessing the grid.
            if (0 <= r1 < ROWS and 0 <= c1 < COLS) and (board.grid[r1][c1] is None):
                r2, c2 = r_start + dr * 2, c_start + dc * 2
                if 0 <= r2 < ROWS and 0 <= c2 < COLS:
                    target2 = board.grid[r2][c2]
                    if target2 is None or target2.color == self.opponent_color:
                        moves.append((r2, c2))
        return moves

class Queen(Piece):
    def symbol(self): return "♕" if self.color == "white" else "♛"
    def get_valid_moves(self, board, pos):
        moves = []; r_start, c_start = pos
        for dr, dc in DIRECTIONS['queen']:
            r, c = r_start + dr, c_start + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = board.grid[r][c]
                if target is None: moves.append((r, c))
                else:
                    if target.color != self.color: moves.append((r, c))
                    break
                r += dr; c += dc
        return moves

class Rook(Piece):
    def symbol(self): return "♖" if self.color == "white" else "♜"
    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['rook']:
            r, c = r_start + dr, c_start + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = board.grid[r][c]
                if target:
                    if target.color != self.color:
                        moves.append((r, c))
                    else:
                        break
                else:
                    moves.append((r,c))
                r += dr; c += dc
        return moves

class Bishop(Piece):
    def symbol(self): return "♗" if self.color == "white" else "♝"
    def get_valid_moves(self, board, pos):
        moves = set(); r_start, c_start = pos
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
        moves = []; r_start, c_start = pos
        for dr, dc in DIRECTIONS['knight']:
            nr, nc = r_start + dr, c_start + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and (not board.grid[nr][nc] or board.grid[nr][nc].color != self.color):
                moves.append((nr,nc))
        return moves

class Pawn(Piece):
    def __init__(self, color):
        super().__init__(color)
        self.direction = -1 if self.color == "white" else 1
        self.starting_row = 6 if self.color == "white" else 1

    def symbol(self): return "♙" if self.color == "white" else "♟"
    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        one_r = r_start + self.direction
        if 0 <= one_r < ROWS:
            target = board.grid[one_r][c_start]
            if target is None or target.color == self.opponent_color: moves.append((one_r, c_start))
        if r_start == self.starting_row:
            one_step_clear = (0 <= one_r < ROWS) and (board.grid[one_r][c_start] is None)
            if one_step_clear:
                two_r = r_start + (2 * self.direction)
                if 0 <= two_r < ROWS:
                    target = board.grid[two_r][c_start]
                    if target is None or target.color == self.opponent_color: moves.append((two_r, c_start))
        for dc_offset in [-1, 1]:
            new_c = c_start + dc_offset
            if 0 <= new_c < COLS:
                target = board.grid[r_start][new_c]
                if target and target.color == self.opponent_color: moves.append((r_start, new_c))
        return moves

# ---------------------------------------------
# Board Class
# ---------------------------------------------
class Board:
    def __init__(self, setup=True):
        self.grid = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.white_king_pos = None; self.black_king_pos = None
        if setup: self._setup_initial_board()
    def _setup_initial_board(self):
        pieces = {0: [(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King), (5, Bishop), (6, Knight), (7, Rook)], 1: [(i, Pawn) for i in range(8)], 6: [(i, Pawn) for i in range(8)], 7: [(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King), (5, Bishop), (6, Knight), (7, Rook)]}
        for r, piece_list in pieces.items():
            color = "black" if r < 2 else "white"
            for c, piece_class in piece_list: self.add_piece(piece_class(color), r, c)
    def add_piece(self, piece, r, c):
        self.grid[r][c] = piece
        if isinstance(piece, King):
            if piece.color == "white": self.white_king_pos = (r, c)
            else: self.black_king_pos = (r, c)
    def remove_piece(self, r, c):
        piece = self.grid[r][c]
        if not piece: return
        pos = (r, c)
        if isinstance(piece, King):
            if piece.color == "white" and self.white_king_pos == pos: self.white_king_pos = None
            elif piece.color == "black" and self.black_king_pos == pos: self.black_king_pos = None
        self.grid[r][c] = None
    def move_piece(self, start, end):
        piece = self.grid[start[0]][start[1]]
        if not piece: return
        self.grid[start[0]][start[1]] = None
        self.grid[end[0]][end[1]] = piece
        piece.has_moved = True
        if isinstance(piece, King):
            if piece.color == "white": self.white_king_pos = end
            else: self.black_king_pos = end
    def find_king_pos(self, color): return self.white_king_pos if color == 'white' else self.black_king_pos
    def clone(self):
        new_board = Board(setup=False)
        new_board.grid = [[p.clone() if p else None for p in row] for row in self.grid]
        new_board.white_king_pos = self.white_king_pos
        new_board.black_king_pos = self.black_king_pos
        return new_board
    def make_move(self, start, end):
        moving_piece = self.grid[start[0]][start[1]]
        if not moving_piece: return
        target_piece = self.grid[end[0]][end[1]]
        is_capture = target_piece is not None
        if isinstance(moving_piece, Rook): self._apply_rook_piercing(start, end, moving_piece.color)
        if is_capture: self.remove_piece(end[0], end[1])
        self.move_piece(start, end)
        if isinstance(moving_piece, Queen) and is_capture: self._apply_queen_aoe(end, moving_piece.color)
        elif isinstance(moving_piece, Knight): self._apply_knight_aoe(end)
        elif isinstance(moving_piece, Pawn):
            promotion_rank = 0 if moving_piece.color == "white" else (ROWS - 1)
            if end[0] == promotion_rank: self.remove_piece(end[0], end[1]); self.add_piece(Queen(moving_piece.color), end[0], end[1])
        self._check_all_knight_evaporation()
    def _apply_queen_aoe(self, pos, queen_color):
        self.remove_piece(pos[0], pos[1])
        for dr, dc in ADJACENT_DIRS:
            adj_r, adj_c = pos[0] + dr, pos[1] + dc
            if 0 <= adj_r < ROWS and 0 <= adj_c < COLS:
                adj_piece = self.grid[adj_r][adj_c]
                if adj_piece and adj_piece.color != queen_color: self.remove_piece(adj_r, adj_c)
    def _apply_rook_piercing(self, start, end, rook_color):
        if start[0] == end[0]: d = (0, 1 if end[1] > start[1] else -1)
        else: d = (1 if end[0] > start[0] else -1, 0)
        cr, cc = start[0] + d[0], start[1] + d[1]
        while (cr, cc) != end:
            target = self.grid[cr][cc]
            if target and target.color != rook_color: self.remove_piece(cr, cc)
            cr += d[0]; cc += d[1]
    def _apply_knight_aoe(self, knight_pos):
        knight_instance = self.grid[knight_pos[0]][knight_pos[1]]
        if not knight_instance: return
        to_remove = []; enemy_knights_destroyed = False
        for dr, dc in DIRECTIONS['knight']:
            r, c = knight_pos[0] + dr, knight_pos[1] + dc
            if 0 <= r < ROWS and 0 <= c < COLS:
                target = self.grid[r][c]
                if target and target.color != knight_instance.color:
                    to_remove.append((r, c))
                    if isinstance(target, Knight): enemy_knights_destroyed = True
        for r,c in to_remove: self.remove_piece(r,c)
        if enemy_knights_destroyed: self.remove_piece(knight_pos[0], knight_pos[1])
    def _check_all_knight_evaporation(self):
        all_knights = []
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.grid[r][c]
                if isinstance(piece, Knight): all_knights.append(((r, c), piece))
        for pos, knight in all_knights:
            if self.grid[pos[0]][pos[1]] is knight: self._apply_knight_aoe(pos)

# ----------------------------------------------------
# Game Logic Functions
# ----------------------------------------------------
def create_initial_board(): return Board()

def generate_threat_map(board, attacking_color):
    threats = set()
    for r_start in range(ROWS):
        for c_start in range(COLS):
            piece = board.grid[r_start][c_start]
            if not piece or piece.color != attacking_color: continue
            pos = (r_start, c_start)
            base_moves = piece.get_valid_moves(board, pos)
            threats.update(base_moves)
            if isinstance(piece, Queen):
                for move in base_moves:
                    if board.grid[move[0]][move[1]] is not None:
                        for dr, dc in ADJACENT_DIRS: threats.add((move[0] + dr, move[1] + dc))
            elif isinstance(piece, Knight):
                 for move in base_moves:
                    for dr, dc in DIRECTIONS['knight']: threats.add((move[0] + dr, move[1] + dc))
    return threats

def is_in_check(board, color):
    king_pos = board.find_king_pos(color)
    if not king_pos: return True
    opponent_color = "black" if color == "white" else "white"
    return king_pos in generate_threat_map(board, opponent_color)

def _generate_legal_moves(board, color, yield_boards=False):
    """
    A generator that yields legal moves. This is the new, efficient core of move validation.
    If yield_boards is True, it yields (move, resulting_board_state).
    """
    for r in range(ROWS):
        for c in range(COLS):
            piece = board.grid[r][c]
            if piece and piece.color == color:
                start_pos = (r, c)
                for end_pos in piece.get_valid_moves(board, start_pos):
                    sim_board = board.clone()
                    sim_board.make_move(start_pos, end_pos)
                    if not is_in_check(sim_board, color):
                        if yield_boards:
                            yield (start_pos, end_pos), sim_board
                        else:
                            yield (start_pos, end_pos)

def get_all_legal_moves(board, color):
    """Returns a simple list of all legal moves (start_pos, end_pos) for a given color."""
    return list(_generate_legal_moves(board, color, yield_boards=False))

def has_legal_moves(board, color):
    """Efficiently checks if any legal moves exist for a given color."""
    try:
        next(_generate_legal_moves(board, color, yield_boards=False))
        return True
    except StopIteration:
        return False
        
def check_game_over(board, turn_color_who_just_moved):
    next_player_color = "black" if turn_color_who_just_moved == "white" else "white"
    if not has_legal_moves(board, next_player_color):
        if is_in_check(board, next_player_color): return "checkmate", turn_color_who_just_moved
        else: return "stalemate", None
    return None, None